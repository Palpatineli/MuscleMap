#!/usr/bin/env python
"""Train MuscleMap-compatible MONAI UNet checkpoints from NIfTI datasets.

The deterministic, CPU-heavy part of preprocessing is cached per case below the
dataset directory. Random augmentation and patch sampling remain online so each
epoch sees new training examples.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from gzip import BadGzipFile
from gzip import open as gzip_open
from hashlib import sha256
from importlib.resources import files as import_files
from io import BytesIO
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Hashable, Literal, cast, override

from monai.data import DataLoader, MetaTensor, list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.metrics.metric import CumulativeIterationMetric
from monai.networks.nets.unet import UNet
from monai.transforms import MapTransform
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld, SpatialPadd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, RandRotated, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.utils import set_determinism
from nibabel import Nifti1Header, load
from nibabel.filebasedimages import ImageFileError
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import zstandard as zstd

from muscle_map.mm_util import DatasetParameter, DatasetStats, ModelConfig


DATA_STATS_FILE = "data_stats.json"
CACHE_ESTIMATE_SAMPLES = 5
"""Number of cases used for the always-on refined cache-size estimate."""

CACHE_SCHEMA_VERSION = 1
CACHE_FILE_SUFFIX = ".npz.zst"
CACHE_COMPRESSION_LEVEL = 1
DEFAULT_CACHE_MAX_GB = 10.0
TRAINING_STATE_FILE = "last_training_state.pt"

AmpMode = Literal["off", "bf16", "fp16"]
RotationMode = Literal["random", "fixed"]
ArrayProducer = Callable[["Sample"], tuple[np.ndarray, np.ndarray]]
SignatureProducer = Callable[["Sample"], dict[str, Any]]
UpperEstimate = Callable[["Sample"], int]


class RemapLabelValuesd(MapTransform):
    """Map original segmentation label values to compact training class IDs."""

    def __init__(self, keys: Sequence[Hashable], id_map: Mapping[int, int | str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.id_map: dict[int, int] = {int(key): int(value) for key, value in id_map.items()}

    @override
    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        result = dict(data)
        for key in self.keys:
            label = result[key]
            if not isinstance(label, torch.Tensor):
                label = torch.as_tensor(label)
            remapped = torch.zeros_like(label, dtype=torch.long)
            for original, compact in self.id_map.items():
                remapped[label == original] = compact
            result[key] = remapped
        return result


class SqueezeLastSpatialDimd(MapTransform):
    """Convert single-slice 3D patches into 2D patches for 2D models."""

    def __init__(self, keys: Sequence[Hashable], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    @override
    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        result = dict(data)
        for key in self.keys:
            value = result[key]
            if getattr(value, "ndim", 0) >= 4 and value.shape[-1] == 1:
                result[key] = value.squeeze(-1)
        return result


@dataclass
class ArgStats:
    """Arguments for the dataset-statistics command."""

    dataset_dir: str
    output: str


@dataclass
class ArgTrain:
    """Arguments for the training command."""

    dataset_dir: str
    result_dir: str
    epochs: int = 1000
    save_every: int = 10
    spacing: tuple[float, float, float] | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    num_workers: int | None = None
    seed: int | None = None
    cache_dir: str | None = None
    cache_max_gb: float = DEFAULT_CACHE_MAX_GB
    yes: bool = False
    resume: str | None = None
    amp: AmpMode = "off"
    auto_batch_for_vram: float | None = None
    rotation_mode: RotationMode = "random"

    def __post_init__(self) -> None:
        if self.spacing is not None:
            self.spacing = cast(tuple[float, float, float], tuple(self.spacing))
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1")
        if self.cache_max_gb <= 0:
            raise ValueError("cache_max_gb must be positive")
        if self.auto_batch_for_vram is not None and self.auto_batch_for_vram <= 0:
            raise ValueError("auto_batch_for_vram must be positive")


@dataclass(frozen=True)
class Sample:
    """One image/label pair in the training dataset."""

    image: Path
    label: Path
    case_id: str


@dataclass
class PreparedCacheEntry:
    """A compressed cache entry ready for an atomic write."""

    sample: Sample
    source: dict[str, Any]
    payload: bytes
    array_bytes: int


@dataclass(frozen=True)
class CacheEstimate:
    """The conservative and sample-based cache estimates shown before preprocessing."""

    current_bytes: int
    conservative_bytes: int
    refined_bytes: int
    missing_cases: int
    sampled_cases: int


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset stats and model training."""
    parser = argparse.ArgumentParser(
        description="Train a MuscleMap-compatible MONAI UNet checkpoint from NIfTI image/label pairs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Collect dataset image spacing and size.")
    stats_parser.add_argument("-d", "--dataset_dir", required=True, type=str,
                              help="Dataset root containing imagesTr and labelsTr.")
    stats_parser.add_argument("-o", "--output", default=DATA_STATS_FILE, type=str,
                              help="Output JSON file or directory for dataset statistics.")

    train_parser = subparsers.add_parser("train", help="Train a checkpoint.")
    train_parser.add_argument("-d", "--dataset_dir", required=True, type=str,
                              help="Dataset root containing imagesTr and labelsTr.")
    train_parser.add_argument("-o", "--result_dir", required=True, type=str,
                              help="Directory containing metadata and training outputs.")
    train_parser.add_argument("--epochs", default=1000, type=int, help="Training epochs. Default: 1000.")
    train_parser.add_argument("--save_every", default=10, type=int,
                              help="Save inference and resume checkpoints every N epochs. Default: 10.")
    train_parser.add_argument("--spacing", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                              help="Override target spacing in mm, e.g. --spacing 1.0 1.0 3.0.")
    train_parser.add_argument("--batch_size", default=None, type=int,
                              help="Override metadata training.batch_size.")
    train_parser.add_argument("-l", "--learning_rate", default=None, type=float,
                              help="Override metadata training.learning_rate.")
    train_parser.add_argument("--num_workers", default=None, type=int,
                              help="Override metadata training.num_workers.")
    train_parser.add_argument("--seed", default=None, type=int, help="Override metadata training.seed.")
    train_parser.add_argument("--cache-dir", default=None, type=str,
                              help="Persistent preprocessed-cache directory. Default: <dataset>/preprocessed.")
    train_parser.add_argument("--cache-max-gb", default=DEFAULT_CACHE_MAX_GB, type=float,
                              help="Hard cache quota in decimal GB. Default: 10.")
    train_parser.add_argument("--yes", action="store_true",
                              help="Accept the cache-size estimate without an interactive prompt.")
    train_parser.add_argument("--resume", nargs="?", const="auto", default=None, type=str,
                              help="Resume from a training state file, or use 'auto' for the latest saved state.")
    train_parser.add_argument("--amp", default="off", choices=("off", "bf16", "fp16"),
                              help="Automatic mixed precision mode. Default: off.")
    train_parser.add_argument("--auto-batch-for-vram", default=None, type=float, metavar="GB",
                              help="Calibrate a fixed batch size within this decimal-GB VRAM target.")
    train_parser.add_argument("--rotation-mode", default="random", choices=("random", "fixed"),
                              help="Use fresh random rotations, or cache deterministic per-case rotations.")
    return parser


def parse_args() -> ArgStats | ArgTrain:
    """Parse command-line arguments into the relevant typed dataclass."""
    args = get_parser().parse_args()
    output_class: type[ArgStats] | type[ArgTrain]
    if args.command == "stats":
        output_class = ArgStats
    elif args.command == "train":
        output_class = ArgTrain
    else:
        raise NotImplementedError(f"unknown subcommand {args.command}")
    args_dict = vars(args)
    args_dict.pop("command")
    return output_class(**args_dict)


def _discover_cases(dataset_dir: Path) -> list[Sample]:
    """Return all one-channel NIfTI image/label pairs below a dataset root."""
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labelsTr directory: {labels_dir}")

    cases: list[Sample] = []
    for image_path in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = image_path.name[:-12]
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for case '{case_id}': {label_path}")
        cases.append(Sample(image=image_path, label=label_path, case_id=case_id))

    if not cases:
        raise ValueError(f"No cases found in {images_dir}; expected files ending in _0000.nii.gz.")
    return cases


def _collect_dataset_stats(dataset_dir: Path) -> DatasetStats:
    """Collect image shapes and voxel spacing without loading NIfTI payloads."""
    cases = _discover_cases(dataset_dir)
    spacings: list[list[float]] = []
    sizes: list[list[int]] = []
    for case in tqdm(cases, desc="Collecting dataset statistics"):
        image = load(case.image)
        header = cast(Nifti1Header, image.header)
        sizes.append([int(value) for value in header.get_data_shape()[:3]])
        spacings.append([float(value) for value in header.get_zooms()[:3]])
    return DatasetStats(dataset_dir=dataset_dir.resolve(), num_cases=len(cases), spacings=spacings, sizes=sizes)


def _verify_nifti_file(path: Path) -> None:
    """Force a NIfTI payload read so truncated compressed files fail before training."""
    try:
        image = load(path)
        if path.name.endswith(".gz"):
            with gzip_open(path, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
        else:
            data = np.asanyarray(cast(Any, image).dataobj, order="C")
            del data
    except (BadGzipFile, EOFError, ImageFileError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"Unreadable NIfTI file '{path}': {exc}") from exc


def _verify_training_cases(cases: Sequence[Sample]) -> None:
    """Validate every image and label before expensive preprocessing starts."""
    for case in tqdm(cases, desc="Verifying NIfTI files"):
        try:
            _verify_nifti_file(case.image)
            _verify_nifti_file(case.label)
        except RuntimeError as exc:
            raise RuntimeError(f"Invalid training case '{case.case_id}'. {exc}") from exc


def _build_config(
    data_parameter: DatasetParameter,
    dataset_stats: DatasetStats,
    spacing_override: tuple[float, float, float] | None,
) -> ModelConfig:
    """Build a training config with a rounded median or explicit target spacing."""
    default_config_file = import_files("muscle_map") / "model_config_default.json"
    config = ModelConfig.load_config(default_config_file, dataset=data_parameter.to_dict())  # pyright: ignore[reportUnknownMemberType]
    config.architecture.in_channels = len(data_parameter.channel_names)
    if spacing_override is None:
        median_spacing = np.median(np.asarray(dataset_stats.spacings, dtype=np.float64), axis=0)
        config.image.spacing = cast(
            tuple[float, float, float],
            tuple(float(round(value, 1)) for value in median_spacing),
        )
    else:
        config.image.spacing = cast(tuple[float, float, float], tuple(float(value) for value in spacing_override))
    return config


def _make_preprocess_transforms(config: ModelConfig, original_to_compact: Mapping[int, int]) -> Compose:
    """Create the deterministic transform prefix persisted in the case cache."""
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config.image.spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=20),
        EnsureTyped(keys=["image", "label"]),
        RemapLabelValuesd(keys=["label"], id_map=original_to_compact),
    ])


def _make_training_transforms(config: ModelConfig, rotation_mode: RotationMode) -> Compose:
    """Create per-epoch stochastic augmentation and patch sampling transforms."""
    transforms: list[MapTransform] = []
    if rotation_mode == "random" and config.training.rotation_degrees > 0:
        radians = float(np.deg2rad(config.training.rotation_degrees))
        transforms.append(
            RandRotated(
                keys=["image", "label"],
                range_x=radians,
                range_y=radians,
                range_z=radians,
                prob=1.0,
                mode=("bilinear", "nearest"),
                padding_mode="border",
            )
        )
    transforms.extend([
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=config.image.train_patch_size,
            method="end",
            mode="constant",
        ),
        EnsureTyped(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config.image.train_patch_size,
            pos=1,
            neg=1,
            num_samples=config.training.samples_per_volume,
            image_key="image",
            image_threshold=0,
        ),
    ])
    if config.architecture.spatial_dims == 2:
        transforms.append(SqueezeLastSpatialDimd(keys=["image", "label"]))
    return Compose(transforms)


def _to_numpy(value: Any) -> np.ndarray:
    """Materialize a MONAI tensor-like value as a detached NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, value: Any) -> None:
    """Atomically write human-readable JSON with Unix line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temp_path.replace(path)


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class CaseCache:
    """Filesystem-visible, compressed cache for deterministic per-case arrays."""

    def __init__(
        self,
        root: Path,
        quota_root: Path,
        specification: Mapping[str, Any],
        source_signature: SignatureProducer,
        array_producer: ArrayProducer,
        upper_estimate: UpperEstimate,
    ) -> None:
        self.specification: dict[str, Any] = dict(specification)
        self.fingerprint: str = _fingerprint(self.specification)
        self.root: Path = root / self.fingerprint
        self.quota_root: Path = quota_root
        self.source_signature: SignatureProducer = source_signature
        self.array_producer: ArrayProducer = array_producer
        self.upper_estimate: UpperEstimate = upper_estimate
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            _write_json(manifest_path, {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "compression": {"algorithm": "zstd", "level": CACHE_COMPRESSION_LEVEL},
                "preprocessing": self.specification,
            })

    def data_path(self, sample: Sample) -> Path:
        return self.root / f"{sample.case_id}{CACHE_FILE_SUFFIX}"

    def metadata_path(self, sample: Sample) -> Path:
        return self.root / f"{sample.case_id}.json"

    def _read_metadata(self, sample: Sample) -> dict[str, Any] | None:
        path = self.metadata_path(sample)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as stream:
                return cast(dict[str, Any], json.load(stream))
        except (OSError, json.JSONDecodeError):
            return None

    def is_current(self, sample: Sample) -> bool:
        """Return whether a case cache file matches its current source files."""
        if not self.data_path(sample).is_file():
            return False
        metadata = self._read_metadata(sample)
        return metadata is not None and metadata.get("source") == self.source_signature(sample)

    def missing_cases(self, cases: Sequence[Sample]) -> list[Sample]:
        return [sample for sample in cases if not self.is_current(sample)]

    def _encode(self, image: np.ndarray, label: np.ndarray) -> tuple[bytes, int]:
        image_array = np.ascontiguousarray(image, dtype=np.float32)
        if np.any(label < 0):
            raise ValueError("Training labels must be non-negative after remapping.")
        label_dtype = np.uint16 if int(np.max(label, initial=0)) <= np.iinfo(np.uint16).max else np.uint32
        label_array = np.ascontiguousarray(label, dtype=label_dtype)
        stream = BytesIO()
        np.savez(stream, image=image_array, label=label_array)
        raw = stream.getvalue()
        payload = zstd.ZstdCompressor(level=CACHE_COMPRESSION_LEVEL).compress(raw)
        return payload, image_array.nbytes + label_array.nbytes

    def prepare(self, sample: Sample) -> PreparedCacheEntry:
        """Run preprocessing for one case without modifying the cache."""
        image, label = self.array_producer(sample)
        payload, array_bytes = self._encode(image, label)
        return PreparedCacheEntry(
            sample=sample,
            source=self.source_signature(sample),
            payload=payload,
            array_bytes=array_bytes,
        )

    def quota_bytes(self) -> int:
        """Return compressed cache bytes in the configured cache namespace."""
        if not self.quota_root.exists():
            return 0
        return sum(path.stat().st_size for path in self.quota_root.rglob(f"*{CACHE_FILE_SUFFIX}") if path.is_file())

    def old_entry_size(self, sample: Sample) -> int:
        path = self.data_path(sample)
        return path.stat().st_size if path.exists() else 0

    def write(self, entry: PreparedCacheEntry, max_bytes: int) -> None:
        """Atomically write one entry while enforcing the global cache quota."""
        projected_size = self.quota_bytes() - self.old_entry_size(entry.sample) + len(entry.payload)
        if projected_size > max_bytes:
            message = f"Writing '{entry.sample.case_id}' would exceed the cache quota: {_format_bytes(projected_size)} > {_format_bytes(max_bytes)}."
            raise RuntimeError(message)

        data_path = self.data_path(entry.sample)
        temp_path = data_path.with_name(f".{data_path.name}.tmp")
        with temp_path.open("wb") as stream:
            stream.write(entry.payload)
        temp_path.replace(data_path)
        _write_json(self.metadata_path(entry.sample), {
            "case_id": entry.sample.case_id,
            "fingerprint": self.fingerprint,
            "source": entry.source,
            "compressed_bytes": len(entry.payload),
            "array_bytes": entry.array_bytes,
        })

    def read(self, sample: Sample) -> dict[str, np.ndarray]:
        """Load one cached image/label pair as independent contiguous arrays."""
        path = self.data_path(sample)
        if not self.is_current(sample):
            raise RuntimeError(f"Cache entry is missing or stale for case '{sample.case_id}': {path}")
        compressed = path.read_bytes()
        raw = zstd.ZstdDecompressor().decompress(compressed)
        with np.load(BytesIO(raw), allow_pickle=False) as arrays:
            image = np.ascontiguousarray(arrays["image"])
            label = np.ascontiguousarray(arrays["label"])
        return {"image": image, "label": label}

    def entry_array_bytes(self, sample: Sample) -> int:
        metadata = self._read_metadata(sample)
        if metadata is None:
            raise RuntimeError(f"Missing cache metadata for '{sample.case_id}'.")
        return int(metadata["array_bytes"])


class CachedTrainingDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Read cached base volumes and apply stochastic training transforms on demand."""

    def __init__(self, cache: CaseCache, cases: Sequence[Sample], transform: Compose) -> None:
        self.cache: CaseCache = cache
        self.cases: list[Sample] = list(cases)
        self.transform: Compose = transform

    def __len__(self) -> int:
        return len(self.cases)

    @override
    def __getitem__(self, index: int) -> Any:
        return self.transform(self.cache.read(self.cases[index]))


def _format_bytes(num_bytes: int | float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1000.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1000.0
    raise AssertionError("unreachable")


def _estimate_resampled_case_bytes(sample: Sample, target_spacing: tuple[float, float, float]) -> int:
    """Return a fast full-volume upper bound before foreground cropping and compression."""
    image = load(sample.image)
    header = cast(Nifti1Header, image.header)
    source_shape = np.asarray(header.get_data_shape()[:3], dtype=np.float64)
    source_spacing = np.asarray(header.get_zooms()[:3], dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    if np.any(target <= 0):
        raise ValueError(f"Target spacing must be positive, got {target_spacing}.")
    resampled_shape = np.maximum(1, np.ceil(source_shape * source_spacing / target).astype(np.int64))
    voxels = int(np.prod(resampled_shape))
    # A float32 image plus a uint16 compact label, one percent container margin, and 1 MiB metadata slack.
    return int((voxels * (np.dtype(np.float32).itemsize + np.dtype(np.uint16).itemsize)) * 1.01) + 1024 * 1024


def _confirm_cache_estimate(estimate: CacheEstimate, max_bytes: int, assume_yes: bool) -> None:
    logging.info(
        "Cache estimate for %s missing cases: conservative=%s, refined from %s samples=%s, current=%s, quota=%s.",
        estimate.missing_cases,
        _format_bytes(estimate.conservative_bytes),
        estimate.sampled_cases,
        _format_bytes(estimate.refined_bytes),
        _format_bytes(estimate.current_bytes),
        _format_bytes(max_bytes),
    )
    if estimate.conservative_bytes <= max_bytes:
        logging.info("The conservative header-only estimate fits within the configured cache quota.")
    else:
        logging.warning("The conservative estimate exceeds the quota; the refined estimate is advisory only.")

    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Cache preprocessing needs confirmation in a non-interactive shell. Re-run with --yes.")
    response = input("Build missing preprocessing cache entries? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        raise RuntimeError("Cache preprocessing cancelled by user.")


def _ensure_case_cache(
    cache: CaseCache,
    cases: Sequence[Sample],
    max_bytes: int,
    assume_yes: bool,
) -> None:
    """Estimate, confirm, then build only missing or stale per-case entries."""
    pending = cache.missing_cases(cases)
    if not pending:
        logging.info("All %s preprocessing cache entries are current in %s.", len(cases), cache.root)
        return

    current_bytes = cache.quota_bytes()
    old_pending_bytes = sum(cache.old_entry_size(sample) for sample in pending)
    upper_by_case = {sample.case_id: cache.upper_estimate(sample) for sample in pending}
    sample_cases = pending[:min(CACHE_ESTIMATE_SAMPLES, len(pending))]
    logging.info("Preparing %s representative cache-estimate samples.", len(sample_cases))
    prepared_samples = [cache.prepare(sample) for sample in sample_cases]
    sampled_upper = sum(upper_by_case[entry.sample.case_id] for entry in prepared_samples)
    sampled_compressed = sum(len(entry.payload) for entry in prepared_samples)
    compression_ratio = sampled_compressed / max(sampled_upper, 1)
    remaining_upper = sum(upper_by_case[sample.case_id] for sample in pending[len(sample_cases):])
    conservative = current_bytes - old_pending_bytes + sum(upper_by_case.values())
    refined = current_bytes - old_pending_bytes + sampled_compressed + int(remaining_upper * compression_ratio)
    estimate = CacheEstimate(
        current_bytes=current_bytes,
        conservative_bytes=conservative,
        refined_bytes=refined,
        missing_cases=len(pending),
        sampled_cases=len(sample_cases),
    )
    _confirm_cache_estimate(estimate, max_bytes, assume_yes)

    prepared_by_case = {entry.sample.case_id: entry for entry in prepared_samples}
    for sample in tqdm(pending, desc="Caching deterministic preprocessing"):
        entry = prepared_by_case.get(sample.case_id)
        if entry is None:
            entry = cache.prepare(sample)
        cache.write(entry, max_bytes)


def _cache_specification(config: ModelConfig, original_to_compact: Mapping[int, int]) -> dict[str, Any]:
    """Return every setting that affects the deterministic cache payload."""
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "orientation": "RAS",
        "spacing": list(config.image.spacing),
        "normalize": {"nonzero": True},
        "foreground_crop": {"source_key": "image", "margin": 20},
        "label_map": {str(key): value for key, value in sorted(original_to_compact.items())},
    }


def _build_base_cache(
    dataset_dir: Path,
    cache_dir: Path,
    config: ModelConfig,
    original_to_compact: Mapping[int, int],
) -> CaseCache:
    """Create the deterministic preprocessing cache for the current configuration."""
    transform = _make_preprocess_transforms(config, original_to_compact)

    def source_signature(sample: Sample) -> dict[str, Any]:
        return {"image": _file_signature(sample.image), "label": _file_signature(sample.label)}

    def produce(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
        transformed = cast(
            Mapping[str, Any],
            transform(cast(Any, {"image": str(sample.image), "label": str(sample.label)})),
        )
        return _to_numpy(transformed["image"]), _to_numpy(transformed["label"])

    def upper_estimate(sample: Sample) -> int:
        return _estimate_resampled_case_bytes(sample, config.image.spacing)

    return CaseCache(
        root=cache_dir,
        quota_root=dataset_dir,
        specification=_cache_specification(config, original_to_compact),
        source_signature=source_signature,
        array_producer=produce,
        upper_estimate=upper_estimate,
    )


def _fixed_rotation_seed(seed: int, case_id: str) -> int:
    value = sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], byteorder="little") % (2**32)


def _build_fixed_rotation_cache(
    dataset_dir: Path,
    cache_dir: Path,
    base_cache: CaseCache,
    config: ModelConfig,
    seed: int,
) -> CaseCache:
    """Create a separate cache of deterministic, per-case random rotations."""
    training_root = cache_dir.parent / "training_cache"
    radians = float(np.deg2rad(config.training.rotation_degrees))

    def source_signature(sample: Sample) -> dict[str, Any]:
        base_path = base_cache.data_path(sample)
        return {"base_cache": _file_signature(base_path), "base_fingerprint": base_cache.fingerprint}

    def produce(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
        data = base_cache.read(sample)
        if config.training.rotation_degrees <= 0:
            return data["image"], data["label"]
        transform = RandRotated(
            keys=["image", "label"],
            range_x=radians,
            range_y=radians,
            range_z=radians,
            prob=1.0,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        )
        transform.set_random_state(seed=_fixed_rotation_seed(seed, sample.case_id))
        rotated = transform(cast(Any, data))
        return _to_numpy(rotated["image"]), _to_numpy(rotated["label"])

    def upper_estimate(sample: Sample) -> int:
        # keep_size=True means rotation cannot exceed its base cache array size.
        return int(base_cache.entry_array_bytes(sample) * 1.01) + 1024 * 1024

    specification = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "base_fingerprint": base_cache.fingerprint,
        "rotation_mode": "fixed",
        "rotation_degrees": config.training.rotation_degrees,
        "seed": seed,
    }
    return CaseCache(
        root=training_root,
        quota_root=dataset_dir,
        specification=specification,
        source_signature=source_signature,
        array_producer=produce,
        upper_estimate=upper_estimate,
    )


def _write_training_namespace(cache_dir: Path, base_fingerprint: str, rotation_mode: RotationMode, seed: int) -> None:
    """Record the random-mode namespace even though random rotations are not persisted."""
    root = cache_dir.parent / "training_cache" / f"{base_fingerprint}-{rotation_mode}"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "manifest.json", {
        "base_fingerprint": base_fingerprint,
        "rotation_mode": rotation_mode,
        "seed": seed,
        "persisted": rotation_mode == "fixed",
    })


def _make_loader(
    dataset: CachedTrainingDataset,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build a fixed-shape training loader with persistent worker processes."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    effective_batch_size = min(batch_size, len(dataset))
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        drop_last=len(dataset) >= effective_batch_size,
        generator=generator,
        **worker_kwargs,
    )


def _create_model(config: ModelConfig, out_channels: int, device: torch.device) -> UNet:
    """Create an eager UNet; callers compile it only after any checkpoint load."""
    return UNet(out_channels=out_channels, **config.architecture.to_dict()).to(device)


def _autocast_context(amp_mode: AmpMode):
    if amp_mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """Remove MONAI metadata before a tensor enters a compiled model."""
    return value.as_tensor() if isinstance(value, MetaTensor) else value


def _calibration_step(
    dataset: CachedTrainingDataset,
    volume_batch_size: int,
    config: ModelConfig,
    out_channels: int,
    learning_rate: float,
    device: torch.device,
    amp_mode: AmpMode,
) -> tuple[bool, int]:
    """Run a compiled representative training step and return fit status and peak VRAM."""
    generator = torch.Generator().manual_seed(0)
    loader = _make_loader(dataset, volume_batch_size, num_workers=0, generator=generator, shuffle=False)
    batch = next(iter(loader))
    eager_model: UNet | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        torch.cuda.empty_cache()
        eager_model = _create_model(config, out_channels, device)
        optimizer = torch.optim.AdamW(
            eager_model.parameters(),
            lr=learning_rate,
            weight_decay=float(config.training.weight_decay),
        )
        model = cast(torch.nn.Module, torch.compile(eager_model, dynamic=False))  # pyright: ignore[reportUnknownMemberType]
        scaler = torch.amp.GradScaler("cuda", enabled=amp_mode == "fp16")
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        images = _plain_tensor(batch["image"]).to(device, non_blocking=True)
        labels = _plain_tensor(batch["label"]).to(device, non_blocking=True).long()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_mode):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        return True, int(torch.cuda.max_memory_allocated(device))
    except torch.OutOfMemoryError:
        return False, 0
    finally:
        del batch
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        if eager_model is not None:
            del eager_model
        torch.cuda.empty_cache()


def _auto_batch_size(
    dataset: CachedTrainingDataset,
    config: ModelConfig,
    out_channels: int,
    learning_rate: float,
    device: torch.device,
    amp_mode: AmpMode,
    target_gb: float,
) -> int:
    """Find the largest fixed volume batch with a conservative VRAM headroom."""
    target_bytes = int(target_gb * 1_000_000_000)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    if target_bytes > free_bytes:
        message = f"Requested --auto-batch-for-vram {_format_bytes(target_bytes)} but only {_format_bytes(free_bytes)} is currently free on {device}."
        raise RuntimeError(message)
    usable_bytes = int(target_bytes * 0.9)
    max_candidate = len(dataset)
    best = 0
    candidate = 1
    first_too_large = max_candidate + 1
    while candidate <= max_candidate:
        fits, peak = _calibration_step(
            dataset, candidate, config, out_channels, learning_rate, device, amp_mode,
        )
        logging.info(
            "VRAM calibration: volume_batch_size=%s, peak=%s, target=%s, fits=%s.",
            candidate,
            _format_bytes(peak),
            _format_bytes(usable_bytes),
            fits and peak <= usable_bytes,
        )
        if not fits or peak > usable_bytes:
            first_too_large = candidate
            break
        best = candidate
        candidate *= 2

    if best == 0:
        raise RuntimeError(
            f"One training volume does not fit within the conservative target of {_format_bytes(usable_bytes)}."
        )

    low = best + 1
    high = min(first_too_large - 1, max_candidate)
    if first_too_large == max_candidate + 1:
        high = max_candidate
    while low <= high:
        candidate = (low + high) // 2
        fits, peak = _calibration_step(
            dataset, candidate, config, out_channels, learning_rate, device, amp_mode,
        )
        if fits and peak <= usable_bytes:
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    logging.info("Selected calibrated volume batch size %s.", best)
    return best


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module from a torch.compile wrapper when present."""
    return cast(torch.nn.Module, getattr(model, "_orig_mod", model))


def _save_checkpoint(model: torch.nn.Module, config: ModelConfig, output_path: Path) -> Path:
    """Save inference-compatible, uncompiled model weights and configuration."""
    pth_path = output_path.with_suffix(".pth")
    json_path = output_path.with_suffix(".json")
    torch.save(_unwrap_model(model).state_dict(), pth_path)
    _write_json(json_path, config.to_dict())
    return pth_path


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _save_training_state(
    output_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: ModelConfig,
    cache_fingerprint: str,
    loader_generator: torch.Generator,
    amp_mode: AmpMode,
) -> None:
    """Persist all state needed to continue training from a scheduled checkpoint."""
    state = {
        "epoch": epoch,
        "model_state": _unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": _capture_rng_state(),
        "loader_generator_state": loader_generator.get_state(),
        "config": config.to_dict(),
        "cache_fingerprint": cache_fingerprint,
        "amp": amp_mode,
    }
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(state, temp_path)
    temp_path.replace(output_path)


def _load_training_state(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         scaler: torch.amp.GradScaler, loader_generator: torch.Generator) -> int:
    """Load a previous training state into eager model and optimizer objects."""
    state = torch.load(path, map_location="cuda", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Training state '{path}' is not a valid checkpoint dictionary.")
    model.load_state_dict(cast(Mapping[str, Any], state["model_state"]))
    optimizer.load_state_dict(cast(dict[str, Any], state["optimizer_state"]))
    scaler.load_state_dict(cast(dict[str, Any], state["scaler_state"]))
    loader_generator.set_state(cast(torch.Tensor, state["loader_generator_state"]))
    _restore_rng_state(cast(Mapping[str, Any], state["rng_state"]))
    return int(cast(int, state["epoch"]))


def _resolve_resume_path(resume: str | None, result_dir: Path) -> Path | None:
    if resume is None:
        return None
    path = result_dir / TRAINING_STATE_FILE if resume == "auto" else Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"Training resume state not found: {path}")
    return path


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    dice_metric: CumulativeIterationMetric,
    amp_mode: AmpMode,
) -> tuple[float, float]:
    """Evaluate the final model against sampled training patches for a smoke metric."""
    model.eval()
    validation_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            images = _plain_tensor(batch["image"]).to(device, non_blocking=True)
            labels = _plain_tensor(batch["label"]).to(device, non_blocking=True).long()
            with _autocast_context(amp_mode):
                logits = model(images)
                validation_loss += loss_fn(logits, labels).item()
            predictions = torch.argmax(logits, dim=1, keepdim=True)
            out_channels = logits.shape[1]
            predictions_onehot = F.one_hot(predictions.squeeze(1), num_classes=out_channels)
            labels_onehot = F.one_hot(labels.squeeze(1), num_classes=out_channels)
            channel_dim = predictions_onehot.ndim - 1
            dice_metric(
                y_pred=predictions_onehot.movedim(channel_dim, 1).float(),
                y=labels_onehot.movedim(channel_dim, 1).float(),
            )
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return validation_loss / max(len(loader), 1), mean_dice


def _run_stats(args: ArgStats) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    stats = _collect_dataset_stats(Path(args.dataset_dir))
    output = Path(args.output)
    if output.suffix.lower() != ".json":
        output.mkdir(exist_ok=True, parents=True)
        output /= DATA_STATS_FILE
    else:
        output.parent.mkdir(exist_ok=True, parents=True)
    _write_json(output, stats.to_dict())
    logging.info("Wrote dataset statistics to %s.", output)


def _run_training(args: ArgTrain) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA or ROCm. CPU is supported only for inference.")
    device = torch.device("cuda")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s] %(message)s",
        handlers=[logging.FileHandler(result_dir / "training.log", mode="a"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Training on %s", torch.cuda.get_device_name(device))

    dataset_stats = DatasetStats.load_config(result_dir / DATA_STATS_FILE)  # pyright: ignore[reportUnknownMemberType]
    dataset_parameter = DatasetParameter.load_config(result_dir / "dataset_parameter.json")  # pyright: ignore[reportUnknownMemberType]
    config = _build_config(dataset_parameter, dataset_stats, args.spacing)
    _write_json(result_dir / "training_config.json", config.to_dict())

    seed = args.seed if args.seed is not None else config.training.seed
    set_determinism(seed=seed)
    labels = sorted({value for value in config.dataset.labels.values() if value > 0})
    original_to_compact = {0: 0}
    for compact_id, original_id in enumerate(labels, start=1):
        original_to_compact[original_id] = compact_id

    cases = _discover_cases(Path(args.dataset_dir))
    logging.info("Discovered %s training cases.", len(cases))
    if config.training.verify_nifti_files:
        _verify_training_cases(cases)

    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else Path(args.dataset_dir) / "preprocessed"
    cache_dir = cache_dir.resolve()
    quota_bytes = int(args.cache_max_gb * 1_000_000_000)
    base_cache = _build_base_cache(Path(args.dataset_dir).resolve(), cache_dir, config, original_to_compact)
    _ensure_case_cache(base_cache, cases, quota_bytes, args.yes)
    _write_training_namespace(cache_dir, base_cache.fingerprint, args.rotation_mode, seed)

    active_cache = base_cache
    if args.rotation_mode == "fixed":
        rotation_cache = _build_fixed_rotation_cache(Path(args.dataset_dir).resolve(), cache_dir, base_cache, config, seed)
        _ensure_case_cache(rotation_cache, cases, quota_bytes, args.yes)
        active_cache = rotation_cache

    train_dataset = CachedTrainingDataset(active_cache, cases, _make_training_transforms(config, args.rotation_mode))
    out_channels = len(labels) + 1
    learning_rate = args.learning_rate if args.learning_rate is not None else float(config.training.learning_rate)
    num_workers = args.num_workers if args.num_workers is not None else config.training.num_workers
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
    if args.auto_batch_for_vram is not None:
        batch_size = _auto_batch_size(
            train_dataset,
            config,
            out_channels,
            learning_rate,
            device,
            args.amp,
            args.auto_batch_for_vram,
        )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = _make_loader(train_dataset, batch_size, num_workers, loader_generator, shuffle=True)
    eager_model = _create_model(config, out_channels, device)
    optimizer = torch.optim.AdamW(
        eager_model.parameters(),
        lr=learning_rate,
        weight_decay=float(config.training.weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    start_epoch = 1
    resume_path = _resolve_resume_path(args.resume, result_dir)
    if resume_path is not None:
        completed_epoch = _load_training_state(resume_path, eager_model, optimizer, scaler, loader_generator)
        start_epoch = completed_epoch + 1
        logging.info("Resuming from %s at epoch %s.", resume_path, start_epoch)

    # Compile only after loading weights, so saved checkpoints retain unwrapped key names.
    model = cast(torch.nn.Module, torch.compile(eager_model, dynamic=False))  # pyright: ignore[reportUnknownMemberType]
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    logging.info("Using volume batch size %s (%s patches per source volume).", batch_size, config.training.samples_per_volume)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = _plain_tensor(batch["image"]).to(device, non_blocking=True)
            labels_tensor = _plain_tensor(batch["label"]).to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(args.amp):
                logits = model(images)
                loss = loss_fn(logits, labels_tensor)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        logging.info("Epoch %s/%s - train_loss=%.5f", epoch, args.epochs, epoch_loss)
        if epoch % args.save_every == 0:
            checkpoint = _save_checkpoint(model, config, result_dir / f"checkpoint_epoch_{epoch:04d}")
            _save_training_state(
                result_dir / TRAINING_STATE_FILE,
                model,
                optimizer,
                scaler,
                epoch,
                config,
                active_cache.fingerprint,
                loader_generator,
                args.amp,
            )
            logging.info("Saved checkpoint: %s", checkpoint)

    final_path = _save_checkpoint(model, config, result_dir / "final_model")
    _save_training_state(
        result_dir / TRAINING_STATE_FILE,
        model,
        optimizer,
        scaler,
        args.epochs,
        config,
        active_cache.fingerprint,
        loader_generator,
        args.amp,
    )
    logging.info("Training complete. Final checkpoint: %s", final_path)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    mean_loss, mean_dice = _run_validation(model, train_loader, device, loss_fn, dice_metric, args.amp)
    logging.info("Training loss: %.5f. Pseudo dice: %.5f", mean_loss, mean_dice)


def main() -> None:
    """Dispatch the selected training command."""
    args = parse_args()
    if isinstance(args, ArgStats):
        _run_stats(args)
    else:
        _run_training(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        sys.exit(1)
