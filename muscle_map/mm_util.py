from importlib.abc import Traversable
from typing import Any, Literal, NotRequired, Self, TypedDict, cast
import os
import logging
import sys
import json
import math
from dataclasses import dataclass, field
import hashlib
import tempfile
import urllib.request
import numpy as np
import numpy.typing as npt
from nibabel import save, load, Nifti1Header, Nifti1Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from monai.transforms.transform import MapTransform
import gc
import torch
import shutil
import psutil
from scipy import ndimage as ndi
from mashumaro.mixins.json import DataClassJSONMixin
from pathlib import Path
from tqdm import tqdm

_MODELS_DIR = Path(__file__).parent / "models"
_TEMPLATES_DIR = Path(__file__).parent / "templates"

class JSONLoaderMixin(DataClassJSONMixin):
    @classmethod
    def load_config(cls, json_path: Path | Traversable, **kwargs) -> Self:
        with json_path.open() as f:
            json_data = json.load(f)
        for key, value in kwargs.items():
            json_data[key] = value
        return cls.from_dict(json_data)  # pyright: ignore[reportUnknownMemberType]

@dataclass
class DatasetParameter(JSONLoaderMixin):
    # compatible with the dataset.json file from nnunetv2
    description: str
    labels: dict[str, int]
    name: str
    numTraining: int
    reference: str
    release: str
    channel_names: dict[str, str]
    file_ending: str

@dataclass
class ArchitectureConfig(JSONLoaderMixin):
    # minimal config for monai unet
    spatial_dims: Literal[2, 3]
    in_channels: int
    channels: list[int]
    act: str
    strides: list[int]
    num_res_units: int
    norm: str

@dataclass
class ImageParameter:
    roi_size: list[int]
    spatial_window_batch_size: int
    train_patch_size: list[int]
    spacing: tuple[float, float, float]

@dataclass
class TrainParameter:
    batch_size: int
    samples_per_volume: int
    num_workers: int
    learning_rate: float = field(default=1E-4)
    weight_decay: float = field(default=1E-5)
    seed: int = field(default=2026)
    rotation_degrees: int = field(default=15)

@dataclass
class ModelConfig(JSONLoaderMixin):
    architecture: ArchitectureConfig
    image: ImageParameter
    training: TrainParameter
    dataset: DatasetParameter

@dataclass
class DatasetStats(JSONLoaderMixin):
    dataset_dir: Path
    num_cases: int
    spacings: list[list[float]]
    sizes: list[list[int]]

# concept_id: the Zenodo "concept record ID" that always points to all versions.
# Fill these in after publishing each model on Zenodo.
class ModelMeta(TypedDict):
    record_id: str
    versions: NotRequired[dict[str, str]]
    pth_filename: str
    json_filename: str

ZENODO_MODELS: dict[str, ModelMeta] = {
    "abdomen": {
        "record_id": "19631081",
        "pth_filename": "contrast_agnostic_abdomen_model.pth",
        "json_filename": "contrast_agnostic_abdomen_model.json",
    },
    "forearm": {
        "record_id": "19633115",
        "pth_filename": "contrast_agnostic_forearm_model.pth",
        "json_filename": "contrast_agnostic_forearm_model.json",
    },
    "leg": {
        "record_id": "19633057",
        "pth_filename": "contrast_agnostic_leg_model.pth",
        "json_filename": "contrast_agnostic_leg_model.json",
    },
    "pelvis": {
        "record_id": "19632902",
        "pth_filename": "contrast_agnostic_pelvis_model.pth",
        "json_filename": "contrast_agnostic_pelvis_model.json",
    },
    "thigh": {
        "record_id": "19633000",
        "pth_filename": "contrast_agnostic_thigh_model.pth",
        "json_filename": "contrast_agnostic_thigh_model.json",
    },
    "wholebody": {
        "record_id": "19631184",  # concept DOI — always resolves to latest
        "versions": {
            "1.0": "19631185",
            "1.1": "19976722",
            "1.2": "19976860",
            "1.3": "19976940",
        },
        "pth_filename": "contrast_agnostic_wholebody_model.pth",
        "json_filename": "contrast_agnostic_wholebody_model.json",
    },
}




def _get_model_cache_dir(region: str, version: str) -> Path:
    return _MODELS_DIR / region / f"v{version}"


def _latest_cached_version(region: str, pth_filename: str, json_filename: str) -> Path | None:
    """Return the cache dir of the most recent locally cached version, or None."""
    region_dir = _MODELS_DIR / region
    if not region_dir.is_dir():
        return None
    candidates: list[Path] = []
    for d in region_dir.iterdir():
        if d.is_dir() and (d / pth_filename).exists() and (d / json_filename).exists():
            candidates.append(d)
    if not candidates:
        return None
    # Sort by folder name (v0.0, v1.0, v1.3 …) — lexicographic is fine for semver with leading v
    return sorted(candidates, key=lambda p: p.name)[-1]


def _zenodo_get(url: str) -> dict[str, Any]:
    """GET a Zenodo API URL and return parsed JSON, or raise ConnectionError."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise ConnectionError(f"Zenodo API request failed ({url}): {type(e).__name__}: {e}") from e


def _fetch_zenodo_latest(record_id: str) -> tuple[str, dict[str, str]]:
    """Fetch the latest version from Zenodo. Returns (resolved_version, {filename: url})."""
    data = _zenodo_get(f"https://zenodo.org/api/records/{record_id}/versions/latest")
    resolved_version = data.get("metadata", {}).get("version", "unknown")
    file_urls = {f["key"]: f["links"]["self"] for f in data.get("files", [])}
    return resolved_version, file_urls


def _fetch_zenodo_version(version_record_id: str) -> tuple[str, dict[str, str]]:
    """Fetch a specific version record from Zenodo. Returns (version, {filename: url})."""
    data = _zenodo_get(f"https://zenodo.org/api/records/{version_record_id}")
    resolved_version = data.get("metadata", {}).get("version", "unknown")
    file_urls = {f["key"]: f["links"]["self"] for f in data.get("files", [])}
    return resolved_version, file_urls


def _verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected


class _DownloadProgress(tqdm):  # pyright: ignore[reportMissingTypeArgument]
    def update_to(self, block_count=1, block_size=1, total=-1):
        if total >= 0:
            self.total: int = total
        self.update(block_count * block_size - self.n)


def _download_file(url: str, dest: Path, sha256: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with _DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as progress:
            urllib.request.urlretrieve(url, tmp_path, reporthook=progress.update_to)
        if sha256:
            if not _verify_sha256(tmp_path, sha256):
                tmp_path.unlink(missing_ok=True)
                logging.error(f"SHA256 mismatch for '{dest.name}'. Download may be corrupt.")
                sys.exit(1)
        tmp_path.rename(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def check_for_model_update(region: str) -> tuple[str | None, str | None]:
    """
    Check whether a newer model version is available on Zenodo.
    Returns (cached_version_or_None, zenodo_version_or_None).
    Does not download anything.
    """
    model_info = ZENODO_MODELS.get(region)
    if not model_info:
        return None, None
    record_id = model_info["record_id"]
    pth_filename = model_info["pth_filename"]
    json_filename = model_info["json_filename"]
    cached_dir = _latest_cached_version(region, pth_filename, json_filename)
    cached_v = cached_dir.name[1:] if cached_dir else None  # strip leading 'v'
    try:
        zenodo_v, _ = _fetch_zenodo_latest(record_id)
    except ConnectionError:
        return cached_v, None
    return cached_v, zenodo_v


def ensure_model_downloaded(region: str, version: str = "latest") -> tuple[Path, Path]:
    """
    Ensure both .pth and .json for *region* are cached locally.
    Returns (pth_path, json_path).
    """
    model_info = ZENODO_MODELS.get(region)
    if model_info is None:
        logging.error(f"No Zenodo entry configured for region '{region}'.")
        sys.exit(1)

    record_id    = model_info["record_id"]
    pth_filename = model_info["pth_filename"]
    json_filename = model_info["json_filename"]

    if record_id == "XXXXXXX":
        logging.error(
            f"Zenodo record ID for '{region}' has not been configured yet. "
            + "Please update ZENODO_MODELS in mm_util.py after publishing."
        )
        sys.exit(1)

    def _use_cached_fallback(reason: str) -> tuple[Path, Path]:
        """Fall back to the highest locally cached version, or exit if none exists."""
        cached_dir = _latest_cached_version(region, pth_filename, json_filename)
        if cached_dir is not None:
            logging.info(f"{reason} Using cached '{region}' model ({cached_dir.name}).")
            return cached_dir / pth_filename, cached_dir / json_filename
        logging.error(
            f"{reason} No cached model found for '{region}'. "
            + "Please run MuscleMap with an internet connection at least once to download the model."
        )
        sys.exit(1)

    # Specific version requested: check cache first, then try Zenodo
    if version != "latest":
        cache_dir = _get_model_cache_dir(region, version)
        pth_path  = cache_dir / pth_filename
        json_path = cache_dir / json_filename
        if pth_path.exists() and json_path.exists():
            logging.info(f"Using cached '{region}' model v{version}.")
            return pth_path, json_path
        # Not cached — try to download from Zenodo if we have the record ID
        version_record_id = model_info.get("versions", {}).get(version)
        if version_record_id:
            logging.info(f"Downloading '{region}' model v{version} from Zenodo...")
            try:
                _, file_urls = _fetch_zenodo_version(version_record_id)
                for filename, dest in [(pth_filename, pth_path), (json_filename, json_path)]:
                    if filename not in file_urls:
                        logging.error(f"File '{filename}' not found in Zenodo record v{version}.")
                        sys.exit(1)
                    _download_file(file_urls[filename], dest)
                return pth_path, json_path
            except ConnectionError:
                return _use_cached_fallback(f"Zenodo unreachable. Could not download v{version}.")
        return _use_cached_fallback(
            f"Version '{version}' not found locally and no Zenodo record configured for it."
        )

    # Latest requested: warn on first use, then fetch from Zenodo
    if _latest_cached_version(region, pth_filename, json_filename) is None:
        logging.info(f"No local model found for '{region}'.")

    logging.info(f"Contacting Zenodo to resolve '{region}' model version...")
    try:
        resolved_version, file_urls = _fetch_zenodo_latest(record_id)
    except ConnectionError:
        return _use_cached_fallback("Zenodo unreachable.")

    cache_dir = _get_model_cache_dir(region, resolved_version)
    pth_path  = cache_dir / pth_filename
    json_path = cache_dir / json_filename

    if pth_path.exists() and json_path.exists():
        logging.info(f"Using cached '{region}' model v{resolved_version}.")
        return pth_path, json_path

    # A newer version is available — always download it and inform the user via logging.
    # To use a specific version, pass --model_version explicitly.
    cached_dir = _latest_cached_version(region, pth_filename, json_filename)
    if cached_dir is not None:
        cached_v = cached_dir.name[1:]  # strip leading 'v'
        if cached_v != resolved_version:
            logging.info(
                f"New '{region}' model version available: v{resolved_version} "
                + f"(current: v{cached_v}). Downloading automatically. "
                + f"To keep a specific version, use --model_version {cached_v}."
            )

    for filename, dest in [(pth_filename, pth_path), (json_filename, json_path)]:
        if filename not in file_urls:
            logging.error(
                f"File '{filename}' not found in Zenodo record v{resolved_version}. "
                + f"Available files: {list(file_urls.keys())}"
            )
            sys.exit(1)
        logging.info(f"Downloading '{filename}' (v{resolved_version}) from Zenodo...")
        _download_file(file_urls[filename], dest)

    logging.info(
        f"'{region}' model v{resolved_version} downloaded successfully. "
        + "To use a different version, use --model_version <version>."
    )
    return pth_path, json_path

# concept_id: the Zenodo "concept record ID" that always points to all versions.
# Fill these in after publishing each template set on Zenodo.
ZENODO_TEMPLATES: dict[str, dict[str, str]] = {
    "abdomen": {
        "record_id": "20043147",
    },
}


def ensure_template_downloaded(region: str) -> Path:
    """
    Ensure all template .nii.gz files for *region* are cached locally in
    _TEMPLATES_DIR/<region>/. Downloads from Zenodo on first use.
    Returns the template directory path.
    """
    template_info = ZENODO_TEMPLATES.get(region)
    if template_info is None:
        logging.error(f"No Zenodo template entry configured for region '{region}'.")
        sys.exit(1)

    record_id = template_info["record_id"]

    if record_id == "XXXXXXX":
        logging.error(
            f"Zenodo record ID for '{region}' templates has not been configured yet. "
            + "Please update ZENODO_TEMPLATES in mm_util.py after publishing."
        )
        sys.exit(1)

    template_dir = _TEMPLATES_DIR / region
    main_template = template_dir / f"{region}_template.nii.gz"
    main_dseg = template_dir / f"{region}_template_dseg.nii.gz"

    if main_template.exists() and main_dseg.exists():
        logging.info(f"Using cached '{region}' templates.")
        return template_dir

    logging.info(f"Contacting Zenodo to download '{region}' templates...")
    try:
        _, file_urls = _fetch_zenodo_latest(record_id)
    except ConnectionError as e:
        logging.error(
            f"Could not reach Zenodo to download '{region}' templates: {e}\n"
            + "Please run mm_register_to_template with an internet connection at least once."
        )
        sys.exit(1)

    nii_files = {k: v for k, v in file_urls.items() if k.endswith(".nii.gz")}
    if not nii_files:
        logging.error(f"No .nii.gz files found in Zenodo record for '{region}' templates.")
        sys.exit(1)

    template_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in nii_files.items():
        dest = template_dir / filename
        if not dest.exists():
            logging.info(f"Downloading template '{filename}' from Zenodo...")
            _download_file(url, dest)

    logging.info(f"'{region}' templates downloaded successfully.")
    return template_dir


AUTO_CHUNK_GPU_SAFETY_MARGIN = 0.70
AUTO_CHUNK_CPU_SAFETY_MARGIN = 0.35
AUTO_CHUNK_GPU_MIN_RESERVE_BYTES = 1.5 * 1024**3
AUTO_CHUNK_CPU_MIN_RESERVE_BYTES = 4 * 1024**3
AUTO_CHUNK_GPU_ESTIMATE_OVERHEAD = 2.50
AUTO_CHUNK_CPU_ESTIMATE_OVERHEAD = 2.00
AUTO_CHUNK_CPU_MAX_LOGIT_BYTES = 2 * 1024**3
AUTO_CHUNK_CPU_LOGIT_FRACTION = 0.25

#check_image_exists
def check_image_exists(image_path):
    if not os.path.isfile(image_path):
        logging.error(f"Image file '{image_path}' does not exist or is not a file.")
        sys.exit(1)
    if not os.access(image_path, os.R_OK):
        logging.error(f"Image file '{image_path}' is not readable.")
        sys.exit(1)

def get_config_path(region: str, version: str = "latest") -> Path:
    """Return the JSON config path for a region, downloading from Zenodo if needed."""
    _, json_path = ensure_model_downloaded(region, version)
    return json_path

def get_model_and_config_paths(region: str, specified_model: str, version: str = "latest") -> tuple[str, str]:
    if specified_model:
        model_path  = specified_model
        config_path = os.path.splitext(model_path)[0] + ".json"
        if not os.path.isfile(model_path):
            logging.error(f"Specified model '{specified_model}' does not exist.")
            sys.exit(1)
        if not os.path.isfile(config_path):
            logging.error(f"Config file for model '{specified_model}' does not exist.")
            sys.exit(1)
        return model_path, config_path

    pth_path, json_path = ensure_model_downloaded(region, version)
    return str(pth_path), str(json_path)

def get_template_paths(region, specified_template=None):
    template_dir = ensure_template_downloaded(region)

    if specified_template:
        template_path = str(template_dir / (specified_template + '.nii.gz'))
        template_segmentation_path = str(template_dir / (specified_template + '_dseg.nii.gz'))
    else:
        template_path = str(template_dir / f"{region}_template.nii.gz")
        template_segmentation_path = str(template_dir / f"{region}_template_dseg.nii.gz")

        if not os.path.isfile(template_path):
            logging.error(f"No template file found for region '{region}': {template_path}.")
            sys.exit(1)

        if not os.path.isfile(template_segmentation_path):
            logging.error(f"No template segmentation file found for region '{region}': {template_segmentation_path}.")
            sys.exit(1)

    return template_path, template_segmentation_path

def load_model_config(config_path: Path) -> ModelConfig:
    try:
        return ModelConfig.load_config(config_path)  # pyright: ignore[reportUnknownMemberType]
    except FileNotFoundError:
        logging.error(f"Error: The configuration file '{config_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logging.error(f"Error parsing the configuration file: {exc}")
        sys.exit(1)

def _format_bytes(num_bytes: int | float) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ("KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"

def _release_memory(device: torch.device | None = None):
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

def _is_oom_error(exc: Exception) -> bool:
    current = exc
    while current is not None:
        if isinstance(current, (MemoryError, torch.cuda.OutOfMemoryError)):
            return True
        msg = str(current).lower()
        if any(
            token in msg
            for token in (
                "out of memory",
                "cuda error: out of memory",
                "can't allocate memory",
                "cannot allocate memory",
                "std::bad_alloc",
                "bad alloc",
            )
        ):
            return True
        current = current.__cause__
    return False

def _resolve_target_spacing(header_zooms, target_pixdim):
    if not target_pixdim:
        return tuple(float(z) for z in header_zooms[:3])

    resolved: list[float] = []
    for axis, zoom in enumerate(header_zooms[:3]):
        try:
            target = float(target_pixdim[axis])
        except (IndexError, TypeError, ValueError):
            target = float(zoom)
        if target <= 0:
            target = float(zoom)
        resolved.append(target)
    return tuple(resolved)

def _read_int_file(path: Path, allow_zero=False):
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound or value >= 1 << 60:
        return None
    return value

def _get_cgroup_memory_budget():
    proc_cgroup = Path("/proc/self/cgroup")
    candidates: list[tuple[Path, Path]] = []

    if proc_cgroup.exists():
        try:
            lines = proc_cgroup.read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            _, controllers, relative_path = parts
            rel = relative_path.lstrip("/")

            if controllers == "":
                base = Path("/sys/fs/cgroup")
                scoped = base / rel if rel else base
                candidates.extend(
                    [
                        (scoped / "memory.max", scoped / "memory.current"),
                        (base / "memory.max", base / "memory.current"),
                    ]
                )
                continue

            if "memory" not in controllers.split(","):
                continue

            base = Path("/sys/fs/cgroup/memory")
            scoped = base / rel if rel else base
            candidates.extend(
                [
                    (scoped / "memory.limit_in_bytes", scoped / "memory.usage_in_bytes"),
                    (base / "memory.limit_in_bytes", base / "memory.usage_in_bytes"),
                ]
            )

    candidates.extend(
        [
            (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
            (
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ),
        ]
    )

    seen: set[tuple[str, str]] = set()
    for limit_path, current_path in candidates:
        key = (str(limit_path), str(current_path))
        if key in seen:
            continue
        seen.add(key)

        limit_bytes = _read_int_file(limit_path)
        current_bytes = _read_int_file(current_path, allow_zero=True)
        if limit_bytes is None or current_bytes is None:
            continue

        return {
            "limit_bytes": limit_bytes,
            "current_bytes": current_bytes,
            "available_bytes": max(limit_bytes - current_bytes, 0),
            "source": str(limit_path.parent),
        }

    return None

def _get_system_memory_budget():
    mem = psutil.virtual_memory()
    free_bytes = int(mem.available)
    total_bytes = int(mem.total)
    source = "system"

    cgroup_budget = _get_cgroup_memory_budget()
    if cgroup_budget is not None:
        free_bytes = min(free_bytes, int(cgroup_budget["available_bytes"]))
        total_bytes = min(total_bytes, int(cgroup_budget["limit_bytes"]))
        source = f"system+cgroup:{cgroup_budget['source']}"

    return free_bytes, total_bytes, source

def estimate_auto_chunk_size(image_path, device: torch.device | None, out_channels=None, target_pixdim=None):
    img_nii = load(image_path)
    header = cast(Nifti1Header, img_nii.header)
    dims = header.get_data_shape()[:3]
    zooms = header.get_zooms()[:3]
    del img_nii

    if len(dims) < 3 or dims[-1] < 1:
        return 1

    _release_memory(device)

    system_free_bytes, system_total_bytes, system_source = _get_system_memory_budget()
    system_reserve_bytes = max(AUTO_CHUNK_CPU_MIN_RESERVE_BYTES, int(system_total_bytes * 0.10))
    system_usable_bytes = int(max(system_free_bytes - system_reserve_bytes, 0) * AUTO_CHUNK_CPU_SAFETY_MARGIN)

    memory_source = system_source
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()  # pyright: ignore[reportUnnecessaryComparison]
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        reserve_bytes = max(AUTO_CHUNK_GPU_MIN_RESERVE_BYTES, int(total_bytes * 0.10))
        safety_margin = AUTO_CHUNK_GPU_SAFETY_MARGIN
        overhead_factor = AUTO_CHUNK_GPU_ESTIMATE_OVERHEAD
        memory_source = f"cuda:{device_index}"
    else:
        free_bytes = system_free_bytes
        total_bytes = system_total_bytes
        reserve_bytes = system_reserve_bytes
        safety_margin = AUTO_CHUNK_CPU_SAFETY_MARGIN
        overhead_factor = AUTO_CHUNK_CPU_ESTIMATE_OVERHEAD

    usable_bytes = int(max(free_bytes - reserve_bytes, 0) * safety_margin)
    if usable_bytes <= 0:
        logging.warning(
            "Auto chunk sizing found no free headroom after reserves; falling back to 1 slice "
            + f"(free={_format_bytes(free_bytes)} reserve={_format_bytes(reserve_bytes)})."
        )
        return 1

    resolved_spacing = _resolve_target_spacing(zooms, target_pixdim)
    resampled_dims = [
        max(1, int(math.ceil(float(dim) * float(zoom) / float(spacing))))
        for dim, zoom, spacing in zip(dims, zooms, resolved_spacing)
    ]

    out_channels = max(int(out_channels or 1), 1)
    f32 = np.dtype(np.float32).itemsize  # 4

    depth_scale = max(float(zooms[2]) / float(resolved_spacing[2]), 1e-6)

    # Memory at resampled resolution (inference): input tensor + logits
    resampled_bytes_per_slice = int(math.ceil(
        resampled_dims[0] * resampled_dims[1] * depth_scale
        * (f32 + out_channels * f32)  # input + logits
    ))

    # Memory at original resolution (post-processing inverse resample):
    # affine grid (3 floats) + resampled output (out_channels floats)
    original_bytes_per_slice = int(math.ceil(
        dims[0] * dims[1] * 1.0  # original voxels per input slice (no depth scaling)
        * (3 * f32 + out_channels * f32)  # grid + output
    ))

    bytes_per_input_slice = int(math.ceil(
        (resampled_bytes_per_slice + original_bytes_per_slice) * overhead_factor
    ))

    estimated_chunk = max(1, min(int(dims[2]), usable_bytes // max(bytes_per_input_slice, 1)))

    # On CPU, post-processing competes for the same system RAM — apply extra caps.
    # On GPU, post-processing stays on the device and the overhead factor already covers it.
    system_cap_chunk = None
    logit_cap_chunk = None
    if not (device is not None and device.type == "cuda" and torch.cuda.is_available()):
        cpu_bytes_per_input_slice = int(math.ceil(
            (resampled_bytes_per_slice + original_bytes_per_slice)
            * AUTO_CHUNK_CPU_ESTIMATE_OVERHEAD
        ))
        system_cap_chunk = max(1, system_usable_bytes // max(cpu_bytes_per_input_slice, 1))

        logit_bytes_per_input_slice = int(math.ceil(
            resampled_dims[0] * resampled_dims[1] * depth_scale
            * out_channels * f32
        ))
        max_logit_bytes = min(int(system_usable_bytes * AUTO_CHUNK_CPU_LOGIT_FRACTION), AUTO_CHUNK_CPU_MAX_LOGIT_BYTES)
        logit_cap_chunk = max(1, max_logit_bytes // max(logit_bytes_per_input_slice, 1))

        estimated_chunk = min(estimated_chunk, system_cap_chunk, logit_cap_chunk)

    logging.info(
        "Auto chunk sizing: free=%s usable=%s reserve=%s estimated=%s slice(s)(source=%s,overhead=%.2f%s).",
        _format_bytes(free_bytes),
        _format_bytes(usable_bytes),
        _format_bytes(reserve_bytes),
        estimated_chunk,
        memory_source,
        overhead_factor,
        (f", cpu_cap={system_cap_chunk}, logit_cap={logit_cap_chunk}") if system_cap_chunk is not None else "",
    )
    return estimated_chunk

def _run_inference_on_file(
    image_path: Path,
    pre_transforms,
    post_transforms,
    amp_context,
    device: torch.device,
    inferer,
    model: torch.nn.Module,
):
    data = None
    tensor = None
    pred = None
    single_pred = None
    post_in = None
    post_out = None
    seg_tensor = None
    try:
        data = {"image": image_path}
        data = pre_transforms(data)
        tensor = data["image"]
        if device.type == "cpu":
            tensor = tensor.float()
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device, non_blocking=device.type == "cuda")

        with amp_context, torch.inference_mode():
            pred = inferer(tensor, model)

        single_pred = pred.squeeze(0).squeeze(0)
        del pred
        pred = None
        post_in = {
            "pred": single_pred,
            "image": data["image"],
            "image_meta_dict": data["image_meta_dict"],
        }
        post_out = post_transforms(post_in)
        seg_tensor = post_out["pred"].detach().cpu().to(torch.int16)
        return seg_tensor.numpy().copy()
    finally:
        del data, tensor, pred, single_pred, post_in, post_out, seg_tensor
        _release_memory(device)

def _write_temp_chunk(image_proxy, affine, header, temp_dir: Path, start, end):
    vol_chunk = np.asarray(image_proxy.dataobj[..., start:end], dtype=np.float32)
    chunk_path = temp_dir.joinpath(f"chunk_{start}_{end}.nii")
    save(Nifti1Image(vol_chunk, affine, header.copy()), chunk_path)
    del vol_chunk
    return chunk_path

def save_nifti(data: np.ndarray, affine: npt.NDArray[np.float32], header: Nifti1Header, out_path):
    new_hdr = header.copy()
    img = Nifti1Image(data, affine, new_hdr)

    _, qcode = header.get_qform(coded=True)
    _, scode = header.get_sform(coded=True)
    img.set_qform(affine, int(qcode))
    img.set_sform(affine, int(scode))
    save(img, out_path)

def validate_extract_args(args):
    if args.method == 'dixon':
        if not args.fat_image or not args.water_image or not args.segmentation_image:
            print("For dixon method, you must provide -f (fat image), -w (water image), and -s (segmentation image).")
            exit(1)
    elif args.method in ['kmeans', 'gmm']:
        if not args.input_image or not args.components or not args.segmentation_image:
            print("For kmeans or gmm method, you must provide -i (input image), -c (number of components), and -s (segmentation image).")
            exit(1)
    elif args.method == 'average':
        if not args.input_image or not args.segmentation_image:
            print("For average, you must provide -i (input image) and -s (segmentation image).")
    string_args = ['fat_image', 'water_image', 'segmentation_image', 'input_image', 'region', 'output_dir']
    for arg_name in string_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} argument must be a string.")
            sys.exit(1)

def validate_register_to_template_args(args):
    if not args.input_image or not args.components or not args.segmentation_image:
        print("You must provide -i (input image), -s (segmentation image), and -r (region).")
        exit(1)
    string_args = ['input_image', 'segmentation_image', 'region', 'output_dir']
    for arg_name in string_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} argument must be a string.")
            sys.exit(1)

def extract_image_data(image_path: Path) -> tuple[Nifti1Image, npt.NDArray[np.float32], npt.NDArray[np.float32], Nifti1Header, tuple[int, int, int], tuple[int, int, int]]:
    img = load(image_path)
    img_array = img.get_fdata()

    header = cast(Nifti1Header, img.header)
    dim_x, dim_y, dim_z = header['dim'][1:4] #dim_z = number of axial slices
    pixdim_x, pixdim_y, pixdim_z = header['pixdim'][1:4] #voxel dimensions in mm
    affine = img.affine
    header = cast(Nifti1Header, img.header)
    return img, img_array, affine, header, (dim_x, dim_y, dim_z), (pixdim_x, pixdim_y, pixdim_z)

def add_slice_counts(
    results_entry: dict[int, dict[str, Any]],
    label_img:     np.ndarray,
    pix_dim:       tuple[float, float, float],
    col_name:      str = "Slices with segmentation",
) -> dict[int, dict[str, Any]]:

    if label_img.ndim != 3:
        raise ValueError("label_img must be 3-D")

    pix_dim = cast(tuple[float, float, float], tuple(float(p) for p in pix_dim))
    max_axis = int(np.argmax(pix_dim))
    if max(pix_dim) / min(pix_dim) < 1.01:
        max_axis = 2
    axes_to_reduce = tuple(ax for ax in range(3) if ax != max_axis)

    for lbl in np.unique(label_img):
        lbl = int(lbl)
        if lbl == 0:
            continue
        slice_present = np.any(label_img == lbl, axis=axes_to_reduce)
        slice_count   = int(slice_present.sum())

        entry = results_entry.get(lbl, {"Label": lbl, "Anatomy": ""})
        entry[col_name] = slice_count
        results_entry[lbl] = entry

    return results_entry

def apply_clustering(args, mask_img, components):
    if args.method == 'kmeans':
        clustering = KMeans(n_clusters = components, init = 'k-means++', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.labels_
    elif args.method == 'gmm':
        clustering = GaussianMixture(n_components = components, covariance_type = 'full', init_params = 'kmeans', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.predict(mask_img)
    else:
        raise ValueError("Either KMeans or GMM must be activated.")
    return labels, clustering

def create_image_array(img_array, mask_array, label, muscle_upper, fat_lower, components):
    if components not in (2, 3):
        raise ValueError("components must be 2 or 3")

    muscle_label = (mask_array == label)
    if components == 2:
        muscle_array    = muscle_label & (img_array <  muscle_upper)
        fat_array       = muscle_label & (img_array >= muscle_upper)
        undefined_array = np.zeros_like(img_array, dtype=bool)  # placeholder
    else:  # components == 3
        muscle_array    = muscle_label & (img_array <  muscle_upper)
        undefined_array = muscle_label & (img_array >= muscle_upper) & (img_array < fat_lower)
        fat_array       = muscle_label & (img_array >= fat_lower)
    return muscle_array, fat_array, undefined_array

def create_output_dir(output_dir: Path | None = None):
    if not output_dir:
        output_dir = Path.cwd()  # Use the current working directory if no output directory is provided
    else:
        # Construct the path to the output directory from the current working directory
        output_dir = output_dir.absolute()

    if not output_dir.exists():
        output_dir.mkdir()
        logging.info(f"Output directory {output_dir} created")
    return output_dir

def build_entry_dict_metrics(
    label_img: np.ndarray,
    model_config: ModelConfig,
    region: bool = False,
) -> dict[int, dict[str, Any]]:

    results_entry: dict[int, dict[str, Any]] = {}

    idx = {v: k.lower().replace(' ', '_') for k, v in model_config.dataset.labels.items()}

    unmatched_labels: list[int] = []
    for lbl in np.unique(label_img):
        lbl = int(lbl)
        if lbl <= 0:
            continue

        anatomy_text = idx.get(lbl, "") if (region and idx) else ""
        if region and idx and anatomy_text == "":
            unmatched_labels.append(lbl)

        results_entry[lbl] = {
            "Anatomy": anatomy_text,
            "Label":   lbl,
        }

    if region and idx and unmatched_labels:
        logging.warning(
            "No MuscleMap anatomy-side mapping was found for the following label IDs in "
            + "the current region configuration: %s. Only label numbers will be given",
            ", ".join(map(str, unmatched_labels))
        )

    return results_entry

def calculate_metrics_dixon(
    result_entry: dict[int, dict[str, Any]],
    label_img: np.ndarray,
    fat_array: np.ndarray,
    water_array: np.ndarray,
    pix_dim: tuple[float, float, float],
) -> dict[int, dict[str, Any]]:

    # raise value error when shapes do no match or when 4D is given as input
    if not (label_img.shape == water_array.shape == fat_array.shape):
        raise ValueError("Shape mismatch: segmentation image, water image, and fat image must have identical shapes.")
    if len(pix_dim) != 3:
        raise ValueError("pix_dim must be a 3-tuple (mm, mm, mm)")  # pyright: ignore[reportUnreachable]

    # fix voxel_vol_ml to calculate volume in ml
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # 1) Creating total fat fraction map for formula fat_signal/fat_signal + water signal.
    denom = fat_array + water_array
    ff_map = np.divide(
        fat_array, denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=(denom != 0)
    ).astype(np.float32)

    # 2) Flatten voor aggregration and set to int64 for efficiency
    flat_ff  = ff_map.ravel()
    flat_lbl = label_img.astype(np.int64).ravel()

    # 3) get max label from image. Labels in entry but not in image will be skipped.
    max_label = int(flat_lbl.max()) if flat_lbl.size else 0

    # 4) Sum and count per label
    sum_per_lbl   = np.bincount(flat_lbl, weights=flat_ff, minlength=max_label + 1)
    count_per_lbl = np.bincount(flat_lbl, minlength=max_label + 1)

    # 5) mean per label
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_lbl = np.divide(
            sum_per_lbl, count_per_lbl,
            out=np.zeros_like(sum_per_lbl, dtype=np.float32),
            where=(count_per_lbl != 0)
        )
    # 6) Update result_entry with fat percentages for each label
    for _k, entry in result_entry.items():
        lbl = int(entry.get("Label", _k))
        if lbl == 0:
            continue
        if lbl <= max_label and count_per_lbl[lbl] > 0:
            fat_pct = round(float(mean_per_lbl[lbl] * 100.0), 2)
            vol_ml  = round(float(count_per_lbl[lbl] * voxel_vol_ml), 2)
        else:
            fat_pct = np.nan
            vol_ml  = np.nan
        entry.update({
            "Fat (%)":     fat_pct,
            "Volume (ml)": vol_ml,
        })
    # 7) Return updated result_entry dictionary
    return result_entry

def calculate_metrics_average(
    result_entry: dict[int, dict[str, Any]],
    label_img: np.ndarray,
    img_array: np.ndarray,
    pix_dim: tuple[float, float, float],
) -> dict[int, dict[str, Any]]:

    #Raise ValueError when mismatch or image not in 3D
    if label_img.shape != img_array.shape:
        raise ValueError("Shape mismatch: Segmentation image and img_array must have the same shape")
    if len(pix_dim) != 3:
        raise ValueError("pix_dim must be a 3-tuple (mm, mm, mm)")  # pyright: ignore[reportUnreachable]

    # fix voxel_vol_ml to calculate volume in ml
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # Vectorized aggregations
    flat_lbl = label_img.astype(np.int64).ravel()
    flat_val = img_array.astype(np.float32).ravel()
    max_label = int(flat_lbl.max()) if flat_lbl.size else 0

    # Vectorized calculations for sum and count
    sum_per_lbl   = np.bincount(flat_lbl, weights=flat_val, minlength=max_label + 1)
    count_per_lbl = np.bincount(flat_lbl, minlength=max_label + 1)

    # ignore dividing by zero error and only divide where count_per_lbl is > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_lbl = np.divide(
            sum_per_lbl, count_per_lbl,
            out=np.zeros_like(sum_per_lbl, dtype=np.float32),
            where=(count_per_lbl != 0)
        )
    labels_present = np.flatnonzero(count_per_lbl)
    for lbl in labels_present:
        if lbl == 0:
            continue
        avg_int = round(float(mean_per_lbl[lbl]), 2)
        vol_ml  = round(float(count_per_lbl[lbl] * voxel_vol_ml), 2)

        entry = result_entry.get(int(lbl), {"Anatomy": "", "Label": int(lbl)})
        entry.update({
            "Average Intensity": avg_int,
            "Volume (ml)":       vol_ml,
        })
        result_entry[int(lbl)] = entry

    for _k, entry in list(result_entry.items()):
        lbl = int(entry.get("Label", _k))
        if lbl == 0:
            continue
        if lbl > max_label or count_per_lbl[lbl] == 0:
            entry.setdefault("Average Intensity", np.nan)
            entry.setdefault("Volume (ml)",       np.nan)

    return result_entry

def absolute_path(relative_path):
    base_path = os.path.dirname(__file__)  # Gets the directory where the script is located
    return os.path.join(base_path, relative_path)

class RemapLabels(MapTransform):
    def __init__(self, keys, id_map, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.id_map = id_map
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            out = lab.clone()
            for orig, tgt in self.id_map.items():
                out[lab == orig] = tgt
            d[key] = out
        return d

class SqueezeTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            out = lab.squeeze(0) if lab.dim() > 3 and lab.shape[0] == 1 else lab  # Remove channel dim if [1, H, W, D]
            d[key] = out
        return d

def connected_chunks(
    seg: np.ndarray,
    labels: np.ndarray | None = None,
    connectivity: int = 1,  # 3=26-connectivity
) -> np.ndarray:

    """
    Keeps only the largest connected component per label in a multi-label segmentation.
    - Supports both 3D (X,Y,Z) and 4D (1,X,Y,Z) arrays.
    - Incorporated to get optimize RAM memory management during inference for large images
    -
    """
    #Ensure that is nparray
    seg = np.asarray(seg)

    # Add channel dimension if necessary (to unify shape to 4D)
    remove_dim = False
    if seg.ndim == 3:
        seg_ch = seg[None, ...]
        remove_dim = True
    elif seg.ndim == 4 and seg.shape[0] == 1:
        seg_ch = seg
    else:
        raise ValueError(f"Expected (X,Y,Z) or (1,X,Y,Z), got {seg.shape}")

    # find labels excluding background (0)
    if labels is None:
        labels = np.unique(seg_ch)
    labels = labels[labels != 0]
    if labels.size == 0:
        result = seg_ch.astype(np.int16, copy=False)
        return result[0] if remove_dim else result

    # Extract the 3D volume from channel 0 for processing
    vol = seg_ch[0]

    # Connectivity structure for 3D, rank 3 = 26-connectivity to be not to conversative
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)

    # Buffers for 3D mask and labels
    mask3d = np.empty(vol.shape, dtype=bool)
    lab3d  = np.empty(vol.shape, dtype=np.int32)

    # Process each label independently on the 3D volume
    for lab_id in labels:
        np.equal(vol, lab_id, out=mask3d)
        if not mask3d.any():
            continue
        # Label connected components on mask3d
        ndi.label(mask3d, structure=structure, output=lab3d)
        max_lab = int(lab3d.max())
        if max_lab <= 1:
            continue

        # Compute sizes and pick the largest component
        counts = np.bincount(lab3d.ravel())
        keep = counts[1:].argmax() + 1
        del counts

        # Zero out everything except the largest component for this label
        np.logical_and(mask3d, lab3d != keep, out=mask3d)
        vol[mask3d] = 0

    # Write back the processed 3D volume into output array
    seg_ch[0] = vol
    # Cleanup
    del mask3d, lab3d

    # Convert to int16 and drop channel dim if needed
    result = seg_ch.astype(np.int16, copy=False)
    if remove_dim:
        result = result[0]
    return result

def is_nifti(path: str) -> bool:
    p = path.lower()
    return p.endswith(".nii.gz") or p.endswith(".nii")

def _make_out_path(image_path: Path, output_dir: Path, tag="_dseg") -> Path:
    fname = image_path.name
    if fname.endswith(".nii.gz"):
        base = fname[:-7]
    elif fname.endswith(".nii"):
        base = fname[:-4]
    else:
        raise ValueError(f'image_path must end either in ".nii.gz" or ".nii", instead we got "{image_path}"')
    return output_dir.joinpath(f"{base}{tag}.nii.gz")

CUDA_DEVICE = torch.device('cuda')

def run_inference(
    image_path,
    output_dir: Path,
    pre_transforms,
    post_transforms,
    model: torch.nn.Module,
    amp_context=None,
    chunk_size=25,
    device: torch.device = CUDA_DEVICE,
    inferer=None,
    out_channels=None,
    target_pixdim=None,
):
    out_path = _make_out_path(image_path, output_dir, "_dseg")
    img_nii: Nifti1Image = load(image_path)
    affine = img_nii.affine.copy()
    header = cast(Nifti1Header, img_nii.header).copy()
    dims = header.get_data_shape()
    D = dims[-1]
    auto_chunking = isinstance(chunk_size, str) and chunk_size.lower() == "auto"

    if auto_chunking:
        chunk_size = estimate_auto_chunk_size(
            image_path,
            device,
            out_channels=out_channels,
            target_pixdim=target_pixdim,
        )
    else:
        chunk_size = int(chunk_size)

    chunk_size = max(1, min(chunk_size, D))
    logging.info("Using chunk size: %s%s", chunk_size, " (auto)" if auto_chunking else "")

    temp_dir = output_dir.joinpath("temp_chunks")
    full_seg = None
    try:
        if chunk_size >= D:
            try:
                seg_np = _run_inference_on_file(
                    image_path,
                    pre_transforms,
                    post_transforms,
                    amp_context,
                    device,
                    inferer,
                    model,
                )
            except Exception as exc:
                if not (auto_chunking and _is_oom_error(exc) and D > 1):
                    raise
                chunk_size = max(1, D // 2)
                logging.warning(
                    "Auto chunking hit OOM on the full volume; retrying with chunk size %s.",
                    chunk_size,
                )
            else:
                full_seg = connected_chunks(seg_np)
                save(Nifti1Image(full_seg, affine, header), out_path)
                del seg_np
                return out_path

        os.makedirs(temp_dir, exist_ok=True)
        full_seg = np.zeros(dims, dtype=np.int16)
        start = 0
        while start < D:
            end = min(start + chunk_size, D)
            chunk_path = None
            try:
                chunk_path = _write_temp_chunk(img_nii, affine, header, temp_dir, start, end)
                seg_np = _run_inference_on_file(
                    chunk_path,
                    pre_transforms,
                    post_transforms,
                    amp_context,
                    device,
                    inferer,
                    model,
                )
                full_seg[..., start:end] = seg_np
                del seg_np
                start = end
            except Exception as exc:
                if not (auto_chunking and _is_oom_error(exc)):
                    raise
                if chunk_size == 1:
                    raise RuntimeError(
                        "Auto chunking could not find a safe chunk size. Inference still OOMs at 1 slice."
                    ) from exc
                new_chunk_size = max(1, chunk_size // 2)
                logging.warning(
                    "OOM while processing slices %s:%s with chunk size %s; retrying with %s.",
                    start,
                    end,
                    chunk_size,
                    new_chunk_size,
                )
                chunk_size = new_chunk_size
            finally:
                if chunk_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)

        full_seg = connected_chunks(full_seg)
        save(Nifti1Image(full_seg, affine, header), out_path)
        return out_path
    finally:
        del img_nii
        if full_seg is not None:
            del full_seg
        _release_memory(device)
        shutil.rmtree(temp_dir, ignore_errors=True)
