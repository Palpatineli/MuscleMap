#!/usr/bin/env python
import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files as import_files
import logging
import sys
import json
from tqdm import tqdm
from pathlib import Path
from typing import Hashable, cast, override
from monai.metrics.metric import CumulativeIterationMetric
from nibabel import load, Nifti1Header
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset, list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet
from monai.transforms import MapTransform
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    CropForegroundd,
    RandCropByPosNegLabeld,
    SpatialPadd,
)
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, RandRotated, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.utils import set_determinism

from muscle_map.mm_util import DatasetParameter, DatasetStats, ModelConfig

DATA_STATS_FILE = "data_stats.json"


class RemapLabelValuesd(MapTransform):
    """Map original segmentation label values to compact training class IDs."""

    def __init__(self, keys, id_map: Mapping[int, int | str], allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.id_map: dict[int, int] = {int(k): int(v) for k, v in id_map.items()}

    @override
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            out = torch.zeros_like(label, dtype=torch.long)
            for orig, compact in self.id_map.items():
                out[label == orig] = compact
            d[key] = out
        return d


class SqueezeLastSpatialDimd(MapTransform):
    """Convert single-slice 3D patches into 2D patches for 2D models."""

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    @override
    def __call__(self, data):
        d: dict[Hashable, np.ndarray] = dict(data)
        for key in self.keys:
            value = d[key]
            if value.ndim >= 4 and value.shape[-1] == 1:
                d[key] = value.squeeze(-1)
        return d


@dataclass
class ArgStats:
    dataset_dir: str
    output: str

@dataclass
class ArgTrain:
    dataset_dir: str
    result_dir: str
    epochs: int = 1000
    save_every: int = 10
    spacing: tuple[float, float, float] | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    num_workers: int | None = None
    seed: int | None = None

    def __post_init__(self):
        self.spacing = cast(tuple[float, float, float], tuple(self.spacing)) if self.spacing is not None else None

def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset stats and model training."""
    parser = argparse.ArgumentParser(
        description="Train a MuscleMap-compatible MONAI UNet checkpoint from NIfTI image/label pairs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stats_parser = subparsers.add_parser("stats", help="Collect dataset image spacing and size.")
    stats_parser.add_argument("-d", "--dataset_dir", required=True, type=str,
        help="Dataset root containing imagesTr and labelsTr.")
    stats_parser.add_argument("-o", "--output", default="data_stats.json", type=str,
        help="Output JSON file for dataset statistics.")

    train_parser = subparsers.add_parser("train", help="Train a checkpoint.")
    train_parser.add_argument("-d", "--dataset_dir", required=True, type=str,
        help="Dataset root containing imagesTr and labelsTr.")
    train_parser.add_argument("-o", "--result_dir", required=True, type=str,
        help="Directory for dataset stats, configs, and checkpoints.")
    train_parser.add_argument("--epochs", default=1000, type=int, help="Training epochs. Default: 1000.")
    train_parser.add_argument("--save_every", default=10, type=int,
        help="Save a MuscleMap-compatible .pth checkpoint every N epochs. Default: 10.")
    train_parser.add_argument("--spacing", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
        help="Override target spacing in mm, e.g. --spacing 1.0 1.0 3.0.")
    train_parser.add_argument("--batch_size", default=None, type=int, help="Override metadata training.batch_size.")
    train_parser.add_argument("-l", "--learning_rate", default=None, type=float, help="Override metadata training.learning_rate.")
    train_parser.add_argument("--num_workers", default=None, type=int, help="Override metadata training.num_workers.")
    train_parser.add_argument("--seed", default=None, type=int, help="Override metadata training.seed.")
    return parser

def parse_args() -> ArgStats | ArgTrain:
    args = get_parser().parse_args()
    if args.command == "stats":
        OutClass = ArgStats
    elif args.command == "train":
        OutClass = ArgTrain
    else:
        raise NotImplementedError(f"unknown subcommand {args.command}")
    args_dict = vars(args)
    args_dict.pop('command')
    return OutClass(**args_dict)

@dataclass
class Sample:
    image: Path
    label: Path
    case_id: str

def _discover_cases(dataset_dir: Path) -> list[Sample]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir() or not images_dir.exists():
        raise FileNotFoundError(f"Missing imagesTr directory: {images_dir}")
    if not labels_dir.is_dir() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing labelsTr directory: {labels_dir}")

    cases: list[Sample] = []
    for image_path in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = image_path.name[:-12]
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for case '{case_id}': {label_path}")
        cases.append(Sample(image_path, label_path, case_id))

    if not cases:
        raise ValueError(f"No cases found in {images_dir}; expected files ending in _0000.nii.gz.")
    return cases

def _collect_dataset_stats(dataset_dir: Path) -> DatasetStats:
    cases = _discover_cases(dataset_dir)
    spacings: list[list[float]] = []
    sizes: list[list[int]] = []
    for case in tqdm(cases):
        img = load(case.image)
        sizes.append([int(v) for v in cast(Nifti1Header, img.header).get_data_shape()[:3]])
        spacings.append([float(v) for v in cast(Nifti1Header, img.header).get_zooms()[:3]])
        del img
    return DatasetStats(dataset_dir=Path(dataset_dir).resolve(), num_cases=len(cases), spacings=spacings, sizes=sizes)

def _build_config(data_param: DatasetParameter, datastat: DatasetStats) -> ModelConfig:
    default_config_file = import_files("muscle_map") / "model_config_default.json"
    config = ModelConfig.load_config(default_config_file, dataset=data_param.to_dict())  # pyright: ignore[reportUnknownMemberType]
    config.architecture.in_channels = len(data_param.channel_names)
    config.image.spacing = tuple(np.median(np.array(datastat.spacings), axis=0))
    return config

def _make_transforms(config: ModelConfig, orig_to_compact: dict[int, int], training: bool = True) -> Compose:
    spacing = config.image.spacing
    patch_size = config.image.train_patch_size
    rotation_degrees = config.training.rotation_degrees
    rotation_radians = np.deg2rad(rotation_degrees)
    samples_per_volume = config.training.samples_per_volume

    transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=20),
    ]

    if training and rotation_degrees > 0:
        transforms.append(
            RandRotated(
                keys=["image", "label"],
                range_x=rotation_radians,
                range_y=rotation_radians,
                range_z=rotation_radians,
                prob=1.0,
                mode=("bilinear", "nearest"),
                padding_mode="border",
            )
        )

    transforms.extend(
        [
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
                method="end",
                mode="constant",
            ),
            EnsureTyped(keys=["image", "label"]),
            RemapLabelValuesd(keys=["label"], id_map=orig_to_compact),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=samples_per_volume if training else 1,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    if config.architecture.spatial_dims == 2:
        transforms.append(SqueezeLastSpatialDimd(keys=["image", "label"]))
    return Compose(transforms)

def _save_checkpoint(model: torch.nn.Module, config: ModelConfig, output_path: Path):
    pth_path = output_path.with_suffix(".pth")
    json_path = output_path.with_suffix(".json")
    torch.save(model.state_dict(), pth_path)
    json.dump(config.to_dict(), json_path.open('w', encoding='utf-8'), indent=4)
    return pth_path


def _run_validation(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                    loss_fn: torch.nn.modules.loss._Loss,  # pyright: ignore[reportPrivateUsage]
                    dice_metric: CumulativeIterationMetric) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            logits = model(images)
            val_loss += loss_fn(logits, labels).item()
            preds = torch.argmax(logits, dim=1, keepdim=True)
            out_channels = logits.shape[1]
            preds_onehot = F.one_hot(preds.squeeze(1), num_classes=out_channels)
            labels_onehot = F.one_hot(labels.squeeze(1), num_classes=out_channels)
            channel_dim = preds_onehot.ndim - 1
            preds_onehot = preds_onehot.movedim(channel_dim, 1).float()
            labels_onehot = labels_onehot.movedim(channel_dim, 1).float()
            dice_metric(y_pred=preds_onehot, y=labels_onehot)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return val_loss / max(len(loader), 1), mean_dice


def main():
    args = parse_args()

    if isinstance(args, ArgStats):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        stats = _collect_dataset_stats(Path(args.dataset_dir))
        if not (output := Path(args.output)).exists() or output.is_dir():
            output.mkdir(exist_ok=True, parents=True)
            output /= DATA_STATS_FILE
        else:
            output.parent.mkdir(exist_ok=True)
        json.dump(stats.to_dict(), output.open('w', encoding='utf-8'), indent=4)
        logging.info(f"Wrote dataset statistics to {args.output}.")
    else:
        result_dir = Path(args.result_dir)
        logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s - %(levelname)s] %(message)s",
                            handlers=[logging.FileHandler(result_dir / "training.log", mode='a'),
                                      logging.StreamHandler(sys.stdout)])
        stats = DatasetStats.load_config(result_dir / DATA_STATS_FILE)  # pyright: ignore[reportUnknownMemberType]
        image_parameter = DatasetParameter.load_config(result_dir / "dataset_parameter.json")  # pyright: ignore[reportUnknownMemberType]
        config = _build_config(image_parameter, stats)
        json.dump(config.to_dict(), result_dir.joinpath("training_config.json").open('w', encoding='utf-8'), indent=4)
        seed = args.seed or config.training.seed
        set_determinism(seed=seed)

        labels = sorted({x for x in config.dataset.labels.values() if x > 0})
        orig_to_compact = {0: 0}
        for compact_id, original_id in enumerate(labels, start=1):
            orig_to_compact[original_id] = compact_id

        train_cases = _discover_cases(Path(args.dataset_dir))
        logging.info(f"Discovered {len(train_cases)} training cases.", )

        train_ds = Dataset(data=train_cases, transform=_make_transforms(config, orig_to_compact, training=True))

        batch_size = args.batch_size or config.training.batch_size
        num_workers = args.num_workers if args.num_workers is not None else config.training.num_workers
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on {device}")

        model = UNet(**config.architecture.to_dict()).to(device)
        learning_rate = args.learning_rate or float(config.training.learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=float(config.training.weight_decay))
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader):
                images = batch["image"].to(device)
                labels_t = batch["label"].to(device).long()
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, labels_t)
                loss.backward()
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                epoch_loss += loss.item()

            length = max(len(train_loader), 1)
            epoch_loss /= length
            logging.info(f"Epoch {epoch}/{args.epochs} - train_loss={epoch_loss:.5f}")

            if epoch % args.save_every == 0:
                ckpt_path = _save_checkpoint(model, config, result_dir / f"checkpoint_epoch_{epoch:04d}")
                logging.info("Saved checkpoint: %s", ckpt_path)

        final_path = _save_checkpoint(model, config, result_dir / "final_model")
        logging.info(f"Training complete. Final checkpoint: {final_path}")

        # pseudo dice
        dice_fn = DiceMetric(include_background=False, reduction="mean")
        mean_ce, mean_dice = _run_validation(model, train_loader, device, loss_fn, dice_fn)
        logging.info(f"Training loss: {mean_ce}. Pseudo dice: {mean_dice}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        sys.exit(1)
