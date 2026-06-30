#!/usr/bin/env python
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset, list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers.factories import Norm
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

from muscle_map.mm_util import load_model_config


class RemapLabelValuesd(MapTransform):
    """Map original segmentation label values to compact training class IDs."""

    def __init__(self, keys, id_map, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.id_map = {int(k): int(v) for k, v in id_map.items()}

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

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            value = d[key]
            if value.ndim >= 4 and value.shape[-1] == 1:
                d[key] = value.squeeze(-1)
        return d


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset stats and model training."""
    parser = argparse.ArgumentParser(
        description="Train a MuscleMap-compatible MONAI UNet checkpoint from NIfTI image/label pairs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser(
        "stats",
        help="Compute dataset median spacing and image size from imagesTr.",
    )
    stats_parser.add_argument(
        "-d",
        "--dataset_dir",
        required=True,
        type=str,
        help="Dataset root containing imagesTr and labelsTr.",
    )
    stats_parser.add_argument(
        "-o",
        "--output",
        default="dataset_stats.json",
        type=str,
        help="Output JSON file for dataset statistics.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a checkpoint.",
    )
    train_parser.add_argument(
        "-d",
        "--dataset_dir",
        required=True,
        type=str,
        help="Dataset root containing imagesTr and labelsTr.",
    )
    train_parser.add_argument(
        "--val_dataset_dir",
        default=None,
        type=str,
        help="Optional held-out validation dataset root with imagesTr and labelsTr.",
    )
    train_parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        type=str,
        help="Training/model metadata JSON. See examples/train_metadata_mock.json.",
    )
    train_parser.add_argument(
        "-o",
        "--output_dir",
        default="training_output",
        type=str,
        help="Directory for checkpoints and copied configs.",
    )
    train_parser.add_argument("--epochs", default=1000, type=int, help="Training epochs. Default: 1000.")
    train_parser.add_argument(
        "--save_every",
        default=10,
        type=int,
        help="Save a MuscleMap-compatible .pth checkpoint every N epochs. Default: 10.",
    )
    train_parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override target spacing in mm, e.g. --spacing 1.0 1.0 3.0.",
    )
    train_parser.add_argument(
        "--stats",
        default=None,
        type=str,
        help="Dataset stats JSON from 'mm_train stats'. Used when metadata parameters.pix_dim is 'median'.",
    )
    train_parser.add_argument("--batch_size", default=None, type=int, help="Override metadata training.batch_size.")
    train_parser.add_argument("--learning_rate", default=None, type=float, help="Override metadata training.learning_rate.")
    train_parser.add_argument("--num_workers", default=None, type=int, help="Override metadata training.num_workers.")
    train_parser.add_argument("--seed", default=None, type=int, help="Override metadata training.seed.")
    train_parser.add_argument(
        "-g",
        "--use_GPU",
        default="Y",
        choices=["Y", "N"],
        help="Use GPU when available. Default: Y.",
    )
    return parser


def _metadata_value(metadata, section, key, default):
    return metadata.get(section, {}).get(key, default)


def _load_metadata(path):
    metadata = load_model_config(path)
    if "labels" not in metadata or not metadata["labels"]:
        raise ValueError("Metadata must contain a non-empty 'labels' list.")
    return metadata


def _discover_cases(dataset_dir):
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labelsTr directory: {labels_dir}")

    cases = []
    for image_path in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = image_path.name[:-12]
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for case '{case_id}': {label_path}")
        cases.append({"image": str(image_path), "label": str(label_path), "case_id": case_id})

    if not cases:
        raise ValueError(f"No cases found in {images_dir}; expected files ending in _0000.nii.gz.")
    return cases


def _compute_dataset_stats(dataset_dir):
    import nibabel as nib

    cases = _discover_cases(dataset_dir)
    spacings = []
    sizes = []
    for case in cases:
        img = nib.load(case["image"])
        sizes.append([int(v) for v in img.header.get_data_shape()[:3]])
        spacings.append([float(v) for v in img.header.get_zooms()[:3]])
        del img

    spacing_median = np.median(np.asarray(spacings, dtype=np.float64), axis=0)
    size_median = np.median(np.asarray(sizes, dtype=np.float64), axis=0)
    return {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "num_cases": len(cases),
        "median_spacing": [round(float(v), 1) for v in spacing_median],
        "median_size": [int(round(float(v))) for v in size_median],
        "spacings": spacings,
        "sizes": sizes,
    }


def _build_config(metadata):
    labels = sorted({int(entry["value"]) for entry in metadata["labels"]})
    out_channels = len(labels) + 1

    config = dict(metadata)
    config.setdefault("model", {})
    config.setdefault("parameters", {})

    config["model"].setdefault("version", "de-novo")
    config["model"].setdefault("spatial_dims", 3)
    config["model"].setdefault("in_channels", 1)
    config["model"]["out_channels"] = out_channels
    config["model"].setdefault("channels", [16, 32, 64, 128, 256])
    config["model"].setdefault("act", "PRELU")
    config["model"].setdefault("strides", [2, 2, 2, 2])
    config["model"].setdefault("num_res_units", 2)
    config["model"].setdefault("norm", "instance")

    config["parameters"].setdefault("roi_size", [96, 96, 96])
    config["parameters"].setdefault("spatial_window_batch_size", 1)
    config["parameters"].setdefault("pix_dim", "median")
    config["parameters"].setdefault("train_patch_size", list(config["parameters"]["roi_size"]))
    return config


def _resolve_pix_dim(config, args):
    if args.spacing is not None:
        pix_dim = [round(float(v), 1) for v in args.spacing]
    else:
        configured = config["parameters"].get("pix_dim", "median")
        if isinstance(configured, str) and configured.lower() == "median":
            if not args.stats:
                raise ValueError(
                    "metadata parameters.pix_dim is 'median'; run 'mm_train stats' and pass --stats, "
                    "or override with --spacing X Y Z."
                )
            stats = load_model_config(args.stats)
            pix_dim = stats.get("median_spacing")
            if not pix_dim or len(pix_dim) != 3:
                raise ValueError(f"Stats file '{args.stats}' does not contain a valid median_spacing.")
        else:
            pix_dim = configured

    if len(pix_dim) != 3:
        raise ValueError("Target spacing must contain exactly three values.")
    config["parameters"]["pix_dim"] = [round(float(v), 1) for v in pix_dim]


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _make_transforms(config, orig_to_compact, training=True):
    pix_dim = tuple(config["parameters"]["pix_dim"])
    patch_size = tuple(config["parameters"].get("train_patch_size", [256, 256, 1]))
    rotation_degrees = float(config.get("training", {}).get("rotation_degrees", 15.0))
    rotation_radians = np.deg2rad(rotation_degrees)
    samples_per_volume = int(config.get("training", {}).get("samples_per_volume", 4))

    transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pix_dim, mode=("bilinear", "nearest")),
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
    if int(config["model"]["spatial_dims"]) == 2:
        transforms.append(SqueezeLastSpatialDimd(keys=["image", "label"]))
    return Compose(transforms)


def _make_model(config):
    norm_map = {"instance": Norm.INSTANCE}
    norm_name = config["model"]["norm"]
    if norm_name not in norm_map:
        raise ValueError(f"Unsupported norm '{norm_name}'. Supported values: {sorted(norm_map)}")
    return UNet(
        spatial_dims=int(config["model"]["spatial_dims"]),
        in_channels=int(config["model"]["in_channels"]),
        out_channels=int(config["model"]["out_channels"]),
        channels=tuple(config["model"]["channels"]),
        act=config["model"]["act"],
        strides=tuple(config["model"]["strides"]),
        num_res_units=int(config["model"]["num_res_units"]),
        norm=norm_map[norm_name],
    )


def _save_checkpoint(model, config, output_dir, stem):
    pth_path = output_dir / f"{stem}.pth"
    json_path = output_dir / f"{stem}.json"
    torch.save(model.state_dict(), pth_path)
    _write_json(json_path, config)
    return pth_path


def _run_validation(model, loader, device, loss_fn, dice_metric):
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = get_parser().parse_args()

    if args.command == "stats":
        stats = _compute_dataset_stats(args.dataset_dir)
        _write_json(args.output, stats)
        logging.info("Wrote dataset statistics to %s.", args.output)
        return

    metadata = _load_metadata(args.metadata)
    config = _build_config(metadata)
    _resolve_pix_dim(config, args)

    seed = args.seed if args.seed is not None else int(_metadata_value(config, "training", "seed", 2026))
    set_determinism(seed=seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "training_config.json", config)

    labels = sorted({int(entry["value"]) for entry in config["labels"]})
    orig_to_compact = {0: 0}
    for compact_id, original_id in enumerate(labels, start=1):
        orig_to_compact[original_id] = compact_id

    train_cases = _discover_cases(args.dataset_dir)
    val_cases = _discover_cases(args.val_dataset_dir) if args.val_dataset_dir else []
    logging.info("Discovered %s training cases and %s validation cases.", len(train_cases), len(val_cases))

    train_ds = Dataset(data=train_cases, transform=_make_transforms(config, orig_to_compact, training=True))
    val_ds = Dataset(data=val_cases, transform=_make_transforms(config, orig_to_compact, training=False)) if val_cases else None

    batch_size = args.batch_size or int(_metadata_value(config, "training", "batch_size", 2))
    num_workers = args.num_workers if args.num_workers is not None else int(_metadata_value(config, "training", "num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available() and args.use_GPU == "Y",
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available() and args.use_GPU == "Y",
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_GPU == "Y" else "cpu")
    logging.info("Training on %s.", device)

    model = _make_model(config).to(device)
    learning_rate = args.learning_rate or float(_metadata_value(config, "training", "learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(_metadata_value(config, "training", "weight_decay", 1e-5)))
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels_t = batch["label"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(len(train_loader), 1)
        if val_loader is not None:
            val_loss, mean_dice = _run_validation(model, val_loader, device, loss_fn, dice_metric)
            logging.info(
                "Epoch %s/%s - train_loss=%.5f val_loss=%.5f val_dice=%.5f",
                epoch,
                args.epochs,
                epoch_loss,
                val_loss,
                mean_dice,
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_path = _save_checkpoint(model, config, output_dir, "best_model")
                logging.info("Saved new best checkpoint: %s", best_path)
        else:
            logging.info("Epoch %s/%s - train_loss=%.5f", epoch, args.epochs, epoch_loss)

        if epoch % args.save_every == 0:
            ckpt_path = _save_checkpoint(model, config, output_dir, f"checkpoint_epoch_{epoch:04d}")
            logging.info("Saved checkpoint: %s", ckpt_path)

    final_path = _save_checkpoint(model, config, output_dir, "final_model")
    shutil.copy2(output_dir / "final_model.json", output_dir / "model_config.json")
    logging.info("Training complete. Final checkpoint: %s", final_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        sys.exit(1)
