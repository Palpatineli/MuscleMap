#!/usr/bin/env python
"""Convert legacy MuscleMap model metadata to the current configuration schema."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from monai.networks.nets.unet import UNet
import torch

def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Legacy metadata field '{key}' must be an object.")
    return value


def _required_list(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Legacy metadata field '{key}' must be an array.")
    return value


def _required_value(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Legacy metadata is missing required field '{key}'.")
    return data[key]


def _label_name(label: Mapping[str, Any], value: int, used_names: set[str]) -> str:
    parts = [str(label.get(key, "")).strip() for key in ("region", "anatomy", "side")]
    name = re.sub(r"[^a-z0-9]+", "_", "_".join(part for part in parts if part).lower()).strip("_")
    name = name or f"label_{value}"
    if name in used_names:
        name = f"{name}_{value}"
    used_names.add(name)
    return name


def convert_legacy_config(legacy: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a legacy ``parameters/model/labels`` document to ``ModelConfig`` JSON."""
    parameters = _required_mapping(legacy, "parameters")
    model = _required_mapping(legacy, "model")
    legacy_labels = _required_list(legacy, "labels")

    labels = {"background": 0}
    used_names = set(labels)
    for entry in legacy_labels:
        if not isinstance(entry, Mapping):
            raise ValueError("Every legacy label entry must be an object.")
        value = int(_required_value(entry, "value"))
        if value <= 0 or value in labels.values():
            raise ValueError(f"Legacy label value {value} is invalid or duplicated.")
        labels[_label_name(entry, value, used_names)] = value

    out_channels = int(_required_value(model, "out_channels"))
    if out_channels != len(labels):
        raise ValueError(
            f"Legacy model declares {out_channels} outputs, but its labels define {len(labels)} classes."
        )

    converted = {
        "architecture": {
            key: _required_value(model, key)
            for key in ("spatial_dims", "in_channels", "channels", "act", "strides", "num_res_units", "norm")
        },
        "image": {
            "roi_size": _required_value(parameters, "roi_size"),
            "spatial_window_batch_size": _required_value(parameters, "spatial_window_batch_size"),
            "spacing": _required_value(parameters, "pix_dim"),
            "train_patch_size": None,
        },
        "training": {
            "batch_size": None,
            "samples_per_volume": None,
            "num_workers": parameters.get("num_workers"),
            "learning_rate": None,
            "weight_decay": None,
            "seed": None,
            "rotation_degrees": None,
            "verify_nifti_files": None,
        },
        "dataset": {
            "description": None,
            "labels": labels,
            "name": None,
            "numTraining": None,
            "reference": None,
            "release": None,
            "channel_names": None,
            "file_ending": None,
        },
        "training_compatible": False,
        "missing_training_fields": [
            "image.train_patch_size",
            "training.batch_size",
            "training.samples_per_volume",
            "training.learning_rate",
            "training.weight_decay",
            "training.seed",
            "training.rotation_degrees",
            "training.verify_nifti_files",
            "dataset.channel_names",
            "dataset.file_ending",
        ],
    }
    return converted


def _validate_checkpoint(path: Path, metadata: Mapping[str, Any]) -> None:
    """Verify that converted metadata exactly describes a checkpoint's tensors."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint '{path}' is not a state-dict mapping.")
    architecture = _required_mapping(metadata, "architecture")
    dataset = _required_mapping(metadata, "dataset")
    labels = _required_mapping(dataset, "labels")
    model = UNet(
        out_channels=len(set(labels.values())),
        **architecture,
    )
    model.load_state_dict(state, strict=True)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def get_parser() -> argparse.ArgumentParser:
    """Build the legacy-model metadata conversion CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_json", type=Path, help="Legacy metadata JSON using parameters/model/labels.")
    parser.add_argument("output_json", type=Path, help="Path for the converted ModelConfig JSON.")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to validate against the converted metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output JSON.")
    return parser


def main() -> None:
    """Convert one legacy metadata document."""
    parser = get_parser()
    args = parser.parse_args()
    if not args.legacy_json.is_file():
        parser.error(f"Legacy metadata file does not exist: {args.legacy_json}")
    if args.output_json.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output_json}. Use --overwrite to replace it.")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    try:
        legacy = json.loads(args.legacy_json.read_text(encoding="utf-8"))
        if not isinstance(legacy, Mapping):
            raise ValueError("Legacy metadata must be a JSON object.")
        converted = convert_legacy_config(legacy)
        if args.checkpoint is not None:
            _validate_checkpoint(args.checkpoint, converted)
        _write_json(args.output_json, converted)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Wrote inference-only metadata to {args.output_json}")


if __name__ == "__main__":
    main()
