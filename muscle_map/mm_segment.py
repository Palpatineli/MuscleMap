#!/usr/bin/env python
from pathlib import Path
import warnings
import argparse
from dataclasses import dataclass
import logging
import gc
from contextlib import nullcontext
import re
import sys
from typing import cast
from monai.inferers.inferer import SliceInferer, SlidingWindowInferer
from monai.networks.nets.unet import UNet
from monai.transforms.post.dictionary import AsDiscreted, Invertd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import EnsureTyped, EnsureChannelFirstd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.croppad.dictionary import CropForegroundd, SpatialPadd
from monai.networks.layers.factories import Norm
from time import perf_counter
import torch

from muscle_map.mm_util import (
    ModelConfig,
    RemapLabels,
    SqueezeTransform,
    get_model_and_config_paths,
    is_nifti,
    prediction_path,
    run_inference,
)

warnings.filterwarnings("ignore")
print("Command line arguments received:", sys.argv)

_CHANNEL_SUFFIX = re.compile(r"^(?P<case>.+)_(?P<channel>\d{4})$")


@dataclass(frozen=True)
class InferenceCase:
    """One logical inference case, represented by one or more image channels."""

    image_paths: tuple[Path, ...]
    output_path: Path

def chunk_size_arg(value: str) -> int | str:
    """Parse a positive chunk size or the special value 'auto'."""
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("chunk_size must be a positive integer or 'auto'.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("chunk_size must be at least 1.")
    return parsed


def _nifti_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    raise ValueError(f"Input '{path}' is not a NIfTI image.")


def _expand_input_paths(input_spec: str) -> list[Path]:
    """Expand file and directory CLI arguments into individual NIfTI paths."""
    image_paths: list[Path] = []
    for raw_path in input_spec.split(","):
        path = Path(raw_path.strip()).expanduser()
        if not raw_path.strip():
            raise ValueError("Input paths cannot be empty.")
        if path.is_dir():
            folder_images = sorted(
                (candidate.resolve() for candidate in path.iterdir()
                 if candidate.is_file() and candidate.name.endswith(".nii.gz")),
                key=lambda candidate: candidate.name,
            )
            if not folder_images:
                raise ValueError(f"Input directory '{path}' contains no .nii.gz files.")
            image_paths.extend(folder_images)
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Input image '{path}' does not exist or is not a file.")
        if not is_nifti(str(path)):
            raise ValueError(f"Input '{path}' is not a valid NIfTI (.nii or .nii.gz).")
        image_paths.append(path.resolve())
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("The same input image was supplied more than once.")
    return image_paths


def _build_inference_cases(
    input_spec: str,
    output_dir: Path,
    in_channels: int,
    overwrite: bool,
) -> list[InferenceCase]:
    """Group channel-suffixed files and reject ambiguous or unsafe outputs."""
    grouped: dict[tuple[Path, str], dict[int, Path]] = {}
    for path in _expand_input_paths(input_spec):
        stem = _nifti_stem(path)
        match = _CHANNEL_SUFFIX.fullmatch(stem)
        case_name = match.group("case") if match else stem
        channel = int(match.group("channel")) if match else 0
        channels = grouped.setdefault((path.parent, case_name), {})
        if channel in channels:
            raise ValueError(f"Case '{case_name}' has more than one channel {channel:04d}.")
        channels[channel] = path

    cases: list[InferenceCase] = []
    outputs: set[Path] = set()
    for (_, case_name), channel_map in grouped.items():
        channel_ids = sorted(channel_map)
        if channel_ids != list(range(len(channel_ids))):
            raise ValueError(f"Case '{case_name}' channels must start at _0000 and be contiguous.")
        if len(channel_ids) != in_channels:
            raise ValueError(
                f"Case '{case_name}' has {len(channel_ids)} input channel(s), but the model requires {in_channels}."
            )
        image_paths = tuple(channel_map[channel] for channel in channel_ids)
        output_path = prediction_path(image_paths[0], output_dir)
        if output_path in outputs:
            raise ValueError(f"Multiple cases would write '{output_path}'.")
        outputs.add(output_path)
        if not overwrite and output_path.exists():
            logging.warning("Skipping '%s': output already exists at '%s'.", image_paths[0], output_path)
            continue
        cases.append(InferenceCase(image_paths=image_paths, output_path=output_path))
    return cases

def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for segmentation inference."""
    parser = argparse.ArgumentParser(
            description="Segment an input image according to the specified region.")

    # Required arguments
    required = parser.add_argument_group("Required")

    required.add_argument("-i", '--input_image', required=True, type=str,
                          help="Input image, folder of .nii.gz images, or a comma-separated mix of both.")

    required.add_argument("-r", '--region', required=False, default = 'wholebody', type=str,
                          help="Anatomical region to segment. Supported regions: wholebody, abdomen, pelvis, thigh, and leg. Default is wholebody.")
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    optional.add_argument("-o", '--output_dir', required=False, type=str,
                          help="Output directory. Channel-suffixed inputs such as case_0000.nii.gz produce case.nii.gz; other inputs produce *_dseg.nii.gz.")

    optional.add_argument("--overwrite", action="store_true",
                          help="Replace existing segmentation and color-table outputs.")

    optional.add_argument("-m", '--model', default=None, required=False, type=str,
                          help="Option to specify another model.")

    optional.add_argument("-s", '--overlap', required=False, default = 90, type=float,
                          help="Percent spatial overlap during sliding window inference, higher percent may improve accuracy but will reduce inference speed. Default is 90. If inference speed needs to be increased, the spatial overlap can be lowered. For large high-resolution or whole-body images, we recommend lowering the spatial inference to 50.")

    optional.add_argument("-c", '--chunk_size', required=False, default='auto', type=chunk_size_arg,
                          help="Number of axial slices to process per chunk, or 'auto' to size chunks from available CPU/GPU memory with a safety margin. Default is auto")

    return parser

# main: sets up logging, parses command-line arguments using parser, runs model, inference, post-processing
def main() -> None:
    """Run MuscleMap segmentation inference."""
    gc.collect()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addFilter(lambda r: r.levelno != logging.WARNING)
    logging.info("-" * 60)

    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Processing using cuda or cpu: {device}")

    amp_context = torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext()

    if device.type =='cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("Processing on a CPU will slow down inference speed")

    output = Path.cwd() if args.output_dir is None else Path(args.output_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)

    logging.info("Loading configuration file...")

    model_path, model_config_path = get_model_and_config_paths(args.region, args.model)

    model_config = ModelConfig.load_config(Path(model_config_path))
    logging.info(f"Task: Segmentation  |  Region: {args.region.capitalize()}")

    try:
        test_cases = _build_inference_cases(
            args.input_image,
            output,
            model_config.architecture.in_channels,
            args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logging.error("Input validation failed: %s", exc)
        sys.exit(1)
    logging.info("Discovered %s inference case(s).", len(test_cases))

    norm_map = {
            "instance": Norm.INSTANCE,  # pyright: ignore[reportUnknownMemberType]
            }
    pix_dim = model_config.image.spacing
    spatial_dims = model_config.architecture.spatial_dims
    out_channels = len(set(model_config.dataset.labels.values()))

    labels = sorted(set(model_config.dataset.labels.values()))
    id_map = {0: 0}
    for new_id, orig in enumerate((label for label in labels if label > 0), start=1):
        id_map[orig] = new_id
    inv_id_map = {new_id: orig for orig, new_id in id_map.items()}

    import_norm = norm_map[model_config.architecture.norm]

    if spatial_dims == 2:
        pad_size = (*model_config.image.roi_size[0: 2], 1)
    elif spatial_dims == 3:
        pad_size = model_config.image.roi_size
    else:
        logging.error(f"Unsupported spatial_dims: {spatial_dims}")
        sys.exit(1)

    pre_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pix_dim, mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        CropForegroundd(keys=["image"], source_key="image", margin=20),
        SpatialPadd(
            keys=["image"],
            spatial_size=pad_size,
            method="end",
            mode="constant"),
        EnsureTyped(keys=["image"]),
        ])

    post_transform_device = torch.device("cpu")
    post_transforms_list: list[MapTransform] = [
            Invertd(
                keys="pred", transform= pre_transforms, orig_keys="image",
                meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict", nearest_interp=False,
                to_tensor=True, device=post_transform_device
                ),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeTransform(keys=["pred"])]

    post_transforms_list.extend([
        RemapLabels(keys=["pred"], id_map=inv_id_map)])

    post_transforms = Compose(post_transforms_list)
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    model = UNet(
            spatial_dims=spatial_dims,
            in_channels=model_config.architecture.in_channels,
            out_channels=out_channels,
            channels=model_config.architecture.channels,
            act=model_config.architecture.act,
            strides=model_config.architecture.strides,
            num_res_units=model_config.architecture.num_res_units,
            norm=import_norm)

    model.load_state_dict(state)
    del state
    gc.collect()
    model = model.to(device)
    model.eval()
    if device.type == "cuda":
        # Dynamic shapes are only needed when one compiled model serves multiple cases.
        model = cast(torch.nn.Module, torch.compile(model, dynamic=len(test_cases) > 1))  # pyright: ignore[reportUnknownMemberType]

    overlap_inference = args.overlap / 100
    if spatial_dims == 2:
        inferer = SliceInferer(
            roi_size=model_config.image.roi_size,
            sw_batch_size=model_config.image.spatial_window_batch_size,
            spatial_dim=2,
            mode="gaussian",
            overlap=overlap_inference,
        )
    else:
        inferer = SlidingWindowInferer(
            roi_size=model_config.image.roi_size,
            sw_batch_size=model_config.image.spatial_window_batch_size,
            mode="gaussian",
            overlap=overlap_inference,
        )
    chunk_size = args.chunk_size
    for test_case in test_cases:
        logging.info("Processing %s", ", ".join(str(path) for path in test_case.image_paths))
        t0 = perf_counter()
        try:
            run_inference(
                image_path=test_case.image_paths,
                output_dir=output,
                pre_transforms=pre_transforms,
                post_transforms=post_transforms,
                model=model,
                amp_context=amp_context,
                chunk_size=chunk_size,
                device=device,
                inferer=inferer,
                out_channels=out_channels,
                target_pixdim=pix_dim,
                labels=model_config.dataset.labels,
            )
            logging.info("Inference of %s finished in %.2fs", test_case.output_path, perf_counter() - t0)
        except Exception as e:
            logging.exception("Error processing %s: %s", test_case.image_paths, e)
            continue
# %%
    logging.info("-" * 60)
    logging.info("Inference completed. All outputs saved.")
if __name__ == "__main__":
    main()
