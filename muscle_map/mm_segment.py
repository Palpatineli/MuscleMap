#!/usr/bin/env python
from ntpath import exists
from pathlib import Path
import warnings
import argparse
import logging
import os
import gc
from contextlib import nullcontext
import sys
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
    check_image_exists,
    get_model_and_config_paths,
    is_nifti,
    run_inference,
)

warnings.filterwarnings("ignore")
print("Command line arguments received:", sys.argv)

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

#naming not functional
# get_parser: parses command line arguments, sets up a) required (image, body region), and b) optional arguments (model, output file name, output directory)
def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for segmentation inference."""
    parser = argparse.ArgumentParser(
            description="Segment an input image according to the specified region.")

    # Required arguments
    required = parser.add_argument_group("Required")

    required.add_argument("-i", '--input_image', required=True, type=str,
                          help="Input image to segment. Can be single image or list of images separated by commas.")

    required.add_argument("-r", '--region', required=False, default = 'wholebody', type=str,
                          help="Anatomical region to segment. Supported regions: wholebody, abdomen, pelvis, thigh, and leg. Default is wholebody.")
    # Optional arguments
    optional = parser.add_argument_group("Optional")
    required.add_argument("-o", '--output_dir', required=False, type=str,
                          help="Output directory to save the results, output file name suffix = dseg. If left empty, saves to current working directory.")

    optional.add_argument("-m", '--model', default=None, required=False, type=str,
                          help="Option to specify another model.")

    optional.add_argument("--model_version", default="latest", required=False, type=str,
                          help="Model version to use, e.g. '1.3'. Default: latest available on Zenodo.")

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

    if (output := Path.cwd() if (output := args.output_dir) is None else Path(output).absolute()):
        output.parent.mkdir(exist_ok=True)

    image_paths = [image.strip() for image in args.input_image.split(',')]
    for image_path in image_paths:
        logging.info(f"Checking if image '{image_path}' exists and is readable...")
        check_image_exists(image_path)
        if not is_nifti(image_path):
            logging.error(f"Error: {image_path} is not a valid NIfTI (.nii or .nii.gz)")
            sys.exit(1) 

    logging.info("Loading configuration file...")

    model_path, model_config_path = get_model_and_config_paths(args.region, args.model, args.model_version)

    model_config = ModelConfig.load_config(Path(model_config_path))
    model_version = model_config.architecture.version
    logging.info(f"Task: Segmentation  |  Region: {args.region.capitalize()}  |  Model version: {model_version}")

    norm_map = {
            "instance": Norm.INSTANCE,  # pyright: ignore[reportUnknownMemberType]
            }
    # TODO: find a place for data fingerprint, and to calculate pix_dim, should use _resolve_pix_dim
    pix_dim = tuple(model_config['parameters']['pix_dim'])
    spatial_dims = model_config.architecture.spatial_dims
    out_channels = len(model_config.dataset.labels) - 1

    labels = sorted(list(model_config.dataset.labels.values()))
    id_map = {0: 0}
    for new_id, orig in enumerate(labels, start=1):
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
        Spacingd(keys=["image"], pixdim=model_config.dataset.pix_dim, mode="bilinear"),
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

    test_files = [{"image": image} for image in image_paths]

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
    for test in test_files:
        logging.info(f"Processing {test['image']}")
        t0 = perf_counter()
        try:
            run_inference(
                    test["image"],
                    output,
                    pre_transforms,
                    post_transforms,
                    amp_context,
                    chunk_size,
                    device,
                    inferer,
                    model,
                    out_channels=out_channels,
                    target_pixdim=pix_dim,
                    )
            logging.info(f"Inference of {test} finished in {perf_counter()-t0:.2f}s")
        except Exception as e:
            logging.exception(f"Error processing {test['image']}: {e}")
            continue
# %%
    logging.info("-" * 60)
    logging.info("Inference completed. All outputs saved.")
if __name__ == "__main__":
    main()
