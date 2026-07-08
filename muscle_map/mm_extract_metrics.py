from typing import Any, cast, Final, Literal
from collections.abc import Mapping
from pathlib import Path
import warnings
import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
from nibabel import Nifti1Header

from muscle_map.mm_util import (
    apply_clustering,
    DatasetParameter,
    ModelConfig,
    add_slice_counts,
    build_entry_dict_metrics,
    calculate_metrics_average,
    calculate_metrics_dixon,
    create_output_dir,
    extract_image_data,
    get_config_path,
    validate_extract_args,
)
warnings.filterwarnings("ignore")

def results_entry_to_dataframe(results_entry: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for lbl, entry in results_entry.items():
        label_val = int(entry.get("Label", lbl))
        row = {"Label": label_val}
        row.update(entry)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "Label" in df.columns:
        df = df.drop_duplicates(subset=["Label"]).sort_values("Label")
    return df


def calculate_thresholds(labels, mask_img, num_clusters: Literal[2, 3]):
    clusters = [mask_img[labels == i] for i in range(num_clusters)]
    means = [np.mean(cluster) for cluster in clusters]
    if num_clusters == 2:
        muscle_max = np.max(clusters[0]) if means[0] < means[1] else np.max(clusters[1])
        muscle_img = mask_img[mask_img <= muscle_max]
        fat_min = None # placeholder
        sorted_indices = [0, 1] if means[0] < means[1] else [1, 0]
    elif num_clusters == 3:
        sorted_clusters = sorted(zip(means, clusters, range(len(clusters))), key=lambda x: x[0])
        muscle_img = sorted_clusters[0][1]
        fat_img = sorted_clusters[2][1]
        muscle_max = np.max(muscle_img)
        fat_min= np.min(fat_img)
        sorted_indices = [x[2] for x in sorted_clusters]
    return muscle_max, fat_min, sorted_indices

def _build_anatomy_groups(
    model_config: ModelConfig,
    results_entry: Mapping[int, Mapping[str, Any]],
    cluster_data: dict[int, str],
) -> dict[str, list[int]]:
    """
    Groups cluster_data labels by anatomy name (left + right together).

    Priority:
    1. model_config: groups on `anatomy` field (without side), Title Case.
    2. results_entry fallback: strips trailing ' left' / ' right' from anatomy string.
    3. Last resort: 'Label <id>'.
    """
    _SIDES = (" left", " right", " Left", " Right")

    groups: dict[str, list[int]] = {}

    if model_config:
        for L in model_config.get("labels", []):
            try:
                val = int(L.get("value"))
            except Exception:
                continue
            if val not in cluster_data:
                continue
            anatomy = str(L.get("anatomy", "")).strip().title() or f"Label {val}"
            groups.setdefault(anatomy, []).append(val)

    # Cover labels not matched by model_config (or when model_config is None)
    for lbl in cluster_data:
        if any(lbl in g for g in groups.values()):
            continue
        raw = results_entry.get(lbl, {}).get("Anatomy", "")
        if raw:
            for suffix in _SIDES:
                if raw.endswith(suffix):
                    raw = raw[: -len(suffix)]
                    break
            anatomy = raw.strip().title()
        else:
            anatomy = f"Label {lbl}"
        groups.setdefault(anatomy, []).append(lbl)

    return groups

def calculate_metrics_thresholding(
    args,
    results_entry: dict[int, Any],
    label_img: np.ndarray,
    img_array: np.ndarray,
    affine: np.ndarray,
    header:  Nifti1Header,
    pix_dim: tuple[float, float, float],
    components: int,
    output_dir: str | Path,
    id_part: str,
    qc: bool,
    dataset_config: DatasetParameter,
) -> dict[int, Any]:

    # raise value errors if components is not 2/3 or when mismatch in shape
    if components not in (2, 3):
        raise ValueError("components must be 2 or 3")
    if label_img.shape != img_array.shape:
        raise ValueError("label_img and img_array must have the same shape")

    #prepare empty image array to build up the fat, muscle (and in 3 component; undefined)maps
    total_muscle_image    = np.zeros_like(img_array, dtype=bool)
    total_fat_image       = np.zeros_like(img_array, dtype=bool)
    total_undefined_image = np.zeros_like(img_array, dtype=bool)
    combined_mask = np.zeros_like(label_img, dtype=np.uint8)

    # if GMM is chosen, we will also create an empty array in float32 for each component to store softprob.
    method: Final[str] = cast(str, args.method)
    if method == 'gmm':
        total_probability_maps = [np.zeros(label_img.shape, dtype=np.float32)
                                for _ in range(components)]

    # determine voxel vol ml to easily calculate volume from pixdim
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # --- Phase 1: cluster all labels and collect thresholds ---
    # cluster_data: {lbl: (muscle_max, fat_min, sorted_indices, clustering, mask_img_1d, mask_3d)}
    cluster_data = {}
    for lbl in np.unique(label_img):
        if lbl == 0:
            continue
        mask    = (label_img == lbl)
        mask_img = img_array[mask].reshape(-1, 1)
        labels_cl, clustering = apply_clustering(args, mask_img, components)
        muscle_max, fat_min, sorted_indices = calculate_thresholds(labels_cl, mask_img, components)
        cluster_data[int(lbl)] = (muscle_max, fat_min, sorted_indices, clustering, mask_img, mask)

    # --- QC: one window per anatomy group (left + right combined) ---
    erased_masks: dict[int, np.ndarray] = {}   # {lbl: 3D bool mask}
    if qc:
        from muscle_map.mm_qc_gui import QCManager
        _qc_manager = QCManager()
        anatomy_groups = _build_anatomy_groups(dataset_config, results_entry, cluster_data)
        for anatomy_name, group_lbls in anatomy_groups.items():
            group_thresholds = {lbl: (cluster_data[lbl][0], cluster_data[lbl][1]) for lbl in group_lbls}
            muscle_delta, fat_delta, erased_mask = _qc_manager.show(
                img_array, label_img, group_thresholds, components, anatomy_name
            )
            for lbl in group_lbls:
                d = cluster_data[lbl]
                cluster_data[lbl] = (
                    d[0] + muscle_delta,
                    (d[1] + fat_delta) if d[1] is not None else None,
                    d[2], d[3], d[4], d[5],
                )
                if erased_mask.any():
                    erased_masks[lbl] = erased_mask
            if _qc_manager.quit_requested:
                break
        _qc_manager.destroy()

    # --- Phase 2: compute metrics and build output maps ---
    for lbl, (muscle_max, fat_min, sorted_indices, clustering, mask_img, mask) in cluster_data.items():
        if lbl in erased_masks:
            mask     = mask & ~erased_masks[lbl]
            mask_img = img_array[mask].reshape(-1, 1)
        N            = mask_img.size
        total_volume = N * voxel_vol_ml

        erased = erased_masks.get(lbl)

        entry = results_entry.get(lbl, {"Anatomy": "", "Label": lbl})
        if components == 2:
            muscle_array, fat_array, _ = create_image_array(img_array, label_img, lbl, muscle_max, fat_min, components)
            if erased is not None:
                muscle_array = muscle_array & ~erased
                fat_array    = fat_array    & ~erased
            total_muscle_image |= muscle_array
            total_fat_image    |= fat_array
            combined_mask[muscle_array] = 1
            combined_mask[fat_array]    = 4

            muscle_percentage = 100.0 * np.mean((mask_img.ravel() <= muscle_max))
            fat_percentage    = 100 - muscle_percentage
            muscle_voxels     = np.count_nonzero(mask_img <= muscle_max)
            muscle_volume     = muscle_voxels * voxel_vol_ml
            fat_volume        = (N - muscle_voxels) * voxel_vol_ml
        elif components == 3:
            muscle_array, fat_array, undefined_array = create_image_array(img_array, label_img, lbl, muscle_max, fat_min, components)
            if erased is not None:
                muscle_array    = muscle_array    & ~erased
                fat_array       = fat_array       & ~erased
                undefined_array = undefined_array & ~erased
            total_muscle_image    |= muscle_array
            total_fat_image       |= fat_array
            total_undefined_image |= undefined_array
            combined_mask[muscle_array]    = 1
            combined_mask[undefined_array] = 7
            combined_mask[fat_array]       = 4

            muscle_percentage    = np.nan if N == 0 else 100.0 * np.mean(mask_img <  muscle_max)
            undefined_percentage = np.nan if N == 0 else 100.0 * np.mean((mask_img >= muscle_max) & (mask_img < fat_min))
            fat_percentage       = np.nan if N == 0 else 100.0 * np.mean(mask_img >= fat_min)
            muscle_voxels        = np.count_nonzero(mask_img <= muscle_max)
            muscle_volume        = muscle_voxels * voxel_vol_ml
            undefined_voxels     = np.count_nonzero((mask_img > muscle_max) & (mask_img < fat_min))
            undefined_volume     = undefined_voxels * voxel_vol_ml
            fat_voxels           = np.count_nonzero(mask_img >= fat_min)
            fat_volume           = fat_voxels * voxel_vol_ml
            entry["Undefined (%)"]         = (np.nan if undefined_percentage is None else round(float(undefined_percentage), 2))
            entry["Undefined volume (ml)"] = round(float(undefined_volume),     2)

        entry.update({
            "Muscle (%)":         (np.nan if muscle_percentage is None else round(float(muscle_percentage), 2)),
            "Fat (%)":            (np.nan if fat_percentage is None else round(float(fat_percentage), 2)),
            "Total volume (ml)":  (np.nan if total_volume is None else round(float(total_volume), 2)),
            "Fat volume (ml)":    (np.nan if fat_volume is None else round(float(fat_volume), 2)),
            "Muscle volume (ml)": round(float(muscle_volume), 2),
        })
        results_entry[lbl] = entry

        if method == 'gmm':
            probability_maps        = clustering.predict_proba(mask_img)
            sorted_probability_maps = probability_maps[:, sorted_indices]

            for comp_idx in range(components):
                total_probability_maps[comp_idx][mask] += sorted_probability_maps[:, comp_idx].astype(np.float32)  # pyright: ignore[reportPossiblyUnboundVariable]

    save_nifti(total_muscle_image.astype(np.uint8), affine, header,
               os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_muscle_seg.nii.gz'))
    save_nifti(combined_mask, affine, header,
               os.path.join(output_dir, f"{id_part}_{args.method}_{components}component_combined_seg.nii.gz"))
    if components == 3:
        save_nifti(total_undefined_image.astype(np.uint8), affine, header,
                   os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_undefined_seg.nii.gz'))
    save_nifti(total_fat_image.astype(np.uint8), affine, header,
               os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_fat_seg.nii.gz'))

    if args.method == 'gmm':  # pyright: ignore[reportUnknownMemberType]
        if components == 3:
            component_names = ["muscle", "undefined", "fat"]
        else:
            component_names = ["muscle", "fat"]
        for comp_idx, comp_name in enumerate(component_names):
            out_path = os.path.join(
                output_dir,
                f"{id_part}_gmm_{comp_name}_{components}component_softseg.nii.gz"
            )
        save_nifti(total_probability_maps[comp_idx], affine, header, out_path)  # pyright: ignore[reportPossiblyUnboundVariable]

    return results_entry


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for muscle metrics extraction."""
    parser = argparse.ArgumentParser(description="Extract metrics of muscle size and composition")

    parser.add_argument("-m", '--method', required=True, type=str, choices=['dixon', 'kmeans', 'gmm', 'average'],
                          help="Method to use: kmeans, gmm, dixon, or average")

    parser.add_argument("-i", '--input_image', required=False, type=str,
                          help="Input image for kmeans, gmm, or average method")

    parser.add_argument("-f", '--fat_image', required=False, type=str,
                          help="Fat image for Dixon method")

    parser.add_argument("-w", '--water_image', required=False, type=str,
                          help="Water image for Dixon method")

    parser.add_argument("-s", '--segmentation_image', required=False, type=str,
                          help="Segmentation image for any method")

    parser.add_argument("-c", '--components', required=False, default=None, type=int, choices=[2, 3],
                          help="Number of components for kmeans or gmm (2 or 3)")

    parser.add_argument("-r", '--region', required=False, type=str,
                          help="Anatomical region. Supported regions: wholebody, abdomen, pelvis, thigh, and leg")

    parser.add_argument("-o", '--output_dir', required=False, type=str,
                          help="Output directory to save the results")

    parser.add_argument("--qc", action="store_true",
                          help="Open interactive QC window to adjust thresholds (kmeans/gmm only)")

    parser.add_argument("--model_version", default="latest", required=False, type=str,
                          help="Model version to use, e.g. '2.0'. Default: latest available on Zenodo.")

    return parser

def main() -> None:
    """Extract quantitative metrics from a segmentation mask."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addFilter(lambda r: r.levelno != logging.WARNING)
    logging.info("-" * 60)

    parser = get_parser()
    args = parser.parse_args()
    validate_extract_args(args)

    if args.qc and args.method not in ('gmm', 'kmeans'):
        logging.error("--qc is only supported with -m gmm or -m kmeans.")
        sys.exit(1)

    _, mask, affine, header, _, pix_dim = extract_image_data(args.segmentation_image)

    output_dir=create_output_dir(args.output_dir)

    if args.region:
        model_config_path = get_config_path(args.region, args.model_version)
        model_config = ModelConfig.load_config(model_config_path)
        model_version = model_config.architecture.version
        logging.info(f"Task: Quantification  |  Region: {args.region.capitalize()}  |  Model version: {model_version}")
    else:
        model_config = None
        logging.info("Task: Quantification  |  No region specified")

    if args.method == 'dixon':
        input_filename = os.path.basename(args.fat_image)
    else:
        input_filename = os.path.basename(args.input_image)

    id_part = input_filename[:-7] if input_filename.endswith('.nii.gz') else input_filename

    results_entry = build_entry_dict_metrics(mask, model_config, True)

    # calculate number of slices with segmentation and update results_entry dictionary.
    results_entry = add_slice_counts(results_entry, mask, pix_dim)

    if not np.any(mask):
        raise ValueError("No labels found in segmentation mask")
    else:
        if args.method == 'dixon':
            _, fat_array, _, _,_,_  = extract_image_data(args.fat_image)
            _, water_array, _, _,_,_ = extract_image_data(args.water_image)
            outputs = calculate_metrics_dixon(results_entry, mask, fat_array, water_array, pix_dim)
        elif args.method == 'average':
            _, image_array,_,_, _, _ = extract_image_data(args.input_image)
            outputs = calculate_metrics_average(results_entry, mask, image_array, pix_dim)
        elif args.method in ('kmeans', 'gmm'):
            _, image_array, _, _, _, _, = extract_image_data(args.input_image)
            number_of_components = args.components
            outputs = calculate_metrics_thresholding(args, results_entry, mask, image_array, affine, header,
                                                               pix_dim, number_of_components, output_dir,
                                                               id_part, qc=args.qc, dataset_config=model_config)
        else:
            raise ValueError(f"method must be dixon, average, kneams or gmm, got '{args.method}'")

    # Construct the path to the output CSV file
    if args.method != 'dixon' and args.method != 'average':
        output_filename = f"{id_part}_{args.method}_{args.components}component_results.csv"
    else:
        output_filename = f"{id_part}_{args.method}_results.csv"

    output_file_path = os.path.join(output_dir, output_filename)

    save_outputs = results_entry_to_dataframe(outputs)
    save_outputs.to_csv(
    output_file_path,
    index=False,
    sep=',')

    logging.info("-" * 60)
    logging.info(f"Results have been exported to {output_file_path}")

if __name__ == "__main__":
    main()
