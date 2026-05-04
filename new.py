"""
Dataset verification and statistics script.

Run this to confirm the dataset is correctly structured and that the
annotation loader reads the expected particle counts.

Usage:
    python new.py
    python new.py --dataset ./dataset/10440
"""

import argparse
from pathlib import Path
import numpy as np

from config import CLASS_NAMES, PROTEIN_CLASSES, NUM_CLASSES, CLASS_SIGMAS_VOXELS
from dataset import (
    discover_run_names,
    load_dataset_annotations,
    get_tomogram_path,
    load_tomogram,
    generate_multiclass_heatmap,
)


def summarise_annotations(dataset_dir: str) -> None:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"{'='*60}")

    run_names = discover_run_names(dataset_dir)
    print(f"Runs found ({len(run_names)}): {run_names}\n")

    all_annotations = load_dataset_annotations(dataset_dir)

    # Header
    col_w = 22
    print(f"{'Run':<14}", end="")
    for cls in CLASS_NAMES:
        print(f"{cls[:col_w]:>{col_w}}", end="")
    print()
    print("-" * (14 + col_w * NUM_CLASSES))

    total_per_class = {cls: 0 for cls in CLASS_NAMES}

    for run_name in run_names:
        annot = all_annotations.get(run_name, {})
        print(f"{run_name:<14}", end="")
        for cls in CLASS_NAMES:
            n = len(annot.get(cls, []))
            total_per_class[cls] += n
            print(f"{n:>{col_w}}", end="")
        print()

    print("-" * (14 + col_w * NUM_CLASSES))
    print(f"{'TOTAL':<14}", end="")
    for cls in CLASS_NAMES:
        print(f"{total_per_class[cls]:>{col_w}}", end="")
    print()

    print(f"\nPer-class sigma (voxels at 10 Å/vox):")
    for cls, sigma in zip(CLASS_NAMES, CLASS_SIGMAS_VOXELS):
        radius = PROTEIN_CLASSES[cls]["radius_angstrom"]
        print(f"  {cls:<25}  sigma={sigma:.1f}  radius={radius:.0f} Å  "
              f"({radius/10.0:.1f} vox)")


def check_tomogram(dataset_dir: str, run_name: str = "TS_5_4") -> None:
    print(f"\n{'='*60}")
    print(f"Tomogram check: {run_name}")
    print(f"{'='*60}")

    path = get_tomogram_path(dataset_dir, run_name)
    print(f"Path: {path}")

    vol = load_tomogram(path)
    print(f"Shape (D, H, W): {vol.shape}")
    print(f"Dtype: {vol.dtype}")
    print(f"Value range: [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"Mean ± std: {vol.mean():.3f} ± {vol.std():.3f}")

    # Check heatmap generation
    all_annotations = load_dataset_annotations(dataset_dir)
    annot = all_annotations.get(run_name, {})
    hm = generate_multiclass_heatmap(vol.shape, annot)
    print(f"\nHeatmap shape: {hm.shape}")
    for cls in CLASS_NAMES:
        cls_id = PROTEIN_CLASSES[cls]["id"]
        ch = hm[cls_id]
        n_ann = len(annot.get(cls, []))
        print(f"  {cls:<25}  heatmap max={ch.max():.4f}  n_annotations={n_ann}")


def main():
    parser = argparse.ArgumentParser(description="Dataset verification")
    parser.add_argument("--dataset", default="./dataset/10440", help="Dataset directory")
    parser.add_argument("--run", default="TS_5_4", help="Run to check in detail")
    args = parser.parse_args()

    summarise_annotations(args.dataset)
    check_tomogram(args.dataset, args.run)
    print("\nDataset verification complete.")


if __name__ == "__main__":
    main()
