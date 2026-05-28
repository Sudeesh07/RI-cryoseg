"""
Compare four trained model variants on the validation set.

Loads the best.pt checkpoint for each variant and evaluates on TS_99_9
(the reserved validation run).  Prints a per-class and overall F-beta
comparison table and writes results to compare_results.json.

The four variants form a 2x2 ablation that disentangles architecture
(SO3 vs SO2) from target shape (isotropic vs anisotropic z):

                       targets iso   |   targets aniso
    SO(3) arch  :        so3         |     so3_aniso
    SO(2) arch  :        so2_iso     |     so2

Reading the table:
  so3 vs so2_iso        : architecture effect with isotropic targets
  so3_aniso vs so2      : architecture effect with anisotropic targets
  so3 vs so3_aniso      : target effect inside SO(3)
  so2_iso vs so2        : target effect inside SO(2)

Usage (after training all four variants):
    python compare_variants.py

Or point to specific checkpoints:
    python compare_variants.py \\
        --so3       ./checkpoints/so3/best.pt \\
        --so3_aniso ./checkpoints/so3_aniso/best.pt \\
        --so2       ./checkpoints/so2/best.pt \\
        --so2_iso   ./checkpoints/so2_iso/best.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config import Config, ModelConfig, DataConfig
from dataset import (
    CZIIDataset, DataLoader, build_dataloaders,
    discover_run_names,
)
from model import build_model, EquivariantUNet3D, SO2EquivariantUNet3D
from utils import (
    CLASS_NAMES, PROTEIN_CLASSES, CLASS_WEIGHTS,
    extract_centroids, compute_fbeta_per_class, compute_weighted_fbeta,
)


VARIANT_SPECS = {
    "so3": {
        "model_variant": "so3",
        "z_sigma_scale": 1.0,
        "label": "SO(3)-equivariant  (isotropic targets)",
    },
    "so3_aniso": {
        "model_variant": "so3_aniso",
        "z_sigma_scale": 1.5,
        "label": "SO(3)-equivariant  (aniso z targets)",
    },
    "so2": {
        "model_variant": "so2",
        "z_sigma_scale": 1.5,
        "label": "SO(2)-equivariant  (aniso z targets)",
    },
    "so2_iso": {
        "model_variant": "so2_iso",
        "z_sigma_scale": 1.0,
        "label": "SO(2)-equivariant  (isotropic targets)",
    },
}


def build_cfg_for_variant(variant: str) -> Config:
    """Return a Config pre-configured for the given variant."""
    spec = VARIANT_SPECS[variant]
    cfg = Config()
    cfg.model.model_variant = spec["model_variant"]
    cfg.data.z_sigma_scale  = spec["z_sigma_scale"]
    return cfg


def load_variant(ckpt_path: str, variant: str) -> torch.nn.Module:
    """Load a checkpoint into the correct model class."""
    cfg = build_cfg_for_variant(variant)
    model = build_model(cfg.model, use_equivariant=True)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded {variant:10s} from {ckpt_path}  (epoch {epoch})")
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    score_thresholds: Dict[str, float],
    device: torch.device,
    nms_kernel_sizes: Optional[Dict[str, int]] = None,
) -> Dict:
    """Run the model over the val loader and compute F-beta per class + overall."""
    model.to(device)
    model.eval()

    all_pred = {cls: [] for cls in CLASS_NAMES}
    all_gt   = {cls: [] for cls in CLASS_NAMES}

    for volumes, targets in val_loader:
        volumes = volumes.to(device)
        preds   = model(volumes)

        for b in range(preds.shape[0]):
            pred_np   = preds[b].cpu().numpy()
            target_np = targets[b].cpu().numpy()
            pred_dets = extract_centroids(pred_np, score_thresholds, nms_kernel_sizes=nms_kernel_sizes)
            gt_dets   = extract_centroids(
                target_np, {c: 0.5 for c in CLASS_NAMES},
                nms_kernel_sizes=nms_kernel_sizes, subvoxel=False,
            )
            for cls in CLASS_NAMES:
                if len(pred_dets[cls]) > 0:
                    all_pred[cls].append(pred_dets[cls][:, :3])
                if len(gt_dets[cls]) > 0:
                    all_gt[cls].append(gt_dets[cls][:, :3])

    merged_pred = {
        cls: np.concatenate(all_pred[cls], axis=0) if all_pred[cls] else np.empty((0, 3))
        for cls in CLASS_NAMES
    }
    merged_gt = {
        cls: np.concatenate(all_gt[cls], axis=0) if all_gt[cls] else np.empty((0, 3))
        for cls in CLASS_NAMES
    }

    per_class = compute_fbeta_per_class(merged_pred, merged_gt)
    overall   = compute_weighted_fbeta(merged_pred, merged_gt)
    n_pred    = {cls: len(merged_pred[cls]) for cls in CLASS_NAMES}
    n_gt      = {cls: len(merged_gt[cls])  for cls in CLASS_NAMES}

    return {
        "per_class_fbeta": per_class,
        "weighted_fbeta":  overall,
        "n_pred":          n_pred,
        "n_gt":            n_gt,
    }


def print_comparison_table(results: Dict[str, Dict]):
    """Print a formatted comparison table."""
    variants = list(results.keys())
    cls_w    = 22

    # Header
    print("\n" + "=" * (cls_w + 14 * len(variants)))
    print(f"{'':>{cls_w}}", end="")
    for v in variants:
        print(f"{v:>14}", end="")
    print()
    print("-" * (cls_w + 14 * len(variants)))

    # Per-class F-beta
    print(f"  {'F-beta per class':>{cls_w - 2}}", end="")
    for _ in variants:
        print(" " * 14, end="")
    print()
    for cls in CLASS_NAMES:
        print(f"  {cls:<{cls_w - 2}}", end="")
        for v in variants:
            fb = results[v]["per_class_fbeta"].get(cls, 0.0)
            print(f"{fb:>14.4f}", end="")
        print()

    print("-" * (cls_w + 14 * len(variants)))

    # Weighted overall
    print(f"  {'WEIGHTED F-beta':>{cls_w - 2}}", end="")
    for v in variants:
        fb = results[v]["weighted_fbeta"]
        print(f"{fb:>14.4f}", end="")
    print()
    print("=" * (cls_w + 14 * len(variants)))

    # Annotation counts (sanity check)
    print(f"\n  Ground-truth counts on validation set:")
    for cls in CLASS_NAMES:
        n = next(iter(results.values()))["n_gt"][cls]
        print(f"    {cls:<25} {n:>5}")

    # Verdict
    best_v  = max(variants, key=lambda v: results[v]["weighted_fbeta"])
    best_fb = results[best_v]["weighted_fbeta"]
    print(f"\n  Best variant: {best_v}  (weighted F-beta = {best_fb:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare trained model variants")
    parser.add_argument("--so3",       default=None,
                        help="Path to so3 best.pt (default: checkpoints/so3/best.pt)")
    parser.add_argument("--so3_aniso", default=None,
                        help="Path to so3_aniso best.pt")
    parser.add_argument("--so2",       default=None,
                        help="Path to so2 best.pt")
    parser.add_argument("--so2_iso",   default=None,
                        help="Path to so2_iso best.pt")
    parser.add_argument("--dataset",   default="./dataset/10440")
    parser.add_argument("--output",    default="compare_results.json")
    parser.add_argument("--device",    default=None,
                        help="cuda | cpu (auto-detected if omitted)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\n")

    ckpt_paths = {
        "so3":       args.so3       or "checkpoints/so3/best.pt",
        "so3_aniso": args.so3_aniso or "checkpoints/so3_aniso/best.pt",
        "so2":       args.so2       or "checkpoints/so2/best.pt",
        "so2_iso":   args.so2_iso   or "checkpoints/so2_iso/best.pt",
    }

    # Load available checkpoints (skip missing ones)
    models = {}
    for v, path in ckpt_paths.items():
        if not Path(path).exists():
            print(f"  {v:10s}: checkpoint not found at {path} — skipping")
            continue
        models[v] = load_variant(path, v)

    if not models:
        print("No checkpoints found. Train at least one variant first.")
        return

    # Build val DataLoader — reuse so3 config (val set doesn't change with variant)
    print("\nBuilding validation DataLoader (TS_99_9)...")
    base_cfg = Config()
    base_cfg.data.dataset_dir = args.dataset
    _, val_loader = build_dataloaders(base_cfg, rank=0, world_size=1)
    if val_loader is None:
        print("No validation runs found. Check --dataset path.")
        return
    print(f"Val batches: {len(val_loader)}")

    score_thresholds = base_cfg.inference.score_thresholds
    nms_kernel_sizes = getattr(base_cfg.inference, "nms_kernel_sizes", None)

    # Evaluate each variant
    results = {}
    for v, model in models.items():
        print(f"\nEvaluating {v}...")
        results[v] = evaluate_model(model, val_loader, score_thresholds, device, nms_kernel_sizes)
        fb = results[v]["weighted_fbeta"]
        print(f"  → weighted F-beta: {fb:.4f}")

    # Print comparison table
    print_comparison_table(results)

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(
            {v: {
                "per_class_fbeta": r["per_class_fbeta"],
                "weighted_fbeta":  r["weighted_fbeta"],
                "label":           VARIANT_SPECS[v]["label"],
             }
             for v, r in results.items()},
            f, indent=2,
        )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
