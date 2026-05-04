"""
Inference Pipeline for Protein Centroid Detection.

Performs sliding-window inference over full tomograms, aggregates predictions
with Gaussian-weighted stitching, applies 3D NMS, and extracts centroids.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, InferenceConfig, CLASS_NAMES, NUM_CLASSES
from dataset import load_tomogram, normalize_tomogram, CZIIInferenceDataset
from model import build_model
from utils import (
    aggregate_predictions,
    extract_centroids,
    generate_submission,
    nms_3d,
)


@torch.no_grad()
def predict_tomogram(
    model: torch.nn.Module,
    volume: np.ndarray,
    cfg: InferenceConfig,
    device: torch.device,
    batch_size: int = 4,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Run sliding-window inference on a full tomogram.

    Args:
        model: trained detection model
        volume: (D, H, W) normalized tomogram
        cfg: inference configuration
        device: torch device
        batch_size: patches per forward pass
        use_amp: use mixed precision

    Returns:
        (C, D, H, W) heatmap predictions aggregated over the full volume
    """
    model.eval()

    dataset = CZIIInferenceDataset(
        volume,
        patch_size=cfg.patch_size,
        overlap=cfg.overlap,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    patch_preds: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

    for patches, origins in tqdm(loader, desc="Inference"):
        patches = patches.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            preds = model(patches)  # (B, C, pD, pH, pW)

        preds = preds.float().cpu()

        for i in range(preds.shape[0]):
            origin = tuple(origins[i].numpy().tolist())
            patch_preds.append((preds[i].numpy(), origin))

    # Aggregate with Gaussian weighting
    full_heatmap = aggregate_predictions(
        patch_preds,
        volume_shape=volume.shape,
        num_classes=NUM_CLASSES,
    )

    return full_heatmap


def detect_proteins(
    heatmap: np.ndarray,
    cfg: InferenceConfig,
) -> Dict[str, np.ndarray]:
    """
    Post-process heatmap to extract protein centroids.

    Args:
        heatmap: (C, D, H, W) predicted heatmap
        cfg: inference configuration

    Returns:
        {class_name: (N, 4) array of [z, y, x, score]}
    """
    detections = extract_centroids(
        heatmap,
        score_thresholds=cfg.score_thresholds,
        nms_kernel_sizes=getattr(cfg, "nms_kernel_sizes", None),
    )

    # Print detection counts
    print("Detections per class:")
    for cls_name, dets in detections.items():
        print(f"  {cls_name}: {len(dets)}")

    return detections


# ---------------------------------------------------------------------------
# Rotation-Averaged Inference (Test-Time Augmentation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_rotation_tta(
    model: torch.nn.Module,
    volume: np.ndarray,
    cfg: InferenceConfig,
    device: torch.device,
    batch_size: int = 4,
    use_amp: bool = True,
    num_rotations: int = 4,
) -> np.ndarray:
    """
    Rotation-averaged (TTA) inference.

    Even with an equivariant model, TTA can help at boundaries.
    For a non-equivariant baseline, this provides rotation robustness.

    Applies 90-degree rotations in the XY plane, predicts on each,
    inverts the rotation, and averages.
    """
    heatmaps = []

    for k in range(num_rotations):
        # Rotate volume by k*90 degrees in XY plane
        vol_rot = np.rot90(volume, k=k, axes=(1, 2)).copy()

        # Predict
        hm = predict_tomogram(model, vol_rot, cfg, device, batch_size, use_amp)

        # Inverse rotation on heatmap
        hm_back = np.rot90(hm, k=-k, axes=(2, 3)).copy()
        heatmaps.append(hm_back)

    # Average
    return np.mean(heatmaps, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run inference on cryo-ET tomograms")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Tomogram file or directory")
    parser.add_argument("--output", type=str, default="./predictions", help="Output directory")
    parser.add_argument("--submission", type=str, default="submission.csv", help="Submission CSV path")
    parser.add_argument("--baseline", action="store_true", help="Use standard U-Net baseline")
    parser.add_argument("--tta", action="store_true", help="Use rotation TTA")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model and load weights
    use_equivariant = not args.baseline
    model = build_model(cfg.model, use_equivariant=use_equivariant)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Discover tomograms
    input_path = Path(args.input)
    if input_path.is_file():
        tomo_paths = [input_path]
    else:
        tomo_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix in (".mrc", ".rec", ".npy") or p.is_dir()
        )
    print(f"Found {len(tomo_paths)} tomogram(s)")

    # Run inference
    os.makedirs(args.output, exist_ok=True)
    all_detections = {}
    inf_cfg = cfg.inference

    for tomo_path in tomo_paths:
        tomo_id = tomo_path.stem
        print(f"\n{'=' * 60}")
        print(f"Processing: {tomo_id}")

        # Load and normalize
        volume = load_tomogram(tomo_path)
        volume = normalize_tomogram(volume, cfg.data.clip_percentile)
        print(f"Volume shape: {volume.shape}")

        # Predict
        if args.tta:
            heatmap = predict_with_rotation_tta(
                model, volume, inf_cfg, device, args.batch_size,
                cfg.train.use_amp,
            )
        else:
            heatmap = predict_tomogram(
                model, volume, inf_cfg, device, args.batch_size,
                cfg.train.use_amp,
            )

        # Save raw heatmap
        hm_path = os.path.join(args.output, f"{tomo_id}_heatmap.npy")
        np.save(hm_path, heatmap)
        print(f"Heatmap saved: {hm_path}")

        # Extract centroids
        detections = detect_proteins(heatmap, inf_cfg)
        all_detections[tomo_id] = detections

        # Save per-tomo detections
        det_path = os.path.join(args.output, f"{tomo_id}_detections.npy")
        np.save(det_path, detections)

    # Generate submission CSV
    generate_submission(
        all_detections,
        output_path=args.submission,
        voxel_spacing=cfg.data.voxel_spacing,
    )

    print(f"\nInference complete. Submission: {args.submission}")


if __name__ == "__main__":
    main()
