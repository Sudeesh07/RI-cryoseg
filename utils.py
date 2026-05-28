"""
Utilities: 3D NMS, peak detection, F-beta metric, coordinate helpers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
from scipy.ndimage import maximum_filter

from config import (
    CLASS_NAMES, CLASS_RADII_VOXELS, CLASS_SIGMAS_VOXELS, CLASS_WEIGHTS,
    NUM_CLASSES, PROTEIN_CLASSES,
)


# ---------------------------------------------------------------------------
# 3D Non-Maximum Suppression
# ---------------------------------------------------------------------------

def nms_3d(
    heatmap: torch.Tensor,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    3D Non-Maximum Suppression via max-pooling.

    Keeps only local maxima: voxels that are the maximum within a
    (kernel_size)^3 neighborhood.

    Args:
        heatmap: (C, D, H, W) or (B, C, D, H, W)
        kernel_size: size of the max-pooling kernel

    Returns:
        Suppressed heatmap with non-maxima set to 0.
    """
    squeeze = False
    if heatmap.ndim == 4:
        heatmap = heatmap.unsqueeze(0)
        squeeze = True

    pad = kernel_size // 2
    pooled = F.max_pool3d(heatmap, kernel_size, stride=1, padding=pad)
    keep = (heatmap == pooled).float()
    result = heatmap * keep

    if squeeze:
        result = result.squeeze(0)
    return result


def nms_3d_numpy(
    heatmap: np.ndarray,
    kernel_size: int = 7,
) -> np.ndarray:
    """NumPy version of 3D NMS using scipy maximum_filter."""
    maxfilt = maximum_filter(heatmap, size=kernel_size)
    keep = heatmap == maxfilt
    return heatmap * keep


# ---------------------------------------------------------------------------
# Peak Detection (Centroid Extraction)
# ---------------------------------------------------------------------------

def _nms_kernel_for_class(cls_name: str, nms_kernel_sizes: Optional[Dict[str, int]] = None) -> int:
    """Return the NMS kernel size for a class.

    Uses the per-class dict if provided, otherwise falls back to
    2 * floor(radius_voxels) + 1 (minimum 3).
    """
    if nms_kernel_sizes is not None and cls_name in nms_kernel_sizes:
        return nms_kernel_sizes[cls_name]
    cls_id = PROTEIN_CLASSES[cls_name]["id"]
    radius = CLASS_RADII_VOXELS[cls_id]
    return max(3, 2 * int(radius) + 1)


def _subvoxel_radius_for_class(cls_name: str, max_radius: Optional[int] = None) -> int:
    """
    Pick a sub-voxel refinement neighbourhood that scales with the class
    Gaussian sigma.  A fixed radius=2 (used previously) is too small for
    ribosome-scale particles (sigma ≈ 6 vox) — the centre-of-mass loses
    most of the Gaussian mass.  We use ceil(2 * sigma), which captures
    ~95% of the mass on each axis.  Capped at max_radius so the window
    never exceeds the NMS half-kernel (otherwise it can pull in a nearby
    peak's mass).
    """
    cls_id = PROTEIN_CLASSES[cls_name]["id"]
    sigma = CLASS_SIGMAS_VOXELS[cls_id]
    r = max(2, int(np.ceil(2.0 * sigma)))
    if max_radius is not None:
        r = min(r, max_radius)
    return r


def _subvoxel_centroid(
    hm: np.ndarray,
    z: int,
    y: int,
    x: int,
    radius: int = 2,
) -> Tuple[float, float, float]:
    """
    Refine a peak coordinate to subvoxel precision using a weighted centroid
    over a small neighbourhood of the peak.

    This compensates for the discrete voxel grid: the true physical centroid
    of a Gaussian blob lies between voxels when the peak straddles a border.

    Args:
        hm:     (D, H, W) heatmap channel
        z,y,x:  integer peak coordinates
        radius: neighbourhood half-width — for centroid recovery to be
                accurate it should scale with the Gaussian sigma of the
                target class (see _subvoxel_radius_for_class).

    Returns:
        (z_sub, y_sub, x_sub) subvoxel coordinates
    """
    D, H, W = hm.shape
    z0, z1 = max(0, z - radius), min(D, z + radius + 1)
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)

    patch = hm[z0:z1, y0:y1, x0:x1].astype(np.float64)
    w = patch.sum()
    if w < 1e-12:
        return float(z), float(y), float(x)

    dz = np.arange(z0, z1)
    dy = np.arange(y0, y1)
    dx = np.arange(x0, x1)
    # Broadcasting: patch is (dz, dy, dx)
    z_sub = (patch.sum(axis=(1, 2)) * dz).sum() / w
    y_sub = (patch.sum(axis=(0, 2)) * dy).sum() / w
    x_sub = (patch.sum(axis=(0, 1)) * dx).sum() / w
    return float(z_sub), float(y_sub), float(x_sub)


def _soft_nms_3d_centroids(
    coords: np.ndarray,         # (N, 3) integer z, y, x
    scores: np.ndarray,         # (N,)
    sigma: float,
    score_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian Soft-NMS (Bodla et al. 2017) on already-detected peaks.

    For each peak in score-descending order, decay all lower-scoring peaks'
    scores by exp(-d^2 / (2 sigma^2)).  Peaks whose decayed score drops below
    score_threshold are discarded.  Recovers recall on dense scenes
    (clustered ribosomes, adjacent VLPs) compared to hard NMS which would
    suppress them entirely.
    """
    order = np.argsort(-scores)
    coords = coords[order].astype(np.float64)
    scores = scores[order].astype(np.float64)

    kept_coords = []
    kept_scores = []
    inv2s2 = 1.0 / (2.0 * sigma * sigma + 1e-12)

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        kept_coords.append(coords[i])
        kept_scores.append(scores[i])
        # Decay all later (lower-scoring) peaks proportional to distance.
        if i + 1 < len(scores):
            d2 = np.sum((coords[i + 1:] - coords[i]) ** 2, axis=1)
            scores[i + 1:] *= np.exp(-d2 * inv2s2)

    if not kept_coords:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return (
        np.asarray(kept_coords, dtype=np.float32),
        np.asarray(kept_scores, dtype=np.float32),
    )


def extract_centroids(
    heatmaps: np.ndarray,
    score_thresholds: Dict[str, float],
    nms_kernel: int = 7,
    nms_kernel_sizes: Optional[Dict[str, int]] = None,
    subvoxel: bool = True,
    use_soft_nms: bool = False,
    soft_nms_sigma_factor: float = 0.7,
) -> Dict[str, np.ndarray]:
    """
    Extract protein centroids from multi-class heatmaps.

    Args:
        heatmaps:        (C, D, H, W) predicted heatmaps in [0, 1]
        score_thresholds: per-class detection threshold
        nms_kernel:      fallback NMS kernel size (used when nms_kernel_sizes
                         is None and the class is not in that dict)
        nms_kernel_sizes: per-class NMS kernel dict (preferred over nms_kernel)
        subvoxel:        if True, refine peak coordinates to subvoxel precision

    Returns:
        {class_name: (N, 4) array of [z, y, x, score]}
    """
    detections = {}

    for cls_name in CLASS_NAMES:
        cls_id = PROTEIN_CLASSES[cls_name]["id"]
        hm = heatmaps[cls_id]

        # Class-specific NMS kernel (if provided) else fall back to argument
        k = (_nms_kernel_for_class(cls_name, nms_kernel_sizes)
             if nms_kernel_sizes is not None
             else nms_kernel)
        threshold = score_thresholds.get(cls_name, 0.3)

        if use_soft_nms:
            # Soft-NMS path: take all peaks > threshold/4 and Gaussian-decay.
            # Lower pre-threshold lets borderline-close peaks survive after
            # decay; the final cut is applied inside _soft_nms_3d_centroids.
            prelim_mask = hm > (threshold / 4.0)
            if prelim_mask.sum() == 0:
                detections[cls_name] = np.empty((0, 4), dtype=np.float32)
                continue
            prelim_coords = np.argwhere(prelim_mask)
            prelim_scores = hm[prelim_mask]
            cls_id = PROTEIN_CLASSES[cls_name]["id"]
            radius = CLASS_RADII_VOXELS[cls_id]
            sigma_nms = max(1.0, soft_nms_sigma_factor * radius)
            coords, scores = _soft_nms_3d_centroids(
                prelim_coords, prelim_scores,
                sigma=sigma_nms, score_threshold=threshold,
            )
            if len(coords) == 0:
                detections[cls_name] = np.empty((0, 4), dtype=np.float32)
                continue
            coords = coords.astype(np.int64)
        else:
            hm_nms = nms_3d_numpy(hm, kernel_size=k)
            mask = hm_nms > threshold
            if mask.sum() == 0:
                detections[cls_name] = np.empty((0, 4), dtype=np.float32)
                continue
            coords = np.argwhere(mask)        # (N, 3) — z, y, x
            scores = hm_nms[mask]

        if subvoxel:
            # Window scales with class sigma; capped at half the NMS kernel
            # so it cannot reach the next suppressed peak.
            sub_r = _subvoxel_radius_for_class(cls_name, max_radius=k // 2)
            refined = []
            for (z, y, x), s in zip(coords, scores):
                zs, ys, xs = _subvoxel_centroid(
                    hm, int(z), int(y), int(x), radius=sub_r,
                )
                refined.append([zs, ys, xs, float(s)])
            det = np.array(refined, dtype=np.float32)
        else:
            det = np.column_stack([coords, scores]).astype(np.float32)

        # Sort by score descending
        det = det[det[:, 3].argsort()[::-1]]
        detections[cls_name] = det

    return detections


# ---------------------------------------------------------------------------
# F-beta Metric (CZII Competition Metric)
# ---------------------------------------------------------------------------

def match_predictions(
    pred_coords: np.ndarray,     # (N, 3) z, y, x
    gt_coords: np.ndarray,       # (M, 3) z, y, x
    distance_threshold: float,
) -> Tuple[int, int, int]:
    """
    Match predicted centroids to ground truth using the Hungarian (optimal
    assignment) algorithm.

    Greedy nearest-neighbour can assign a prediction to a suboptimal GT point,
    causing a valid match elsewhere to be missed.  The Hungarian algorithm
    finds the globally optimal bijection (minimising total distance), which
    maximises true positives for a given distance threshold.

    A prediction–GT pair is a true positive only if their assigned distance
    is strictly less than distance_threshold.

    Returns: (true_positives, false_positives, false_negatives)
    """
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 0, 0, 0
    if len(pred_coords) == 0:
        return 0, 0, len(gt_coords)
    if len(gt_coords) == 0:
        return 0, len(pred_coords), 0

    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    dist_matrix = cdist(pred_coords, gt_coords)  # (N, M)

    # Hungarian algorithm on distance matrix — O(N²M) or O(NM²) via scipy
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    tp = int(np.sum(dist_matrix[row_ind, col_ind] < distance_threshold))
    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp
    return tp, fp, fn


def compute_fbeta_per_class(
    pred_centroids: Dict[str, np.ndarray],   # class → (N, 3) or (N, 4)
    gt_centroids: Dict[str, np.ndarray],     # class → (M, 3)
    beta: float = 4.0,
    distance_scale: float = 0.5,
) -> Dict[str, float]:
    """
    Compute per-class F-beta score using the CZII matching criteria.

    Distance threshold for class c = distance_scale × radius_c (in voxels).
    """
    results = {}
    for cls_name in CLASS_NAMES:
        cls_id = PROTEIN_CLASSES[cls_name]["id"]
        radius = CLASS_RADII_VOXELS[cls_id]
        dist_thresh = distance_scale * radius

        pred = pred_centroids.get(cls_name, np.empty((0, 3)))
        gt = gt_centroids.get(cls_name, np.empty((0, 3)))

        # Take only z, y, x (drop score if present)
        if pred.shape[1] > 3:
            pred = pred[:, :3]

        tp, fp, fn = match_predictions(pred, gt, dist_thresh)

        if tp == 0:
            results[cls_name] = 0.0
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fb = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        results[cls_name] = fb

    return results


def compute_weighted_fbeta(
    pred_centroids: Dict[str, np.ndarray],
    gt_centroids: Dict[str, np.ndarray],
    beta: float = 4.0,
    distance_scale: float = 0.5,
) -> float:
    """Compute the overall weighted F-beta score (competition metric)."""
    per_class = compute_fbeta_per_class(
        pred_centroids, gt_centroids, beta, distance_scale,
    )
    total_weight = sum(CLASS_WEIGHTS)
    score = sum(
        CLASS_WEIGHTS[PROTEIN_CLASSES[cls]["id"]] * per_class[cls]
        for cls in CLASS_NAMES
    ) / total_weight
    return score


# ---------------------------------------------------------------------------
# Sliding Window Aggregation
# ---------------------------------------------------------------------------

def aggregate_predictions(
    patch_preds: List[Tuple[torch.Tensor, Tuple[int, int, int]]],
    volume_shape: Tuple[int, int, int],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """
    Aggregate patch-level heatmap predictions into a full-volume heatmap.

    Uses Gaussian weighting: center of each patch contributes more than edges,
    reducing stitching artifacts.

    Args:
        patch_preds: list of (heatmap_patch, origin) where
            heatmap_patch: (C, pD, pH, pW) tensor
            origin: (z0, y0, x0) patch origin in full volume
        volume_shape: (D, H, W) full tomogram shape
        num_classes: number of output classes

    Returns:
        (C, D, H, W) aggregated heatmap
    """
    D, H, W = volume_shape
    output = np.zeros((num_classes, D, H, W), dtype=np.float64)
    weight_map = np.zeros((1, D, H, W), dtype=np.float64)

    for hm_patch, origin in patch_preds:
        if isinstance(hm_patch, torch.Tensor):
            hm_patch = hm_patch.cpu().numpy()
        z0, y0, x0 = origin
        pD, pH, pW = hm_patch.shape[1:]

        # Gaussian window weight (higher weight at center)
        window = _gaussian_window_3d(pD, pH, pW)

        output[:, z0:z0 + pD, y0:y0 + pH, x0:x0 + pW] += hm_patch * window
        weight_map[:, z0:z0 + pD, y0:y0 + pH, x0:x0 + pW] += window

    # Normalize
    weight_map = np.maximum(weight_map, 1e-8)
    output = output / weight_map
    return output.astype(np.float32)


def _gaussian_window_3d(d: int, h: int, w: int, sigma_frac: float = 0.3) -> np.ndarray:
    """Create a 3D Gaussian window for patch weighting."""
    z = np.linspace(-1, 1, d)
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    sigma = sigma_frac
    window = np.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    return window[np.newaxis, ...]  # (1, D, H, W)


# ---------------------------------------------------------------------------
# Coordinate Conversion
# ---------------------------------------------------------------------------

def voxel_to_angstrom(
    coords: np.ndarray,
    voxel_spacing: float = 10.0,
) -> np.ndarray:
    """Convert voxel coordinates to Angstrom coordinates."""
    return coords * voxel_spacing


def angstrom_to_voxel(
    coords: np.ndarray,
    voxel_spacing: float = 10.0,
) -> np.ndarray:
    """Convert Angstrom coordinates to voxel coordinates."""
    return coords / voxel_spacing


# ---------------------------------------------------------------------------
# Submission CSV Generation
# ---------------------------------------------------------------------------

def generate_submission(
    all_detections: Dict[str, Dict[str, np.ndarray]],
    output_path: str = "submission.csv",
    voxel_spacing: float = 10.0,
):
    """
    Generate a competition submission CSV.

    Args:
        all_detections: {tomo_id: {class_name: (N, 4) [z,y,x,score]}}
        output_path: path to write CSV
        voxel_spacing: for coordinate conversion
    """
    import pandas as pd

    rows = []
    for tomo_id, class_dets in all_detections.items():
        for cls_name, dets in class_dets.items():
            for det in dets:
                z, y, x, score = det
                rows.append({
                    "experiment_name": tomo_id,
                    "particle_type": cls_name,
                    "z_coord": z * voxel_spacing,
                    "y_coord": y * voxel_spacing,
                    "x_coord": x * voxel_spacing,
                    "score": score,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(df)} detections)")
