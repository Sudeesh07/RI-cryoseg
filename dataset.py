"""
CZII CryoET Dataset Loader.

Loads tomograms and point annotations from the CryoET Data Portal directory
structure (dataset 10440), generates per-class Gaussian heatmap targets.

Directory structure expected under dataset_dir (e.g. ./dataset/10440):
    {run_name}/
        Reconstructions/VoxelSpacing10.012/
            Tomograms/100/TS_{run_name}.mrc       (denoised, default)
            Tomograms/100/TS_{run_name}.zarr       (OME-Zarr, same data)
            Annotations/100/*_point.ndjson         (ferritin_complex)
            Annotations/101/*_point.ndjson         (beta_amylase)
            Annotations/102/*_point.ndjson         (beta_galactosidase)
            Annotations/103/*_point.ndjson         (cytosolic_ribosome)
            Annotations/104/*_point.ndjson         (thyroglobulin)
            Annotations/105/*_point.ndjson         (pp7_vlp)
            Annotations/106/...segmentationmask.*  (membrane — skip)

Coordinate convention:
    ndjson files store {"x": vox, "y": vox, "z": vox} in voxel units.
    Internally stored as np.array (N, 3) in [z, y, x] order to match
    numpy/MRC (D, H, W) indexing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

from config import (
    Config, DataConfig,
    PROTEIN_CLASSES, CLASS_NAMES, NUM_CLASSES,
    CLASS_RADII_VOXELS, CLASS_SIGMAS_VOXELS,
    ANNOTATION_DIR_TO_CLASS,
)


# ---------------------------------------------------------------------------
# Tomogram I/O
# ---------------------------------------------------------------------------

def load_tomogram(path: Union[str, Path]) -> np.ndarray:
    """Load a 3D tomogram. Supports .mrc/.rec, .zarr directories, .npy."""
    path = Path(path)

    if path.suffix in (".mrc", ".rec"):
        import mrcfile
        with mrcfile.open(str(path), permissive=True) as mrc:
            return mrc.data.copy().astype(np.float32)

    if path.suffix == ".zarr" or (path.is_dir() and (path / ".zattrs").exists()):
        import zarr
        store = zarr.open(str(path), mode="r")
        # OME-Zarr: data at resolution level "0"
        arr = store["0"] if "0" in store else store
        volume = np.array(arr, dtype=np.float32)
        # Squeeze T / C dimensions if present
        while volume.ndim > 3:
            volume = volume[0]
        return volume

    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)

    raise ValueError(f"Unsupported tomogram format: {path}")


def normalize_tomogram(
    volume: np.ndarray,
    clip_pct: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """Clip outliers, then z-score normalise to zero mean, unit variance."""
    lo, hi = np.percentile(volume, clip_pct)
    volume = np.clip(volume, lo, hi)
    mean, std = volume.mean(), volume.std()
    if std > 0:
        volume = (volume - mean) / std
    return volume


# ---------------------------------------------------------------------------
# Annotation loading — CryoET Data Portal ndjson format
# ---------------------------------------------------------------------------

def load_ndjson_points(ndjson_path: Union[str, Path]) -> np.ndarray:
    """
    Parse a CryoET portal point annotation ndjson file.

    Each line is:  {"type": "point", "location": {"x": vox, "y": vox, "z": vox}}

    Coordinates are voxel indices for the reconstruction at the given
    VoxelSpacing.  Returns (N, 3) array in [z, y, x] order.
    """
    coords = []
    with open(ndjson_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            loc = record.get("location", {})
            # Store as [z, y, x] to match numpy (D, H, W) array indexing
            coords.append([loc["z"], loc["y"], loc["x"]])
    return np.array(coords, dtype=np.float32) if coords else np.empty((0, 3), dtype=np.float32)


def load_run_annotations(
    run_dir: Union[str, Path],
    voxel_spacing_dir: str = "VoxelSpacing10.012",
) -> Dict[str, np.ndarray]:
    """
    Load all point annotations for one run.

    Scans Annotations sub-directories, skips segmentation masks.

    Returns:
        {class_name: (N, 3) array of [z, y, x] voxel coords}
    """
    ann_root = Path(run_dir) / "Reconstructions" / voxel_spacing_dir / "Annotations"
    annotations: Dict[str, np.ndarray] = {}

    if not ann_root.exists():
        return annotations

    for ann_dir in sorted(ann_root.iterdir()):
        if not ann_dir.is_dir():
            continue

        dir_id = ann_dir.name  # e.g. "100", "101", ...
        # Use directory-to-class mapping; skip unmapped dirs (e.g. membrane 106)
        cls_name = ANNOTATION_DIR_TO_CLASS.get(dir_id)
        if cls_name is None:
            continue

        # Find the point ndjson file
        ndjson_files = list(ann_dir.glob("*_point.ndjson"))
        if not ndjson_files:
            continue
        # Take the first (should be exactly one per class)
        coords = load_ndjson_points(ndjson_files[0])
        annotations[cls_name] = coords

    return annotations


def load_dataset_annotations(
    dataset_dir: Union[str, Path],
    voxel_spacing_dir: str = "VoxelSpacing10.012",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load annotations for all runs in a dataset directory.

    Returns:
        {run_name: {class_name: (N, 3) array of [z, y, x] coords}}
    """
    dataset_dir = Path(dataset_dir)
    all_annotations: Dict[str, Dict[str, np.ndarray]] = {}

    for run_dir in sorted(dataset_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if not run_name.startswith("TS_"):
            continue
        all_annotations[run_name] = load_run_annotations(run_dir, voxel_spacing_dir)

    return all_annotations


def get_tomogram_path(
    dataset_dir: Union[str, Path],
    run_name: str,
    voxel_spacing_dir: str = "VoxelSpacing10.012",
    tomo_dir: str = "100",         # "100" = denoised default
    prefer_zarr: bool = False,
) -> Path:
    """
    Return the path to a tomogram MRC (or zarr) file for a given run.

    The primary (is_visualization_default=True) denoised reconstruction is
    under Tomograms/100/.
    """
    tomo_root = (
        Path(dataset_dir) / run_name
        / "Reconstructions" / voxel_spacing_dir
        / "Tomograms" / tomo_dir
    )
    # Prefer zarr if requested and available, otherwise MRC
    zarr_path = tomo_root / f"{run_name}.zarr"
    mrc_path = tomo_root / f"{run_name}.mrc"

    if prefer_zarr and zarr_path.exists():
        return zarr_path
    if mrc_path.exists():
        return mrc_path
    if zarr_path.exists():
        return zarr_path
    raise FileNotFoundError(f"No tomogram found for run {run_name} in {tomo_root}")


def discover_run_names(dataset_dir: Union[str, Path]) -> List[str]:
    """Return sorted list of run names (TS_* directories) in the dataset."""
    return sorted(
        d.name
        for d in Path(dataset_dir).iterdir()
        if d.is_dir() and d.name.startswith("TS_")
    )


# ---------------------------------------------------------------------------
# Gaussian Heatmap Generation
# ---------------------------------------------------------------------------

def _gaussian_peak_value(sigma: float, truncate: float = 3.0) -> float:
    """
    Compute the peak value produced by gaussian_filter applied to a unit delta.

    Used to normalise the heatmap so that an isolated particle centroid
    achieves a target value of 1.0 regardless of sigma.
    """
    # Create a small volume, place a delta at centre, apply filter, read peak
    size = max(int(2 * truncate * sigma) * 2 + 1, 7)
    vol = np.zeros((size, size, size), dtype=np.float64)
    c = size // 2
    vol[c, c, c] = 1.0
    filtered = gaussian_filter(vol, sigma=sigma, truncate=truncate)
    return float(filtered[c, c, c])


# Pre-compute normalisation factors for each (sigma_z, sigma_xy) pair.
# Key: (sigma_z_rounded, sigma_xy_rounded). Isotropic uses the same value for both.
_SIGMA_NORM_CACHE: Dict[tuple, float] = {}


def _get_norm_factor(
    sigma: float,
    sigma_z: Optional[float] = None,
    truncate: float = 3.0,
) -> float:
    """
    Compute the peak value of gaussian_filter applied to a unit delta at origin,
    using sigma_z along axis-0 (D/z) and sigma along axes 1,2 (H/y, W/x).

    If sigma_z is None, isotropic sigma is used for all axes.
    """
    sz = sigma_z if sigma_z is not None else sigma
    key = (round(sz, 4), round(sigma, 4))
    if key not in _SIGMA_NORM_CACHE:
        max_s = max(sz, sigma)
        size = max(int(2 * truncate * max_s) * 2 + 1, 7)
        vol = np.zeros((size, size, size), dtype=np.float64)
        c = size // 2
        vol[c, c, c] = 1.0
        filtered = gaussian_filter(vol, sigma=[sz, sigma, sigma], truncate=truncate)
        _SIGMA_NORM_CACHE[key] = float(filtered[c, c, c])
    return _SIGMA_NORM_CACHE[key]


def generate_gaussian_heatmap(
    volume_shape: Tuple[int, int, int],
    centroids: np.ndarray,          # (N, 3) in [z, y, x] voxel coords
    sigma: float = 4.0,
    sigma_z: Optional[float] = None,
    truncate: float = 3.0,
    xy_only: bool = False,
) -> np.ndarray:
    """
    Generate a 3D Gaussian heatmap with peaks at the given centroids.

    Places a unit impulse at each centroid, convolves with an (optionally
    anisotropic) Gaussian kernel, then normalises so that an isolated centroid
    peak = 1.0.  Overlapping Gaussians are clipped to 1.0 (CenterNet convention).

    Args:
        sigma:   in-plane (H, W) Gaussian sigma in voxels.
        sigma_z: axial (D) Gaussian sigma.  Pass a value > sigma to model the
                 missing-wedge PSF elongation along z.  None = isotropic.
        xy_only: if True, only blur along (y, x) — leave z as impulses.
                 The z-direction Gaussian is then applied in the loss with a
                 learnable per-class sigma_z (see TrainConfig.learnable_z_sigma).

    Returns (D, H, W) float32 array in [0, 1].
    """
    heatmap = np.zeros(volume_shape, dtype=np.float64)
    if len(centroids) == 0:
        return heatmap.astype(np.float32)

    D, H, W = volume_shape
    for c in centroids:
        z, y, x = int(round(float(c[0]))), int(round(float(c[1]))), int(round(float(c[2])))
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            heatmap[z, y, x] += 1.0   # additive for overlapping particles

    if xy_only:
        # Blur only along (y, x); leave z untouched so the loss can apply a
        # learnable z-direction Gaussian. sigma=0 on axis 0 is a no-op.
        heatmap = gaussian_filter(
            heatmap, sigma=[0.0, sigma, sigma], truncate=truncate,
        )
        # Normalise so an isolated centroid peaks at 1.0 along the xy plane.
        # The peak of an xy-only blur of a unit impulse equals the 2D peak.
        size = max(int(2 * truncate * sigma) * 2 + 1, 7)
        impulse2d = np.zeros((size, size), dtype=np.float64)
        impulse2d[size // 2, size // 2] = 1.0
        peak2d = float(gaussian_filter(impulse2d, sigma=sigma, truncate=truncate)[size // 2, size // 2])
        if peak2d > 0:
            heatmap /= peak2d
    else:
        # Apply Gaussian filter — anisotropic if sigma_z provided
        sz = sigma_z if sigma_z is not None else sigma
        heatmap = gaussian_filter(heatmap, sigma=[sz, sigma, sigma], truncate=truncate)

        # Normalise: divide by the peak value of a single isolated Gaussian
        norm = _get_norm_factor(sigma, sigma_z=sigma_z, truncate=truncate)
        if norm > 0:
            heatmap /= norm

    # Clip overlapping particle regions at 1.0
    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap.astype(np.float32)


def generate_multiclass_heatmap(
    volume_shape: Tuple[int, int, int],
    annotations: Dict[str, np.ndarray],      # class_name → (N, 3) [z,y,x]
    sigma_per_class: Optional[List[float]] = None,
    z_sigma_scale: float = 1.0,
    truncate: float = 3.0,
    xy_only: bool = False,
) -> np.ndarray:
    """
    Generate a (C, D, H, W) multi-channel Gaussian heatmap.

    Each channel corresponds to one protein class in CLASS_NAMES order.
    sigma_per_class defaults to CLASS_SIGMAS_VOXELS if not provided.

    z_sigma_scale: multiply each class sigma by this factor for the z-axis.
        1.0 = isotropic targets (so3 variant).
        1.5 = z-elongated targets (so3_aniso / so2 variants) to match the
              anisotropic PSF caused by the missing wedge in cryo-ET.

    xy_only: if True, blur only along (y, x).  z-blur is then applied in the
        loss with a learnable per-class sigma_z.  Ignores z_sigma_scale.
    """
    sigmas = sigma_per_class if sigma_per_class is not None else CLASS_SIGMAS_VOXELS
    heatmaps = np.zeros((NUM_CLASSES, *volume_shape), dtype=np.float32)

    for cls_name, centroids in annotations.items():
        if cls_name not in PROTEIN_CLASSES:
            continue
        cls_id = PROTEIN_CLASSES[cls_name]["id"]
        sigma = sigmas[cls_id]
        sigma_z = sigma * z_sigma_scale if z_sigma_scale != 1.0 else None
        heatmaps[cls_id] = generate_gaussian_heatmap(
            volume_shape, centroids,
            sigma=sigma, sigma_z=sigma_z, truncate=truncate,
            xy_only=xy_only,
        )
    return heatmaps


# ---------------------------------------------------------------------------
# Patch Extraction
# ---------------------------------------------------------------------------

def _sample_patch_origin(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    centroids_all: Optional[np.ndarray] = None,
    positive_ratio: float = 0.7,
) -> Tuple[int, int, int]:
    """
    Sample the (z0, y0, x0) top-left corner for a patch.

    With probability `positive_ratio`, centres near a random annotated
    centroid (±patch/4 jitter).  Otherwise draws uniformly at random.
    """
    D, H, W = volume_shape
    pd, ph, pw = patch_size

    use_positive = (
        centroids_all is not None
        and len(centroids_all) > 0
        and np.random.rand() < positive_ratio
    )

    if use_positive:
        i = np.random.randint(len(centroids_all))
        cz, cy, cx = centroids_all[i]
        jitter = np.random.randint(-pd // 4, pd // 4 + 1, size=3)
        z0 = int(cz) - pd // 2 + jitter[0]
        y0 = int(cy) - ph // 2 + jitter[1]
        x0 = int(cx) - pw // 2 + jitter[2]
    else:
        z0 = np.random.randint(0, max(D - pd, 1))
        y0 = np.random.randint(0, max(H - ph, 1))
        x0 = np.random.randint(0, max(W - pw, 1))

    z0 = int(np.clip(z0, 0, max(D - pd, 0)))
    y0 = int(np.clip(y0, 0, max(H - ph, 0)))
    x0 = int(np.clip(x0, 0, max(W - pw, 0)))
    return z0, y0, x0


def extract_random_patch(
    volume: np.ndarray,
    heatmap: np.ndarray,
    patch_size: Tuple[int, int, int],
    centroids_all: Optional[np.ndarray] = None,
    positive_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a random patch from volume and its pre-built heatmap."""
    pd, ph, pw = patch_size
    z0, y0, x0 = _sample_patch_origin(
        volume.shape, patch_size, centroids_all, positive_ratio,
    )
    vol_patch = volume[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    hm_patch  = heatmap[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return vol_patch, hm_patch


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_patch(
    volume: np.ndarray,
    heatmap: np.ndarray,
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations to a volume patch and its heatmap."""
    # Random axis flips — only flip in-plane axes (1=y, 2=x).
    # Flipping axis 0 (z / beam axis) is physically invalid for cryo-ET: it
    # would reverse the beam direction and change the orientation of the missing
    # wedge, producing a distribution the model will never see during inference.
    if cfg.random_flip:
        for axis in [1, 2]:
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis).copy()
                heatmap = np.flip(heatmap, axis=axis + 1).copy()  # +1 for channel dim

    # Random 90° rotations in the XY plane (axis 1,2 of volume = y,x)
    k = np.random.randint(4)
    if k > 0:
        volume = np.rot90(volume, k=k, axes=(1, 2)).copy()
        heatmap = np.rot90(heatmap, k=k, axes=(2, 3)).copy()

    # Additive Gaussian noise
    if cfg.random_noise_std > 0:
        volume = volume + np.random.normal(0, cfg.random_noise_std, volume.shape).astype(np.float32)

    # Random contrast scaling
    lo, hi = cfg.random_contrast
    volume = volume * np.random.uniform(lo, hi)

    return volume, heatmap


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class CZIIDataset(Dataset):
    """
    Patch-based dataset for CZII cryo-ET protein centroid detection.

    Memory layout
    -------------
    Pre-loads all tomograms (~280 MB each × 7 runs ≈ 2 GB total).
    Heatmaps are generated ON THE FLY per patch, not stored globally.

    Storing full-volume (C, D, H, W) heatmaps would require
    6 classes × 184 × 630 × 630 × 4 B ≈ 1.75 GB per run (12 GB total),
    which exceeds typical laptop RAM.  Per-patch generation costs
    ~3 ms and uses negligible memory.

    To avoid edge-particle miss, annotations within
    (gaussian_truncate × max_sigma) voxels outside the patch boundary
    are also included — generate_gaussian_heatmap clips them correctly.
    """

    def __init__(
        self,
        run_names: List[str],
        dataset_dir: Union[str, Path],
        cfg: DataConfig,
        is_train: bool = True,
    ):
        self.cfg = cfg
        self.is_train = is_train

        self.volumes: Dict[str, np.ndarray] = {}
        self.annotations: Dict[str, Dict[str, np.ndarray]] = {}   # raw coords
        self.all_centroids: Dict[str, np.ndarray] = {}

        all_annotations = load_dataset_annotations(dataset_dir, cfg.voxel_spacing_dir)

        for run_name in run_names:
            tomo_path = get_tomogram_path(dataset_dir, run_name, cfg.voxel_spacing_dir)
            vol = load_tomogram(tomo_path)
            if cfg.normalize:
                vol = normalize_tomogram(vol, cfg.clip_percentile)
            self.volumes[run_name] = vol

            annot = all_annotations.get(run_name, {})
            self.annotations[run_name] = annot

            coords_list = [c for c in annot.values() if len(c) > 0]
            self.all_centroids[run_name] = (
                np.concatenate(coords_list, axis=0) if coords_list
                else np.empty((0, 3), dtype=np.float32)
            )

        self.run_names = list(self.volumes.keys())
        self._len = cfg.patches_per_tomo * len(self.run_names)

        # Border around patch to capture Gaussian tails of edge particles
        from config import CLASS_SIGMAS_VOXELS
        self._border = int(max(CLASS_SIGMAS_VOXELS) * cfg.gaussian_truncate) + 1

    def __len__(self) -> int:
        return self._len

    def _make_patch_heatmap(
        self,
        annotations: Dict[str, np.ndarray],
        origin: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Generate a (C, pd, ph, pw) heatmap for one patch on the fly.

        Translates global annotation coordinates to patch-local frame.
        Centroids outside [0, patch_size) are still included because
        generate_gaussian_heatmap skips only out-of-bounds centres, while
        particles just outside the edge still contribute Gaussian tails
        inside the patch (handled by scipy gaussian_filter with truncation).
        """
        z0, y0, x0 = origin
        border = self._border
        pd, ph, pw = patch_size
        # Expanded patch size for Gaussian bleeding at edges
        expanded = (pd + 2 * border, ph + 2 * border, pw + 2 * border)

        # Translate annotations to expanded-patch-local frame
        local_annot: Dict[str, np.ndarray] = {}
        for cls_name, coords in annotations.items():
            if len(coords) == 0:
                continue
            local = coords - np.array([z0 - border, y0 - border, x0 - border],
                                      dtype=np.float32)
            local_annot[cls_name] = local

        # Generate heatmap for expanded region then crop to patch.
        # target_mode == "xy_blur" emits xy-only blurred targets so the loss
        # can apply a learnable z-direction Gaussian (see losses.LearnableZBlurLoss).
        xy_only = getattr(self.cfg, "target_mode", "blur") == "xy_blur"
        hm_expanded = generate_multiclass_heatmap(
            expanded, local_annot,
            z_sigma_scale=self.cfg.z_sigma_scale,
            truncate=self.cfg.gaussian_truncate,
            xy_only=xy_only,
        )
        return hm_expanded[:, border:border + pd, border:border + ph, border:border + pw]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        run_name = self.run_names[idx % len(self.run_names)]
        vol      = self.volumes[run_name]
        annot    = self.annotations[run_name]
        centroids = self.all_centroids[run_name]

        patch_size = self.cfg.patch_size
        z0, y0, x0 = _sample_patch_origin(
            vol.shape, patch_size,
            centroids_all=centroids if self.is_train else None,
            positive_ratio=self.cfg.positive_patch_ratio if self.is_train else 0.0,
        )
        pd, ph, pw = patch_size
        vol_patch = vol[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        hm_patch  = self._make_patch_heatmap(annot, (z0, y0, x0), patch_size)

        if self.is_train:
            vol_patch, hm_patch = augment_patch(vol_patch, hm_patch, self.cfg)

        vol_tensor = torch.from_numpy(np.ascontiguousarray(vol_patch)).unsqueeze(0).float()
        hm_tensor  = torch.from_numpy(np.ascontiguousarray(hm_patch)).float()
        return vol_tensor, hm_tensor


# ---------------------------------------------------------------------------
# Inference Dataset (sliding window)
# ---------------------------------------------------------------------------

def _sliding_origins(size: int, patch: int, stride: int) -> List[int]:
    """
    Compute patch origins along one dimension such that the whole extent
    [0, size) is covered.  The last origin is always max(size - patch, 0)
    to ensure the final patch reaches the end of the volume.
    """
    origins = list(range(0, max(size - patch, 0) + 1, stride))
    # Guarantee the far edge is covered
    last = max(size - patch, 0)
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


class CZIIInferenceDataset(Dataset):
    """
    Sliding-window dataset for full-tomogram inference.

    Yields patches and their (z0, y0, x0) origins for later aggregation.
    """

    def __init__(
        self,
        volume: np.ndarray,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
    ):
        self.volume = volume
        self.patch_size = patch_size
        D, H, W = volume.shape
        pd, ph, pw = patch_size
        sd = max(int(pd * (1 - overlap)), 1)
        sh = max(int(ph * (1 - overlap)), 1)
        sw = max(int(pw * (1 - overlap)), 1)

        # Build grid of patch origins with guaranteed edge coverage
        origins = []
        for z0 in _sliding_origins(D, pd, sd):
            for y0 in _sliding_origins(H, ph, sh):
                for x0 in _sliding_origins(W, pw, sw):
                    origins.append((z0, y0, x0))

        # Deduplicate while preserving order
        seen = set()
        self.origins = []
        for o in origins:
            if o not in seen:
                seen.add(o)
                self.origins.append(o)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z0, y0, x0 = self.origins[idx]
        pd, ph, pw = self.patch_size
        patch = self.volume[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        patch_tensor = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float()
        origin = torch.tensor([z0, y0, x0], dtype=torch.long)
        return patch_tensor, origin


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: Config,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and validation DataLoaders from the CZII dataset directory.

    Splits runs into train / val according to cfg.data.val_run_names.

    Args:
        rank:       DDP process rank (0 for single-GPU).
        world_size: total number of DDP processes (1 for single-GPU).

    When world_size > 1 a DistributedSampler replaces shuffle=True so that
    each GPU processes a non-overlapping subset of the data each epoch.
    Access the sampler via ``train_loader.sampler`` and call
    ``sampler.set_epoch(epoch)`` at the start of each training epoch.
    """
    from torch.utils.data.distributed import DistributedSampler

    data_cfg = cfg.data
    train_cfg = cfg.train

    all_runs = discover_run_names(data_cfg.dataset_dir)
    if not all_runs:
        raise FileNotFoundError(f"No TS_* run directories found in {data_cfg.dataset_dir}")

    val_set = set(data_cfg.val_run_names)
    train_runs = [r for r in all_runs if r not in val_set]
    val_runs = [r for r in all_runs if r in val_set]

    if not train_runs:
        raise ValueError(
            f"No training runs found. all_runs={all_runs}, val_run_names={data_cfg.val_run_names}"
        )

    train_ds = CZIIDataset(train_runs, data_cfg.dataset_dir, data_cfg, is_train=True)

    if world_size > 1:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    # On Windows, DataLoader worker processes use the 'spawn' start method,
    # which pickles the entire Dataset (including all pre-loaded tomogram
    # arrays) and sends it over a pipe.  Large numpy arrays cause
    # OSError [Errno 22] / UnpicklingError: pickle data was truncated.
    # Force num_workers=0 on Windows so loading runs in the main process.
    import platform
    safe_num_workers = 0 if platform.system() == "Windows" else train_cfg.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=safe_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = None
    if val_runs:
        val_ds = CZIIDataset(val_runs, data_cfg.dataset_dir, data_cfg, is_train=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=safe_num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader
