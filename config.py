"""
Configuration for SO(3)-Equivariant Protein Centroid Detection in Cryo-ET.

Architecture: e3nn-based equivariant 3D U-Net with invariant scalar output head.
Task: Detect centroids of 6 protein classes in CZII cryo-ET tomograms.
Dataset: CryoET Object Identification Challenge — dataset ID 10440

Protein classes (exact names from dataset annotation files):
    ferritin_complex, beta_amylase, beta_galactosidase,
    cytosolic_ribosome, thyroglobulin, pp7_vlp

Coordinate convention:
    Tomograms are stored as (D, H, W) = (z, y, x) in numpy/MRC/zarr.
    Annotation ndjson files store coordinates as {"x": float, "y": float, "z": float}
    in VOXEL units at the given VoxelSpacing resolution.
    Loading: coords[z, y, x] = [loc["z"], loc["y"], loc["x"]].

Protein size references:
    ferritin_complex  : EMD-41923 — apo-ferritin, ~120 Å outer diameter
    beta_amylase      : EMD-30405 — tetrameric, ~75 Å diameter
    beta_galactosidase: EMD-0153  — tetrameric, ~160 Å diameter
    cytosolic_ribosome: EMD-3883  — 80S, ~250–300 Å diameter
    thyroglobulin     : EMD-24181 — homodimer, elongated ~180 Å mean diameter
    pp7_vlp           : EMD-41917 — icosahedral VLP, ~270 Å diameter
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Protein class definitions
# ---------------------------------------------------------------------------
PROTEIN_CLASSES: Dict[str, dict] = {
    "ferritin_complex": {
        "id": 0,
        "radius_angstrom": 60.0,    # apo-ferritin outer shell, ~120 Å diameter
        "sigma_voxels": 2.5,        # Gaussian heatmap sigma at 10 Å/voxel
        "weight": 1.0,
        "color": (1.0, 0.0, 0.0),
    },
    "beta_amylase": {
        "id": 1,
        "radius_angstrom": 37.0,    # ~75 Å diameter — smallest particle
        # sigma floored at 2.5 vox: a 1.5-vox Gaussian has too little support
        # to survive 2x avg-pooling at the first encoder level.
        "sigma_voxels": 2.5,
        # CZII scoring metric assigns weight 0 to beta_amylase. Optimising it
        # spends capacity on a class that does not affect the leaderboard.
        # Kept at 0.1 (not 0) so the model still learns a representation but
        # the loss gradient barely flows through it.
        "weight": 0.1,
        "color": (0.0, 0.8, 0.0),
    },
    "beta_galactosidase": {
        "id": 2,
        "radius_angstrom": 80.0,    # tetrameric, ~160 Å diameter
        "sigma_voxels": 3.5,
        # CZII weights this class at 2.0 (was 1.5 here, mismatched).
        "weight": 2.0,
        "color": (0.0, 0.0, 1.0),
    },
    "cytosolic_ribosome": {
        "id": 3,
        "radius_angstrom": 130.0,   # 80S ribosome, ~250–300 Å diameter
        "sigma_voxels": 6.0,
        "weight": 1.0,
        "color": (1.0, 1.0, 0.0),
    },
    "thyroglobulin": {
        "id": 4,
        "radius_angstrom": 90.0,    # homodimer, elongated — mean radius
        "sigma_voxels": 4.0,
        "weight": 2.0,              # elongated shape makes Gaussian harder to fit
        "color": (0.0, 1.0, 1.0),
    },
    "pp7_vlp": {
        "id": 5,
        "radius_angstrom": 135.0,   # icosahedral VLP, ~270 Å diameter
        "sigma_voxels": 6.0,
        "weight": 1.0,
        "color": (1.0, 0.0, 1.0),
    },
}

NUM_CLASSES = len(PROTEIN_CLASSES)          # 6
CLASS_NAMES = list(PROTEIN_CLASSES.keys())
CLASS_WEIGHTS = [PROTEIN_CLASSES[c]["weight"] for c in CLASS_NAMES]
# Radius in voxels at 10 Å/voxel
CLASS_RADII_VOXELS = [PROTEIN_CLASSES[c]["radius_angstrom"] / 10.0 for c in CLASS_NAMES]
# Per-class Gaussian sigma for heatmap generation (in voxels)
CLASS_SIGMAS_VOXELS = [PROTEIN_CLASSES[c]["sigma_voxels"] for c in CLASS_NAMES]

# Maps annotation subdirectory IDs (from dataset 10440) to class names.
# Directory 106 (membrane) is a segmentation mask — not a point annotation.
ANNOTATION_DIR_TO_CLASS: Dict[str, str] = {
    "100": "ferritin_complex",
    "101": "beta_amylase",
    "102": "beta_galactosidase",
    "103": "cytosolic_ribosome",
    "104": "thyroglobulin",
    "105": "pp7_vlp",
}

# Voxel spacing of the primary reconstructions used for training
PRIMARY_VOXEL_SPACING: float = 10.012   # Å/voxel


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    # Root directory of the downloaded CryoET portal dataset
    dataset_dir: str = "./dataset/10440"
    # Subdirectory name under each run's Reconstructions folder
    voxel_spacing_dir: str = "VoxelSpacing10.012"

    # Tomogram subdirectories to stack as input channels.
    # The CZII dataset provides multiple reconstructions per run (denoised,
    # wbp, ctfdeconvolved, isonet, denoiset). Stacking them as input channels
    # is what every top CZII Kaggle solution did — it gives the network
    # complementary views of the same density at no extra training cost.
    # The default "100" is the denoised reconstruction. Set to a tuple to
    # stack multiple, e.g. tomo_dirs=("100", "wbp", "ctfdeconvolved").
    # Missing types are skipped gracefully so the same config works whether
    # all variants exist on disk or only the denoised one.
    tomo_dirs: Tuple[str, ...] = ("100",)

    # Tomogram properties
    voxel_spacing: float = 10.012       # Å/voxel

    # Patch extraction.
    # patch_size is (D, H, W) = (z, y, x). The default (96, 128, 128) matches
    # the tomogram aspect ratio (~184 x 630 x 630) better than cubic 96^3 —
    # each patch captures more xy area where most particles are.
    patch_size: Tuple[int, int, int] = (96, 128, 128)
    patches_per_tomo: int = 64          # patches sampled per tomogram per epoch
    positive_patch_ratio: float = 0.7   # fraction of patches centered near a particle
    # Round-robin per-class positive sampling. When True, the i-th positive
    # patch is centred on a random instance of class i % NUM_CLASSES, which
    # guarantees uniform per-class coverage instead of over-sampling the
    # most populous class (ribosomes).
    balanced_class_sampling: bool = True

    # Gaussian heatmap targets — per-class sigmas are in CLASS_SIGMAS_VOXELS;
    # gaussian_sigma is used only as a per-class fallback if not overridden.
    gaussian_sigma: float = 4.0         # voxels (fallback)
    gaussian_truncate: float = 3.0      # truncate Gaussian at this many sigmas
    # Anisotropic z-axis sigma scaling to account for missing-wedge PSF elongation.
    # sigma_z = sigma_xy * z_sigma_scale. Set to 1.0 for isotropic (so3 variant).
    # Typical missing-wedge elongation factor: 1.5–2.0 at ±60° tilt range.
    z_sigma_scale: float = 1.5

    # Target mode controls what the dataloader emits:
    #   "blur"     : standard, fully-blurred Gaussian heatmaps (default).
    #   "xy_blur"  : xy-blurred only; the z-direction Gaussian is applied
    #                in the loss with a learnable per-class sigma_z.  Used
    #                with TrainConfig.learnable_z_sigma=True so that the
    #                anisotropy is inferred from data rather than hard-coded.
    target_mode: str = "blur"

    # Normalization
    clip_percentile: Tuple[float, float] = (0.5, 99.5)
    normalize: bool = True

    # Augmentation
    random_flip: bool = True
    random_noise_std: float = 0.05
    random_contrast: Tuple[float, float] = (0.9, 1.1)

    # Heavy MONAI-style augmentation pipeline (nnU-Net conventions:
    # gamma, brightness, blur, low-resolution simulation, full-angle z-rotation).
    # When True, the simple flip/contrast/noise pipeline above is replaced
    # by the richer one in dataset._augment_patch_heavy.
    # Required for the SegResNet baseline to learn approximate equivariance
    # from data instead of from architectural constraints.
    heavy_augmentation: bool = False

    # Run names reserved for validation (excluded from training)
    val_run_names: Tuple[str, ...] = ("TS_99_9",)


@dataclass
class ModelConfig:
    """SO(3)-equivariant U-Net model configuration."""
    in_channels: int = 1                 # single density channel
    num_classes: int = NUM_CLASSES       # 6 protein types

    # Irrep specifications for each level.
    # Format: "Ns x 0e + Nv x 1o + Nt x 2e"
    # 0e = even scalar, 1o = odd vector, 2e = even rank-2 tensor.
    encoder_irreps: List[str] = field(default_factory=lambda: [
        "32x0e + 16x1o + 8x2e",    # level 0 — full resolution
        "48x0e + 24x1o + 12x2e",   # level 1
        "64x0e + 32x1o + 16x2e",   # level 2
        "96x0e + 48x1o + 24x2e",   # level 3 — bottleneck
    ])
    decoder_irreps: List[str] = field(default_factory=lambda: [
        "64x0e + 32x1o + 16x2e",   # level 2
        "48x0e + 24x1o + 12x2e",   # level 1
        "32x0e + 16x1o + 8x2e",    # level 0
    ])

    kernel_size: int = 5                 # 5×5×5 equivariant kernels
    lmax: int = 2                        # max spherical harmonics degree
    num_radial_basis: int = 8            # Gaussian radial basis count

    activation: str = "norm"           # "gated" or "norm"
    use_batch_norm: bool = True

    pool_type: str = "avg"              # "avg" only (max-pool breaks equivariance)

    # Architecture variant — controls which symmetry group is enforced:
    #   "so3"       : full SO(3) equivariance, isotropic targets (default)
    #   "so3_aniso" : full SO(3) equivariance + anisotropic z targets
    #   "so2"       : SO(2) equivariance (z-axis only) + anisotropic z targets
    #   "so2_iso"   : SO(2) equivariance + isotropic targets
    # The "_aniso" / "_iso" suffix sets z_sigma_scale in train.py.
    # The four variants form a 2x2 ablation: architecture (SO3/SO2) x
    # targets (iso/aniso), letting you separate the two factors.
    model_variant: str = "so3"

    # When use_equivariant=False, which non-equivariant baseline to build.
    # "segresnet" (default) is the strong MONAI 3D ResNet-style baseline that
    # CZII Kaggle top-5 solutions used; "unet3d" is the plain U-Net.
    # The SegResNet baseline + heavy augmentation is the integrity check that
    # decides whether the equivariant architecture is worth keeping.
    baseline_arch: str = "segresnet"


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 200
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"        # "cosine" or "step"
    warmup_epochs: int = 5

    loss_type: str = "combined"         # "wmse", "focal", "combined"
    focal_alpha: float = 2.0
    focal_gamma: float = 4.0
    wmse_alpha: float = 0.1             # background weight in weighted MSE
    loss_class_weights: bool = True

    # If True, the loss applies a differentiable z-direction Gaussian blur
    # to xy-only-blurred targets with a per-class learnable sigma_z.
    # Requires DataConfig.target_mode == "xy_blur".
    # Lets the network learn the missing-wedge anisotropy per protein class
    # instead of using a single hard-coded z_sigma_scale.
    learnable_z_sigma: bool = False
    learnable_z_sigma_init: float = 1.5     # initial sigma_z / sigma_xy ratio
    learnable_z_sigma_min: float = 0.5      # clamp lower bound (per axis)
    learnable_z_sigma_max: float = 4.0      # clamp upper bound

    use_amp: bool = True

    # Exponential Moving Average of model weights — a near-free regulariser
    # used by the 4th and 13th-place CZII solutions. Shadow weights updated
    # as theta_ema = decay * theta_ema + (1 - decay) * theta, evaluated and
    # checkpointed independently of the live model.
    use_ema: bool = True
    ema_decay: float = 0.999

    # Deep supervision: attach auxiliary heads at decoder levels and sum
    # losses with decreasing weights. nnU-Net default is [0.5, 0.25, 0.125]
    # for the three coarser scales; the primary head keeps weight 1.0.
    deep_supervision: bool = False
    deep_supervision_weights: Tuple[float, ...] = (0.5, 0.25, 0.125)

    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10
    log_dir: str = "./logs"

    val_every: int = 5
    val_patch_overlap: float = 0.25

    # Full-volume validation: runs sliding-window inference on each val run
    # and computes F-beta from the merged centroids — matches real inference
    # behaviour (no positive-patch bias).  Expensive (~1-2 min per run on H100),
    # so by default it only runs every full_volume_val_every epochs.
    # Set to 0 to disable.
    full_volume_val_every: int = 20
    full_volume_val_batch_size: int = 4
    full_volume_val_overlap: float = 0.5

    seed: int = 42
    num_workers: int = 4


@dataclass
class InferenceConfig:
    """Inference and post-processing configuration."""
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    overlap: float = 0.5

    # Per-class NMS kernel sizes (must be odd).
    # Kernel ≈ 2 × floor(radius_voxels) + 1 so that two particles whose
    # centroids are closer than one diameter are suppressed to one detection.
    # Computed at 10 Å/voxel from CLASS_RADII_VOXELS:
    #   ferritin_complex    : r=6  → 13
    #   beta_amylase        : r=3.7→  9
    #   beta_galactosidase  : r=8  → 17
    #   cytosolic_ribosome  : r=13 → 27
    #   thyroglobulin       : r=9  → 19
    #   pp7_vlp             : r=13.5→ 29
    nms_kernel_sizes: Dict[str, int] = field(default_factory=lambda: {
        "ferritin_complex":    13,
        "beta_amylase":         9,
        "beta_galactosidase":  17,
        "cytosolic_ribosome":  27,
        "thyroglobulin":       19,
        "pp7_vlp":             29,
    })
    # Legacy single kernel — kept for callers that pass a flat int.
    nms_kernel_size: int = 7

    score_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "ferritin_complex": 0.3,
        "beta_amylase": 0.2,
        "beta_galactosidase": 0.3,
        "cytosolic_ribosome": 0.3,
        "thyroglobulin": 0.2,
        "pp7_vlp": 0.3,
    })

    fbeta_beta: float = 4.0
    distance_threshold_scale: float = 0.5  # match if dist < scale × radius_voxels

    # Test-time augmentation: average predictions over 4 in-plane rotations
    # (k * 90 degrees about z). Free F-beta lift for non-equivariant models;
    # near-zero gain for SO(3) models (which is itself a useful sanity check).
    use_tta: bool = False

    # Soft-NMS: instead of zeroing all peaks within the NMS kernel, decay
    # their scores by exp(-d^2 / (2 * sigma_nms^2)). Recovers recall on dense
    # scenes (clustered ribosomes, adjacent VLPs). sigma_nms = factor * radius.
    use_soft_nms: bool = False
    soft_nms_sigma_factor: float = 0.7

    output_dir: str = "./predictions"
    submission_csv: str = "./submission.csv"


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
