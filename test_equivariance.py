"""
SO(3) Equivariance and Invariance Tests + Visualizations.

Verifies, at multiple levels of the network, that:
  1. Output heatmaps are rotation-INVARIANT: model(R·X) = R·model(X)
  2. Intermediate features are rotation-EQUIVARIANT: f(R·X)(r) = D(R)·f(X)(R⁻¹·r)
     where D(R) is the Wigner D-matrix for the feature's irrep type.

Two test families:
  A. Exact tests on 90° grid rotations — machine-precision errors expected (~1e-5).
  B. Approximate tests on arbitrary angles via scipy interpolation — bounded by
     the grid discretisation error (~0.01–0.05 depending on angle).

Visualisations produced (saved as PNGs):
  1. equivariance_error_vs_angle.png  — MAE vs rotation angle, equivariant vs baseline
  2. heatmap_comparison.png           — original vs rotated predictions, side by side
  3. wigner_d_structure.png           — block structure of the D-matrix
  4. feature_decomposition.png        — scalar/vector/tensor channel magnitudes

Usage:
    python test_equivariance.py            # run tests + generate all plots
    python test_equivariance.py --tests-only  # skip visualisations
    pytest test_equivariance.py -v         # run as pytest suite
"""

import sys
import argparse
import math
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn

from e3nn import o3
from e3nn.o3 import Irreps

from config import ModelConfig, NUM_CLASSES
from model import EquivariantUNet3D, SO2EquivariantUNet3D, UNet3D


# ============================================================================
# Part 0 — Tiny model for fast testing
# ============================================================================

def build_test_model(
    equivariant: bool = True,
    so2: bool = False,
    lmax: int = 2,
) -> Tuple[nn.Module, ModelConfig]:
    """
    Build a minimal model (fewer channels, smaller kernel) for fast tests.
    BN disabled: running stats are uninitialized at eval time on tiny volumes.
    """
    cfg = ModelConfig(
        in_channels=1,
        num_classes=2,
        encoder_irreps=[
            "8x0e + 4x1o + 2x2e",   # level 0
            "8x0e + 4x1o + 2x2e",   # level 1
            "8x0e + 4x1o + 2x2e",   # level 2
            "8x0e + 4x1o + 2x2e",   # level 3 (bottleneck)
        ],
        decoder_irreps=[
            "8x0e + 4x1o + 2x2e",   # level 2
            "8x0e + 4x1o + 2x2e",   # level 1
            "8x0e + 4x1o + 2x2e",   # level 0
        ],
        kernel_size=3,
        lmax=lmax,
        num_radial_basis=4,
        use_batch_norm=False,   # MUST be False: running stats invalid on tiny test volumes
        activation="norm",      # norm-nonlinearity is simpler to verify
    )
    if not equivariant:
        model = UNet3D(in_channels=1, num_classes=2, base_channels=8)
    elif so2:
        model = SO2EquivariantUNet3D(cfg)
    else:
        model = EquivariantUNet3D(cfg)
    model.eval()
    return model, cfg


# ============================================================================
# Part 1 — Grid rotation tables
# ============================================================================
#
# Convention: volume stored as (B, C, D, H, W) where D=z, H=y, W=x
# Physical coordinates: ξ=W-axis(x), η=H-axis(y), ζ=D-axis(z), centred.
#
# The correspondence between torch.rot90 and rotation matrices was derived
# analytically (see report.md §8) and verified below in verify_grid_rotations().
#
# torch.rot90(x, k=k, dims=dims) ↔ rotation matrix R_GRID_ROTATIONS_5D[(dims_str, k)]
#
# Rotation matrices act on column vectors [x, y, z]^T in the array convention
# ξ=x=w, η=y=h, ζ=z=d (y increases in the row direction, same as array rows).

_d = {
    # (axis, k) → R as a list of rows   (R maps old → new physical coords)
    # Rotations about z (dims=[3,4], HW plane)
    ("z", 1): [[0., -1., 0.], [ 1.,  0., 0.], [0., 0., 1.]],   # R_z(+90°)
    ("z", 2): [[-1., 0., 0.], [ 0., -1., 0.], [0., 0., 1.]],   # R_z(180°)
    ("z", 3): [[0.,  1., 0.], [-1.,  0., 0.], [0., 0., 1.]],   # R_z(-90°)
    # Rotations about x (dims=[2,3], DH plane)
    ("x", 1): [[1.,  0.,  0.], [0.,  0.,  1.], [0., -1.,  0.]],  # R_x(-90°)
    ("x", 2): [[1.,  0.,  0.], [0., -1.,  0.], [0.,  0., -1.]],  # R_x(180°)
    ("x", 3): [[1.,  0.,  0.], [0.,  0., -1.], [0.,  1.,  0.]],  # R_x(+90°)
    # Rotations about y (dims=[2,4], DW plane)
    ("y", 1): [[0.,  0., -1.], [0.,  1.,  0.], [ 1.,  0.,  0.]],  # R_y(-90°)
    ("y", 2): [[-1., 0.,  0.], [0.,  1.,  0.], [ 0.,  0., -1.]],  # R_y(180°)
    ("y", 3): [[0.,  0.,  1.], [0.,  1.,  0.], [-1.,  0.,  0.]],  # R_y(+90°)
}
GRID_ROTATIONS: Dict[Tuple[str, int], torch.Tensor] = {
    k: torch.tensor(v, dtype=torch.float32) for k, v in _d.items()
}
AXIS_DIMS: Dict[str, List[int]] = {
    "z": [3, 4],
    "x": [2, 3],
    "y": [2, 4],
}


def rot90_volume(x: torch.Tensor, axis: str, k: int) -> torch.Tensor:
    """Apply 90° grid rotation to a (B,C,D,H,W) or (B,C,D,H,W) tensor."""
    return torch.rot90(x, k=k, dims=AXIS_DIMS[axis])


def verify_grid_rotations() -> bool:
    """
    Empirical sanity check: place a unit delta in a cubic volume and confirm
    it moves to the position predicted by the rotation matrix.
    """
    N = 8
    c = (N - 1) / 2.0
    # Test point in centred coordinates: ξ=1.5, η=-2.5, ζ=-0.5
    d_old, h_old, w_old = 3, 1, 5   # d=z, h=y, w=x
    r_old = torch.tensor([w_old - c, h_old - c, d_old - c])  # [x,y,z] centred

    all_ok = True
    for (axis, k), R in GRID_ROTATIONS.items():
        vol = torch.zeros(1, 1, N, N, N)
        vol[0, 0, d_old, h_old, w_old] = 1.0
        vol_rot = rot90_volume(vol, axis, k)

        # Find where the delta landed
        nz = (vol_rot[0, 0] == 1.0).nonzero(as_tuple=False)
        if len(nz) != 1:
            print(f"  FAIL ({axis}, k={k}): delta replicated or lost")
            all_ok = False
            continue
        d_new, h_new, w_new = nz[0].tolist()
        r_new_actual = torch.tensor([w_new - c, h_new - c, d_new - c])

        r_new_expected = R @ r_old
        err = (r_new_actual - r_new_expected).abs().max().item()
        if err > 0.1:
            print(f"  FAIL ({axis}, k={k}): expected {r_new_expected.tolist()}, "
                  f"got {r_new_actual.tolist()}")
            all_ok = False

    return all_ok


# ============================================================================
# Part 2 — Wigner D-matrix utilities
# ============================================================================

def get_wigner_d(irreps_str: str, R: torch.Tensor) -> torch.Tensor:
    """
    Return the block-diagonal Wigner D-matrix for the given irreps and rotation R.

    For l=0 (scalar): D = [[1]] block (rotation-invariant).
    For l=1 (vector): D = R itself (3×3).
    For l=2 (rank-2 tensor): D = 5×5 Wigner D-matrix.

    The full D has shape (irreps.dim, irreps.dim) and is block-diagonal.
    """
    irreps = Irreps(irreps_str)
    try:
        D = irreps.D_from_matrix(R)
    except AttributeError:
        # Older e3nn: convert via ZYZ Euler angles
        angles = o3.matrix_to_angles(R)
        D = irreps.D_from_angles(*angles)
    return D


def apply_d_to_features(feat: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Apply Wigner D-matrix D (irreps.dim × irreps.dim) to feature channels.

    feat : (B, C, D, H, W)
    D    : (C, C)
    Returns (B, C, D, H, W)

    This implements: feat_out[b, c, r] = Σ_c' D[c, c'] · feat[b, c', r]
    """
    B, C, Dv, H, W = feat.shape
    feat_flat = feat.reshape(B, C, -1)                    # (B, C, DHW)
    result = torch.einsum('ij,bjk->bik', D, feat_flat)    # (B, C, DHW)
    return result.reshape(B, C, Dv, H, W)


def verify_d_for_l1(R: torch.Tensor) -> bool:
    """
    Check that D^{l=1}(R) = R.
    For l=1 (vector) irreps, the Wigner D-matrix equals the rotation matrix itself.
    This is a fundamental consistency check on the e3nn convention.
    """
    D = get_wigner_d("1x1o", R)
    err = (D - R).abs().max().item()
    return err < 1e-5


# ============================================================================
# Part 3 — Core equivariance tests
# ============================================================================

@torch.no_grad()
def test_output_invariance(
    model: nn.Module,
    axis: str,
    k: int,
    vol_size: int = 32,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Test that the output heatmap is rotation-INVARIANT for a 90° grid rotation.

    Checks:  model(rot90(X)) ≈ rot90(model(X))
    i.e.     applying the rotation to the INPUT  should give the same result as
             applying the same rotation to the OUTPUT of the original forward pass.

    For an SO(3)-equivariant network with scalar output irreps (0e), this should
    hold to machine precision (~1e-5) on exact grid rotations.

    Returns: (mean_abs_error, max_abs_error)
    """
    model.eval()
    torch.manual_seed(seed)
    X = torch.randn(1, 1, vol_size, vol_size, vol_size)

    # Baseline prediction
    hm_orig = model(X)                           # (1, C, D, H, W)

    # Prediction on rotated input
    X_rot = rot90_volume(X, axis, k)
    hm_on_rotated = model(X_rot)                 # (1, C, D, H, W)

    # Expected: same rotation applied to the baseline output
    hm_expected = rot90_volume(hm_orig, axis, k)

    err = (hm_on_rotated - hm_expected).abs()
    return err.mean().item(), err.max().item()


@torch.no_grad()
def test_feature_equivariance(
    model: EquivariantUNet3D,
    cfg: ModelConfig,
    axis: str,
    k: int,
    encoder_level: int,
    vol_size: int = 32,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Test that intermediate encoder features are rotation-EQUIVARIANT.

    Equivariance condition:
        f_level(R·X)(r)  =  D(R) · f_level(X)(R⁻¹·r)

    Rearranged for comparison:
        f_level(R·X)  =  D(R) · spatial_rot(f_level(X), R⁻¹)

    where spatial_rot(feat, R⁻¹) = torch.rot90(feat, k_inv, dims).

    Returns: (mean_abs_error, max_abs_error)
    """
    irreps_str = cfg.encoder_irreps[encoder_level]
    R = GRID_ROTATIONS[(axis, k)]
    dims = AXIS_DIMS[axis]

    activations: Dict[str, torch.Tensor] = {}

    def hook(mod, inp, out):
        activations["feat"] = out.detach()

    handle = model.enc_blocks[encoder_level].register_forward_hook(hook)

    model.eval()
    torch.manual_seed(seed)
    X = torch.randn(1, 1, vol_size, vol_size, vol_size)

    # Run on original
    model(X)
    feat_orig = activations["feat"].clone()

    # Run on rotated input
    X_rot = rot90_volume(X, axis, k)
    model(X_rot)
    feat_on_rot = activations["feat"].clone()

    handle.remove()

    # Build expected: D(R) · spatial_rot(feat_orig, R⁻¹)
    #   R⁻¹ corresponds to torch.rot90(·, k=4-k, dims)
    k_inv = (4 - k) % 4
    feat_at_Rinv_r = torch.rot90(feat_orig, k=k_inv, dims=dims)
    D = get_wigner_d(irreps_str, R)
    feat_expected = apply_d_to_features(feat_at_Rinv_r, D)

    err = (feat_on_rot - feat_expected).abs()
    return err.mean().item(), err.max().item()


def run_all_grid_tests(
    equivariant_model: nn.Module,
    baseline_model: nn.Module,
    cfg: ModelConfig,
    vol_size: int = 32,
) -> None:
    """
    Run the full test suite for all 9 non-trivial 90° grid rotations.
    Prints a report comparing equivariant vs standard model.
    """
    print("\n" + "=" * 70)
    print("INVARIANCE TEST: model(R·X) vs R·model(X)")
    print("=" * 70)
    print(f"{'Rotation':<20} {'Equiv MAE':>12} {'Equiv MAX':>12} "
          f"{'Baseline MAE':>14} {'Baseline MAX':>14}")
    print("-" * 70)

    for (axis, k) in GRID_ROTATIONS:
        eq_mean, eq_max = test_output_invariance(
            equivariant_model, axis, k, vol_size=vol_size)
        bl_mean, bl_max = test_output_invariance(
            baseline_model, axis, k, vol_size=vol_size)
        label = f"R_{axis}({k*90}°)"
        status_eq = "✓" if eq_max < 1e-3 else "✗"
        status_bl = "✓" if bl_max < 1e-3 else "✗"
        print(f"{label:<20} {eq_mean:>10.2e} {status_eq} {eq_max:>10.2e} {status_eq} "
              f"{bl_mean:>12.2e} {status_bl} {bl_max:>12.2e} {status_bl}")

    print("\n" + "=" * 70)
    print("EQUIVARIANCE TEST: f_level(R·X) vs D(R)·spatial_rot(f_level(X), R⁻¹)")
    print("(Equivariant model only — encoder levels 0, 1, 2)")
    print("=" * 70)
    print(f"{'Rotation':<20} {'Level 0':>12} {'Level 1':>12} {'Level 2':>12}")
    print("-" * 70)

    for (axis, k) in GRID_ROTATIONS:
        means = []
        for level in [0, 1, 2]:
            mean, mx = test_feature_equivariance(
                equivariant_model, cfg, axis, k, level, vol_size=vol_size)
            means.append(mean)
        label = f"R_{axis}({k*90}°)"
        print(f"{label:<20} " + " ".join(f"{m:>10.2e} {'✓' if m < 1e-3 else '✗'}" for m in means))

    print()


# ============================================================================
# Part 4 — Arbitrary-angle equivariance (scipy-based)
# ============================================================================

def rotate_volume_scipy(
    vol: np.ndarray,
    angle_deg: float,
    axis: str = "z",
    order: int = 1,
) -> np.ndarray:
    """
    Rotate a (D, H, W) volume by `angle_deg` degrees about `axis` using
    scipy.ndimage.rotate with trilinear interpolation (order=1).

    axis='z' rotates in the H-W (y-x) plane.
    axis='x' rotates in the D-H (z-y) plane.
    axis='y' rotates in the D-W (z-x) plane.
    """
    from scipy.ndimage import rotate as scipy_rotate
    plane = {"z": (1, 2), "x": (0, 1), "y": (0, 2)}[axis]
    return scipy_rotate(
        vol, angle_deg, axes=plane, reshape=False,
        order=order, mode="constant", cval=0.0,
    )


@torch.no_grad()
def measure_equivariance_error_vs_angle(
    model: nn.Module,
    angles_deg: np.ndarray,
    axis: str = "z",
    vol_size: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """
    Measure output invariance error at each rotation angle.

    For each angle θ:
      1. Rotate the input volume by θ (scipy, trilinear interp).
      2. Run forward on both original and rotated volumes.
      3. Rotate the original output by θ.
      4. Return mean |model(R_θ·X) - R_θ·model(X)|.
    """
    model.eval()
    np.random.seed(seed)
    torch.manual_seed(seed)

    vol_np = np.random.randn(vol_size, vol_size, vol_size).astype(np.float32)
    X = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)   # (1,1,D,H,W)

    with torch.no_grad():
        hm_orig = model(X).squeeze(0).cpu().numpy()            # (C,D,H,W)

    errors = []
    for angle in angles_deg:
        # Rotate input
        vol_rot_np = rotate_volume_scipy(vol_np, angle, axis)
        X_rot = torch.from_numpy(vol_rot_np).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            hm_on_rot = model(X_rot).squeeze(0).cpu().numpy()  # (C,D,H,W)

        # Rotate original prediction by the same angle
        hm_expected = np.stack([
            rotate_volume_scipy(hm_orig[c], angle, axis)
            for c in range(hm_orig.shape[0])
        ])

        err = np.abs(hm_on_rot - hm_expected).mean()
        errors.append(err)

    return np.array(errors)


# ============================================================================
# Part 5 — Visualisations
# ============================================================================

def plot_equivariance_error_vs_angle(
    equivariant_model: nn.Module,
    baseline_model: nn.Module,
    vol_size: int = 32,
    save_path: str = "equivariance_error_vs_angle.png",
) -> None:
    """
    Plot mean absolute equivariance error vs rotation angle for both models.

    An ideally equivariant model shows ~flat, near-zero error at all angles.
    A standard model shows ~0 only at multiples of 90°.

    The residual error for the equivariant model reflects grid discretisation
    artefacts from scipy's trilinear interpolation (not an architectural failure).
    """
    import matplotlib.pyplot as plt

    angles = np.linspace(0, 360, 73, endpoint=False)   # every 5°

    print("  Computing equivariant model errors...", flush=True)
    err_eq = measure_equivariance_error_vs_angle(equivariant_model, angles, vol_size=vol_size)

    print("  Computing baseline model errors...", flush=True)
    err_bl = measure_equivariance_error_vs_angle(baseline_model, angles, vol_size=vol_size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(angles, err_eq, label="SO(3)-Equivariant U-Net", linewidth=2, color="royalblue")
    ax.plot(angles, err_bl, label="Standard 3D U-Net (baseline)", linewidth=2,
            color="tomato", linestyle="--")

    # Mark 90° grid positions
    for a in [0, 90, 180, 270, 360]:
        ax.axvline(a, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Rotation angle about z-axis (degrees)")
    ax.set_ylabel("Mean |model(R·X) − R·model(X)|")
    ax.set_title("Equivariance error vs rotation angle\n"
                 "(residual for equivariant model = grid interpolation artefacts)")
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_heatmap_comparison(
    model: nn.Module,
    vol_size: int = 32,
    save_path: str = "heatmap_comparison.png",
) -> None:
    """
    Visualise output heatmap invariance for 90° rotation about z.

    Shows 3 columns:
      Col 1: original prediction (one class, mid z-slice)
      Col 2: prediction on rotated input  — should match col 3
      Col 3: original prediction spatially rotated 90° CCW

    Columns 2 and 3 should look identical for a perfectly invariant model.
    """
    import matplotlib.pyplot as plt

    model.eval()
    np.random.seed(42)
    torch.manual_seed(42)
    vol_np = np.random.randn(vol_size, vol_size, vol_size).astype(np.float32)

    X = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)
    X_rot = rot90_volume(X, "z", 1)

    with torch.no_grad():
        hm = model(X).squeeze(0).cpu().numpy()         # (C,D,H,W)
        hm_rot = model(X_rot).squeeze(0).cpu().numpy() # (C,D,H,W)

    z_mid = vol_size // 2
    cls = 0  # first class

    hm_expected = rot90_volume(
        torch.from_numpy(hm).unsqueeze(0), "z", 1
    ).squeeze(0).numpy()

    diff = np.abs(hm_rot[cls, z_mid] - hm_expected[cls, z_mid])

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    vmin = min(hm[cls, z_mid].min(), hm_rot[cls, z_mid].min())
    vmax = max(hm[cls, z_mid].max(), hm_rot[cls, z_mid].max())

    axes[0].imshow(hm[cls, z_mid], vmin=vmin, vmax=vmax, cmap="hot", origin="upper")
    axes[0].set_title("model(X)\n(original)")

    axes[1].imshow(hm_rot[cls, z_mid], vmin=vmin, vmax=vmax, cmap="hot", origin="upper")
    axes[1].set_title("model(R·X)\n(input rotated 90°)")

    axes[2].imshow(hm_expected[cls, z_mid], vmin=vmin, vmax=vmax, cmap="hot", origin="upper")
    axes[2].set_title("R·model(X)\n(output rotated 90°)")

    im = axes[3].imshow(diff, cmap="RdBu_r", origin="upper")
    plt.colorbar(im, ax=axes[3])
    axes[3].set_title(f"|model(R·X) − R·model(X)|\nmax={diff.max():.2e}")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Heatmap invariance check (class {cls}, z-slice {z_mid})", y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def plot_wigner_d_structure(
    irreps_str: str,
    R: torch.Tensor,
    save_path: str = "wigner_d_structure.png",
) -> None:
    """
    Visualise the block-diagonal structure of the Wigner D-matrix.

    Each block corresponds to one irrep type (0e, 1o, 2e ...).
    Scalars (0e) have 1×1 blocks = [1.0] (invariant).
    Vectors (1o) have 3×3 blocks = the rotation matrix.
    Rank-2 tensors (2e) have 5×5 blocks = D_2(R).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    irreps = Irreps(irreps_str)
    D = get_wigner_d(irreps_str, R).numpy()

    fig, (ax_d, ax_abs) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax_d.imshow(D, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im1, ax=ax_d, fraction=0.046)
    ax_d.set_title(f"Wigner D-matrix\nirr={irreps_str}")

    im2 = ax_abs.imshow(np.abs(D), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax_abs, fraction=0.046)
    ax_abs.set_title("|D|  (block structure)")

    # Draw block boundaries
    offset = 0
    colours = {"0": "red", "1": "blue", "2": "green"}
    for mul, ir in irreps:
        block_size = mul * ir.dim
        colour = colours.get(str(ir.l), "orange")
        for ax in [ax_d, ax_abs]:
            rect = patches.Rectangle(
                (offset - 0.5, offset - 0.5),
                block_size, block_size,
                linewidth=1.5, edgecolor=colour, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(offset + block_size / 2, offset - 2,
                    f"{mul}×{ir}", color=colour, ha="center", fontsize=7)
        offset += block_size

    ax_d.set_xlabel("Input channel index")
    ax_d.set_ylabel("Output channel index")
    ax_abs.set_xlabel("Input channel index")
    ax_abs.set_ylabel("Output channel index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_decomposition(
    model: EquivariantUNet3D,
    cfg: ModelConfig,
    vol_size: int = 32,
    encoder_level: int = 0,
    save_path: str = "feature_decomposition.png",
) -> None:
    """
    Decompose encoder features into scalar / vector / tensor components and
    visualise their magnitudes across a mid-plane slice.

    Scalar magnitude  = |f_0e| (directly the channel value for 0e irreps)
    Vector magnitude  = sqrt(Σ f_{1o}²) per position
    Tensor magnitude  = sqrt(Σ f_{2e}²) per position
    """
    import matplotlib.pyplot as plt

    model.eval()
    torch.manual_seed(0)
    X = torch.randn(1, 1, vol_size, vol_size, vol_size)

    activation: Dict[str, torch.Tensor] = {}

    def hook(mod, inp, out):
        activation["feat"] = out.detach()

    handle = model.enc_blocks[encoder_level].register_forward_hook(hook)
    with torch.no_grad():
        model(X)
    handle.remove()

    feat = activation["feat"].squeeze(0).cpu().numpy()   # (C, D, H, W)
    irreps = Irreps(cfg.encoder_irreps[encoder_level])

    # Decompose by irrep type
    scalar_ch, vector_ch, tensor_ch = [], [], []
    offset = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        block = feat[offset:offset + dim]     # (dim, D, H, W)
        if ir.l == 0:
            scalar_ch.append(block)
        elif ir.l == 1:
            # Reshape to (mul, 3, D, H, W) and compute per-position norm over dim-3 axis
            vector_ch.append(block.reshape(mul, ir.dim, *feat.shape[1:]))
        elif ir.l == 2:
            tensor_ch.append(block.reshape(mul, ir.dim, *feat.shape[1:]))
        offset += dim

    z_mid = feat.shape[1] // 2

    def mag(chunks):
        if not chunks:
            return np.zeros(feat.shape[1:])
        return np.sqrt(sum(np.sum(c ** 2, axis=(0, 1)) for c in chunks))

    scalar_mag = np.abs(np.concatenate(scalar_ch, axis=0)).mean(axis=0) if scalar_ch else np.zeros(feat.shape[1:])
    vector_mag = mag(vector_ch)
    tensor_mag = mag(tensor_ch)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes,
                                [scalar_mag, vector_mag, tensor_mag],
                                ["Scalar (l=0) mean |f|",
                                 "Vector (l=1) field norm",
                                 "Tensor (l=2) field norm"]):
        im = ax.imshow(data[z_mid], cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title}\n(encoder level {encoder_level}, z-slice {z_mid})")
        ax.axis("off")

    fig.suptitle(f"Irrep decomposition of encoder features (level {encoder_level})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# Part 6 — pytest-compatible test functions
# ============================================================================

def _get_models():
    """Cached model creation for pytest (avoids rebuilding per-test)."""
    if not hasattr(_get_models, "_cache"):
        eq_model,  cfg = build_test_model(equivariant=True,  so2=False)
        so2_model, _   = build_test_model(equivariant=True,  so2=True)
        bl_model,  _   = build_test_model(equivariant=False)
        _get_models._cache = (eq_model, cfg, so2_model, bl_model)
    return _get_models._cache


def test_grid_rotation_conventions():
    """Verify that GRID_ROTATIONS table correctly describes each torch.rot90."""
    ok = verify_grid_rotations()
    assert ok, "Rotation matrix conventions do not match torch.rot90 behaviour"


def test_wigner_d_l1_equals_R():
    """For l=1 irreps, D(R) must equal R (fundamental group-theory identity)."""
    for (axis, k), R in GRID_ROTATIONS.items():
        ok = verify_d_for_l1(R)
        assert ok, f"D^l=1({axis},{k}) != R"


def test_equivariant_output_invariance():
    """SO(3) model output should be invariant to all 90° grid rotations."""
    eq_model, cfg, _, _ = _get_models()
    for (axis, k) in GRID_ROTATIONS:
        mean_err, max_err = test_output_invariance(eq_model, axis, k, vol_size=32)
        assert max_err < 5e-3, (
            f"Output invariance failed for R_{axis}(k={k}): "
            f"mean={mean_err:.2e}, max={max_err:.2e}"
        )


def test_equivariant_feature_equivariance():
    """SO(3) encoder features should transform correctly under all 90° rotations."""
    eq_model, cfg, _, _ = _get_models()
    for (axis, k) in GRID_ROTATIONS:
        for level in [0, 1]:
            mean_err, max_err = test_feature_equivariance(
                eq_model, cfg, axis, k, level, vol_size=32)
            assert max_err < 5e-3, (
                f"Feature equivariance failed: R_{axis}(k={k}), level={level}: "
                f"mean={mean_err:.2e}, max={max_err:.2e}"
            )


def test_baseline_not_equivariant():
    """Standard U-Net should FAIL the equivariance test for non-trivial rotations."""
    _, cfg, _, bl_model = _get_models()
    fails = 0
    for (axis, k) in GRID_ROTATIONS:
        if k == 2:   # 180° can accidentally pass for symmetric architectures
            continue
        _, max_err = test_output_invariance(bl_model, axis, k, vol_size=32)
        if max_err > 1e-3:
            fails += 1
    assert fails > 0, "Standard model unexpectedly passed all equivariance tests"


# ============================================================================
# SO(2) equivariance tests
# ============================================================================

def test_so2_invariant_to_z_rotations():
    """
    SO(2) model output must be invariant to in-plane (z-axis) rotations.

    axis='z' rotations operate in the H-W (y-x) plane — the in-plane
    direction for cryo-ET where the missing wedge does NOT break symmetry.
    """
    _, _, so2_model, _ = _get_models()
    for k in [1, 2, 3]:          # 90°, 180°, 270°
        mean_err, max_err = test_output_invariance(so2_model, "z", k, vol_size=32)
        assert max_err < 5e-3, (
            f"SO(2) model failed z-rotation invariance: k={k}, "
            f"mean={mean_err:.2e}, max={max_err:.2e}"
        )


def test_so2_NOT_invariant_to_out_of_plane_rotations():
    """
    SO(2) model output must NOT be invariant to out-of-plane rotations.

    axis='x' and axis='y' rotations tilt the volume out of the z=0 plane —
    the beam direction changes, so the missing-wedge PSF orientation changes.
    The SO(2) model is intentionally not equivariant for these rotations.
    """
    _, _, so2_model, _ = _get_models()
    # At least some out-of-plane 90° rotations should produce non-trivial error
    fails = 0
    for axis in ["x", "y"]:
        for k in [1, 3]:         # 90° and 270° out-of-plane
            _, max_err = test_output_invariance(so2_model, axis, k, vol_size=32)
            if max_err > 1e-3:
                fails += 1
    assert fails > 0, (
        "SO(2) model unexpectedly passed all out-of-plane rotation tests. "
        "It should only be equivariant in-plane (z-axis)."
    )


def test_so2_z_feature_equivariance():
    """
    SO(2) model features should be equivariant under in-plane (z-axis) rotations.
    Same formula as SO(3) feature equivariance but restricted to axis='z'.
    """
    _, cfg, so2_model, _ = _get_models()
    for k in [1, 2, 3]:
        for level in [0, 1]:
            mean_err, max_err = test_feature_equivariance(
                so2_model, cfg, "z", k, level, vol_size=32)
            assert max_err < 5e-3, (
                f"SO(2) z-feature equivariance failed: k={k}, level={level}, "
                f"mean={mean_err:.2e}, max={max_err:.2e}"
            )


def run_so2_grid_tests(so2_model: nn.Module, cfg: ModelConfig, vol_size: int = 32) -> None:
    """
    Print a diagnostic table showing which rotations the SO(2) model is
    equivariant to (z-axis only) and which it is not (x, y axes).
    """
    print("\n" + "=" * 70)
    print("SO(2) MODEL — INVARIANCE TEST SUMMARY")
    print("Expected: z-axis rotations PASS (~0), x/y-axis rotations FAIL (>0)")
    print("=" * 70)
    print(f"{'Rotation':<20} {'MAE':>12} {'MAX':>12} {'Expected':>12}")
    print("-" * 70)
    for (axis, k) in GRID_ROTATIONS:
        mean_err, max_err = test_output_invariance(so2_model, axis, k, vol_size=vol_size)
        expected = "PASS" if axis == "z" else "FAIL"
        actual   = "✓" if max_err < 5e-3 else "✗"
        label    = f"R_{axis}({k * 90}°)"
        print(f"{label:<20} {mean_err:>10.2e} {actual} {max_err:>10.2e} {actual} {expected:>12}")
    print()


# ============================================================================
# Part 7 — Main runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-only", action="store_true",
                        help="Skip visualisations (faster)")
    parser.add_argument("--vol-size", type=int, default=32,
                        help="Test volume size (must be divisible by 8)")
    args = parser.parse_args()

    print("Building models...")
    eq_model,  cfg = build_test_model(equivariant=True,  so2=False, lmax=2)
    so2_model, _   = build_test_model(equivariant=True,  so2=True,  lmax=2)
    bl_model,  _   = build_test_model(equivariant=False)

    # 1. Verify rotation conventions
    print("\n--- Verifying rotation matrix conventions ---")
    ok = verify_grid_rotations()
    print(f"  Grid rotation conventions: {'OK' if ok else 'FAILED'}")

    print("\n--- Verifying D^{l=1}(R) = R ---")
    all_ok = all(verify_d_for_l1(R) for R in GRID_ROTATIONS.values())
    print(f"  Wigner D l=1 check: {'OK' if all_ok else 'FAILED'}")

    # 2. Core test suite — SO(3)
    run_all_grid_tests(eq_model, bl_model, cfg, vol_size=args.vol_size)

    # 3. SO(2) diagnostic
    print("\n--- SO(2) equivariance diagnostic ---")
    run_so2_grid_tests(so2_model, cfg, vol_size=args.vol_size)

    if args.tests_only:
        return

    # 4. Visualisations
    print("\n--- Generating visualisations ---")

    print("  1/4  Equivariance error vs angle...")
    plot_equivariance_error_vs_angle(
        eq_model, bl_model,
        vol_size=args.vol_size,
        save_path="equivariance_error_vs_angle.png",
    )

    print("  2/4  Heatmap comparison...")
    plot_heatmap_comparison(
        eq_model,
        vol_size=args.vol_size,
        save_path="heatmap_comparison.png",
    )

    print("  3/4  Wigner D-matrix structure...")
    R_ex = GRID_ROTATIONS[("z", 1)]
    irreps_ex = cfg.encoder_irreps[0]
    plot_wigner_d_structure(irreps_ex, R_ex, save_path="wigner_d_structure.png")

    print("  4/4  Feature decomposition...")
    plot_feature_decomposition(
        eq_model, cfg,
        vol_size=args.vol_size,
        encoder_level=0,
        save_path="feature_decomposition.png",
    )

    print("\nDone. Plots saved to current directory.")


if __name__ == "__main__":
    main()
