"""
SO(3)- / SO(2)-Equivariant 3D U-Net for Protein Centroid Detection.

Architecture:
  - Encoder: e3nn equivariant voxel convolutions with gated nonlinearities
  - Decoder: equivariant upsampling with skip connections
  - Head: final convolution to scalar (l=0) irreps → rotation-invariant heatmaps

Key design choices:
  1. Features are irreducible representations of SO(3): scalars (l=0),
     vectors (l=1), rank-2 tensors (l=2).
  2. Convolution kernels are built as R(||r||) * Y_l(r/||r||) — radial
     functions times spherical harmonics.
  3. Gated nonlinearities: higher-order features (l>0) are gated by learned
     scalar features passed through sigmoid.
  4. The final layer projects to "Cx0e" (C scalar channels for C protein
     classes), which is automatically SO(3)-invariant.
  5. Radial MLP outputs are normalised by 1/sqrt(weight_numel) to stabilise
     variance regardless of the number of TP coupling paths.
  6. max_radius for GaussianRadialBasis covers all kernel corners:
       3D kernels: sqrt(3) * (K//2)
       2D xy kernels (SO2 model): sqrt(2) * (K//2)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import e3nn
from e3nn import o3
from e3nn.nn import BatchNorm as E3BatchNorm
from e3nn.o3 import Irreps, FullyConnectedTensorProduct

from config import ModelConfig, NUM_CLASSES


# ---------------------------------------------------------------------------
# Radial Basis Functions
# ---------------------------------------------------------------------------

class GaussianRadialBasis(nn.Module):
    """Gaussian radial basis functions for encoding ||r||."""

    def __init__(self, num_basis: int = 8, max_radius: float = 3.5):
        super().__init__()
        self.num_basis = num_basis
        centers = torch.linspace(0, max_radius, num_basis)
        self.register_buffer("centers", centers)
        self.width = 0.5 * (max_radius / num_basis)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """distances: (...,) → (..., num_basis)"""
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.width ** 2))


# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------

class GatedNonlinearity(nn.Module):
    """
    Gated nonlinearity for equivariant features.

    Scalars (l=0) are passed through SiLU.
    Higher-order features (l>0) are gated by learned scalar features
    passed through sigmoid: output = sigmoid(gate) * feature.
    """

    def __init__(self, irreps: Irreps):
        super().__init__()
        self.irreps = irreps

        # Count scalar and non-scalar multiplicities
        self.scalar_idx = []
        self.nonscalar_idx = []
        self.nonscalar_dims = []
        offset = 0
        for mul, ir in irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                self.scalar_idx.append((offset, offset + dim))
            else:
                self.nonscalar_idx.append((offset, offset + dim))
                self.nonscalar_dims.append((mul, ir.dim))
            offset += dim

        # Gate projection: from existing scalars to one gate per non-scalar channel
        n_scalars = sum(e - s for s, e in self.scalar_idx)
        n_gates = sum(mul for mul, _ in self.nonscalar_dims)
        if n_gates > 0 and n_scalars > 0:
            self.gate_proj = nn.Linear(n_scalars, n_gates)
        else:
            self.gate_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, irreps.dim, ...) — spatial dims preserved."""
        spatial_shape = x.shape[2:]
        batch = x.shape[0]

        # Extract scalars
        scalar_parts = []
        for s, e in self.scalar_idx:
            scalar_parts.append(x[:, s:e])
        if scalar_parts:
            scalars = torch.cat(scalar_parts, dim=1)
            # Apply SiLU to scalars
            activated_scalars = F.silu(scalars)
        else:
            activated_scalars = None

        # Gate non-scalar features
        if self.gate_proj is not None and self.nonscalar_idx:
            # Compute gates from scalars
            # Move spatial dims to batch for linear layer
            s_flat = scalars.permute(0, *range(2, 2 + len(spatial_shape)), 1)
            s_flat = s_flat.reshape(-1, scalars.shape[1])
            gates = torch.sigmoid(self.gate_proj(s_flat))
            gates = gates.reshape(batch, *spatial_shape, -1)
            gates = gates.permute(0, len(spatial_shape) + 1, *range(1, 1 + len(spatial_shape)))

            # Apply gates
            gate_offset = 0
            gated_parts = []
            for (s, e), (mul, dim) in zip(self.nonscalar_idx, self.nonscalar_dims):
                feat = x[:, s:e]  # (batch, mul*dim, ...)
                feat = feat.reshape(batch, mul, dim, *spatial_shape)
                g = gates[:, gate_offset:gate_offset + mul]  # (batch, mul, ...)
                g = g.unsqueeze(2)  # (batch, mul, 1, ...)
                gated = feat * g
                gated = gated.reshape(batch, mul * dim, *spatial_shape)
                gated_parts.append(gated)
                gate_offset += mul
        else:
            gated_parts = [x[:, s:e] for s, e in self.nonscalar_idx]

        # Reconstruct output in original irrep order
        out_parts = {}
        offset = 0
        scalar_part_idx = 0
        nonscalar_part_idx = 0
        activated_scalar_offset = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                if activated_scalars is not None:
                    out_parts[offset] = activated_scalars[:, activated_scalar_offset:activated_scalar_offset + dim]
                    activated_scalar_offset += dim
                else:
                    out_parts[offset] = x[:, offset:offset + dim]
            else:
                if nonscalar_part_idx < len(gated_parts):
                    out_parts[offset] = gated_parts[nonscalar_part_idx]
                    nonscalar_part_idx += 1
            offset += dim

        # Concatenate in order
        result = torch.cat([out_parts[k] for k in sorted(out_parts.keys())], dim=1)
        return result


class NormNonlinearity(nn.Module):
    """Norm-based nonlinearity: rescales l>0 features by a nonlinearity applied to their norm."""

    def __init__(self, irreps: Irreps, epsilon: float = 1e-8):
        super().__init__()
        self.irreps = irreps
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[2:]
        batch = x.shape[0]
        parts = []
        offset = 0
        for mul, ir in self.irreps:
            dim = mul * ir.dim
            feat = x[:, offset:offset + dim]
            if ir.l == 0:
                parts.append(F.silu(feat))
            else:
                feat = feat.reshape(batch, mul, ir.dim, *spatial_shape)
                norm = feat.norm(dim=2, keepdim=True) + self.epsilon
                scale = F.silu(norm) / norm
                feat = feat * scale
                feat = feat.reshape(batch, mul * ir.dim, *spatial_shape)
                parts.append(feat)
            offset += dim
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Equivariant Pooling & Upsampling
# ---------------------------------------------------------------------------

class EquivariantAvgPool3d(nn.Module):
    """SO(3)-equivariant average pooling (preserves equivariance since it's linear)."""

    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class EquivariantUpsample3d(nn.Module):
    """SO(3)-equivariant upsampling via trilinear interpolation (linear → equivariant)."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False,
        )


# ---------------------------------------------------------------------------
# Lightweight Equivariant Conv (for efficiency)
# ---------------------------------------------------------------------------

class LightEquivariantConv(nn.Module):
    """
    Lightweight SO(3)-equivariant convolution using pre-computed kernel bases.

    Instead of looping over K^3 kernel positions with individual tensor products,
    this pre-computes the full kernel tensor and applies it as a single 3D
    convolution — much faster for training.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        kernel_size: int = 5,
        lmax: int = 2,
        num_radial_basis: int = 8,
        use_batch_norm: bool = True,
        activation: str = "gated",
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Pre-compute kernel grid
        r = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1)
        grid_flat = grid.reshape(-1, 3)
        distances = grid_flat.norm(dim=-1)
        unit_vec = grid_flat / (distances.unsqueeze(-1) + 1e-8)

        # Pre-compute spherical harmonics (fixed, not learned)
        sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        sh = o3.spherical_harmonics(sh_irreps, unit_vec, normalize=True)
        self.register_buffer("sh", sh)  # (K^3, sh_dim)
        self.register_buffer("distances", distances)  # (K^3,)

        # Radial basis: max_radius must cover the farthest kernel corner.
        # For a 3D KxKxK grid with half-width h = K//2, the corner is at
        # distance sqrt(h²+h²+h²) = h*sqrt(3).  Add 0.5 so that the outermost
        # Gaussian basis centre sits exactly at the corner, not beyond it.
        max_r_3d = math.sqrt(3) * (kernel_size // 2) + 0.5
        self.radial_basis = GaussianRadialBasis(num_radial_basis, max_r_3d)

        # Tensor product (shared_weights=False: weights come from radial MLP)
        self.tp = FullyConnectedTensorProduct(
            irreps_in, sh_irreps, irreps_out, shared_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )
        # Normalise output scale so TP weight variance is O(1) regardless of
        # how many coupling paths weight_numel encodes.  Without this, variance
        # grows with weight_numel and training is unstable for large models.
        with torch.no_grad():
            self.radial_mlp[-1].weight.div_(math.sqrt(max(self.tp.weight_numel, 1)))
            if self.radial_mlp[-1].bias is not None:
                self.radial_mlp[-1].bias.zero_()

        # Batch norm
        # instance=True uses per-sample statistics, which is the correct
        # choice for 3D segmentation at batch_size=2 (DDP SyncBN is fragile
        # at this scale). nnU-Net's default for the same reason.
        self.bn = E3BatchNorm(irreps_out, instance=True) if use_batch_norm else None

        # Nonlinearity
        if activation == "gated":
            self.activation = GatedNonlinearity(irreps_out)
        elif activation == "norm":
            self.activation = NormNonlinearity(irreps_out)
        else:
            self.activation = None

    def _build_kernel(self) -> torch.Tensor:
        """
        Build the (C_out, C_in, K, K, K) equivariant convolution kernel.

        The kernel is constructed as:
            K(r_k) = TP(·, Y_l(r_k / |r_k|), w(|r_k|))
        where TP is a FullyConnectedTensorProduct acting as a linear map from
        input irreps to output irreps.  Since TP is linear in its first argument
        (for fixed spherical harmonics and radial weights), the full kernel matrix
        can be extracted as K[:, :, k] = TP(e_i, sh_k, w_k) for each basis
        vector e_i — i.e. one batched TP call over all K³ × C_in inputs.

        This replaces the original Python loop over K³ positions with a single
        vectorised call, which is ~K³x faster on GPU.
        """
        K3 = self.kernel_size ** 3
        C_in = self.irreps_in.dim
        C_out = self.irreps_out.dim

        # Radial features → TP weights  (K³, weight_numel)
        radial_feats = self.radial_basis(self.distances)   # (K³, num_basis)
        kernel_weights = self.radial_mlp(radial_feats)     # (K³, weight_numel)

        # Identity basis for input irreps — extracting the kernel matrix via:
        #   kernel_col_i = TP(e_i, sh_k, w_k),  stacked over all k and i.
        eye = torch.eye(C_in, device=self.sh.device, dtype=self.sh.dtype)  # (C_in, C_in)

        # Expand to (K³ × C_in) batch dimension for a single vectorised TP call.
        # eye_batch[k*C_in + i] = e_i   (same for all k)
        # sh_batch [k*C_in + i] = sh_k  (same for all i within a given k)
        # w_batch  [k*C_in + i] = w_k   (same for all i within a given k)
        eye_batch = eye.unsqueeze(0).expand(K3, -1, -1).contiguous().reshape(K3 * C_in, C_in)
        sh_batch = self.sh.unsqueeze(1).expand(-1, C_in, -1).contiguous().reshape(K3 * C_in, -1)
        w_batch = kernel_weights.unsqueeze(1).expand(-1, C_in, -1).contiguous().reshape(K3 * C_in, -1)

        # Single batched TP: (K³*C_in, C_out)
        kernel_flat = self.tp(eye_batch, sh_batch, w_batch)

        # Reshape to (K³, C_in, C_out) → permute → (K³, C_out, C_in)
        kernel = kernel_flat.reshape(K3, C_in, C_out).permute(0, 2, 1)  # (K³, C_out, C_in)
        K = self.kernel_size
        kernel = kernel.reshape(K, K, K, C_out, C_in).permute(3, 4, 0, 1, 2)  # (C_out, C_in, K, K, K)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C_in, D, H, W) → (batch, C_out, D, H, W)"""
        kernel = self._build_kernel()
        out = F.conv3d(x, kernel, padding=self.padding)

        # Batch norm
        if self.bn is not None:
            B, C, D, H, W = out.shape
            out = out.permute(0, 2, 3, 4, 1).reshape(-1, C)
            out = self.bn(out)
            out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Nonlinearity
        if self.activation is not None:
            out = self.activation(out)

        return out


# ---------------------------------------------------------------------------
# Equivariant Residual Block
# ---------------------------------------------------------------------------

class EquivariantResBlock(nn.Module):
    """Two equivariant convolutions with a residual connection.

    The `conv_cls` parameter selects which convolution class to use:
      - LightEquivariantConv (default) → SO(3)-equivariant
      - SO2EquivariantConv            → SO(2)-equivariant (beam-axis symmetry)
    """

    def __init__(
        self,
        irreps: Irreps,
        kernel_size: int = 5,
        lmax: int = 2,
        num_radial_basis: int = 8,
        use_batch_norm: bool = True,
        activation: str = "gated",
        conv_cls=None,
    ):
        super().__init__()
        if conv_cls is None:
            conv_cls = LightEquivariantConv
        self.conv1 = conv_cls(
            irreps, irreps, kernel_size, lmax, num_radial_basis,
            use_batch_norm, activation,
        )
        self.conv2 = conv_cls(
            irreps, irreps, kernel_size, lmax, num_radial_basis,
            use_batch_norm, activation="linear",  # no activation before residual add
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


# ---------------------------------------------------------------------------
# SO(2)-Equivariant Convolution — beam-axis symmetry only
# ---------------------------------------------------------------------------

class EquivariantZConv(nn.Module):
    """
    Equivariant 1D convolution along the z (beam) axis.

    For each irrep block (mul, ir) applies a separate Conv3d with a K×1×1
    kernel, independently per irrep component.  This mixes the `mul`
    multiplicities within each irrep type along z, but never mixes the
    `ir.dim` components within an irrep.

    Why this preserves SO(2) equivariance:
        Under rotation R_z(φ), each component m of an l-type irrep transforms
        as e^{imφ}.  Since this conv applies the same weights to each component
        independently (groups=ir.dim in effect), the e^{imφ} factor passes
        straight through: out[m] = e^{imφ} · in[m] after convolution. ✓
    """

    def __init__(self, irreps: Irreps, kernel_size: int = 5):
        super().__init__()
        self.irreps = irreps
        self.kernel_size = kernel_size
        # One conv per irrep block: mixes multiplicities along z only
        self.z_convs = nn.ModuleList()
        for mul, ir in irreps:
            self.z_convs.append(
                nn.Conv3d(
                    mul, mul,
                    kernel_size=(kernel_size, 1, 1),
                    padding=(kernel_size // 2, 0, 0),
                    bias=False,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        parts = []
        offset = 0
        for i, (mul, ir) in enumerate(self.irreps):
            dim = mul * ir.dim
            block = x[:, offset:offset + dim]           # (B, mul*ir.dim, D, H, W)
            # Treat each of the ir.dim components as an independent batch dim
            # so the conv only mixes multiplicities, never irrep components.
            block = block.reshape(B, mul, ir.dim, D, H, W)
            block = block.permute(0, 2, 1, 3, 4, 5)    # (B, ir.dim, mul, D, H, W)
            block = block.reshape(B * ir.dim, mul, D, H, W)
            block = self.z_convs[i](block)              # (B*ir.dim, mul, D, H, W)
            block = block.reshape(B, ir.dim, mul, D, H, W)
            block = block.permute(0, 2, 1, 3, 4, 5)    # (B, mul, ir.dim, D, H, W)
            block = block.reshape(B, mul * ir.dim, D, H, W)
            parts.append(block)
            offset += dim
        return torch.cat(parts, dim=1)


class SO2EquivariantConv(nn.Module):
    """
    SO(2)-equivariant 3D convolution factored into xy + z components.

    Architecture (two additive pathways):
      1. **xy pathway** (SO(2)-equivariant):
            K×K kernel positions at z=0, i.e. in the H-W plane.
            Spherical harmonics evaluated at z=0 become circular harmonics,
            which transform as e^{imφ} under rotation about the beam axis →
            SO(2)-equivariant by the same argument as the full SO(3) conv.
            Applied as a 2D conv independently to each z-slice.

      2. **z pathway** (beam-axis refinement):
            EquivariantZConv applied to the xy-projected features.
            Learns the axial (z) profile of each feature without imposing
            any equivariance beyond what the irrep structure already gives.

    Combined: out = activation( BN( xy(x) + z_conv(xy(x)) ) )

    NOT equivariant under out-of-plane rotations — correct for cryo-ET data
    with a missing wedge that breaks full SO(3) symmetry.

    The key distinction from the SO(3) model: the SO(3) model sees K³ = 125
    kernel positions including many (Δz≠0) that enforce coupling between z
    and xy directions.  This model only uses K² = 25 xy positions for the
    TP-based equivariant part, then adds an explicit, unconstrained z-axis
    refinement.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        kernel_size: int = 5,
        lmax: int = 2,
        num_radial_basis: int = 8,
        use_batch_norm: bool = True,
        activation: str = "gated",
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # ------------------------------------------------------------------
        # xy kernel: K×K positions at z=0 in the (H, W) = (y, x) plane.
        # In the grid's native (D, H, W) ordering, D-offsets = index 0;
        # setting index-0 = 0 restricts positions to the y-x plane.
        # SH evaluated at (0, Δy, Δx) positions are circular harmonics →
        # SO(2)-equivariant about the beam (D) axis.
        # ------------------------------------------------------------------
        r = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gy, gx = torch.meshgrid(r, r, indexing="ij")       # (K, K) each
        # Shape: (K^2, 3) in (D=0, H=Δy, W=Δx) = (0, Δy, Δx) ordering
        grid_xy = torch.stack([
            torch.zeros(kernel_size ** 2),   # D offset = 0 (beam axis zeroed)
            gy.reshape(-1),                   # H offset
            gx.reshape(-1),                   # W offset
        ], dim=-1)                            # (K^2, 3)

        dist_xy = grid_xy[:, 1:].norm(dim=-1)               # in-plane 2D radius
        unit_xy = grid_xy / (dist_xy.unsqueeze(-1) + 1e-8)  # (K^2, 3)

        sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        sh_xy = o3.spherical_harmonics(sh_irreps, unit_xy, normalize=True)  # (K^2, sh_dim)

        self.register_buffer("sh_xy", sh_xy)       # (K^2, sh_dim)
        self.register_buffer("dist_xy", dist_xy)   # (K^2,)

        # Radial basis: farthest xy corner is at sqrt(2) * (K//2).
        max_r_2d = math.sqrt(2) * (kernel_size // 2) + 0.5
        self.radial_basis_xy = GaussianRadialBasis(num_radial_basis, max_r_2d)
        self.tp_xy = FullyConnectedTensorProduct(
            irreps_in, sh_irreps, irreps_out, shared_weights=False,
        )
        self.radial_mlp_xy = nn.Sequential(
            nn.Linear(num_radial_basis, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp_xy.weight_numel),
        )
        # Normalise radial MLP output variance (same reason as LightEquivariantConv).
        with torch.no_grad():
            self.radial_mlp_xy[-1].weight.div_(math.sqrt(max(self.tp_xy.weight_numel, 1)))
            if self.radial_mlp_xy[-1].bias is not None:
                self.radial_mlp_xy[-1].bias.zero_()

        # ------------------------------------------------------------------
        # z pathway: equivariant 1D conv applied to xy-projected features.
        # Use a larger kernel along z to capture the PSF elongation (~1.5-2x)
        # caused by the missing wedge without constraining out-of-plane symmetry.
        # z_kernel_size = 2*kernel_size - 1 gives 9 for the default kernel_size=5.
        # ------------------------------------------------------------------
        z_kernel_size = max(kernel_size, 2 * kernel_size - 1)
        self.z_conv = EquivariantZConv(irreps_out, kernel_size=z_kernel_size)

        # Batch norm and activation
        # instance=True uses per-sample statistics, which is the correct
        # choice for 3D segmentation at batch_size=2 (DDP SyncBN is fragile
        # at this scale). nnU-Net's default for the same reason.
        self.bn = E3BatchNorm(irreps_out, instance=True) if use_batch_norm else None
        if activation == "gated":
            self.activation = GatedNonlinearity(irreps_out)
        elif activation == "norm":
            self.activation = NormNonlinearity(irreps_out)
        else:
            self.activation = None

    def _build_xy_kernel(self) -> torch.Tensor:
        """
        Build the (C_out, C_in, K, K) SO(2)-equivariant xy kernel.
        Same vectorised TP approach as LightEquivariantConv._build_kernel
        but over K^2 positions instead of K^3.
        """
        K2 = self.kernel_size ** 2
        C_in = self.irreps_in.dim
        C_out = self.irreps_out.dim

        radial_feats = self.radial_basis_xy(self.dist_xy)   # (K^2, num_basis)
        kernel_weights = self.radial_mlp_xy(radial_feats)   # (K^2, weight_numel)

        eye = torch.eye(C_in, device=self.sh_xy.device, dtype=self.sh_xy.dtype)
        eye_batch = eye.unsqueeze(0).expand(K2, -1, -1).contiguous().reshape(K2 * C_in, C_in)
        sh_batch  = self.sh_xy.unsqueeze(1).expand(-1, C_in, -1).contiguous().reshape(K2 * C_in, -1)
        w_batch   = kernel_weights.unsqueeze(1).expand(-1, C_in, -1).contiguous().reshape(K2 * C_in, -1)

        kernel_flat = self.tp_xy(eye_batch, sh_batch, w_batch)           # (K^2*C_in, C_out)
        K = self.kernel_size
        kernel = kernel_flat.reshape(K2, C_in, C_out).permute(0, 2, 1)  # (K^2, C_out, C_in)
        kernel = kernel.reshape(K, K, C_out, C_in).permute(2, 3, 0, 1)  # (C_out, C_in, K, K)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, D, H, W) → (B, C_out, D, H, W)"""
        B, C_in, D, H, W = x.shape

        # xy part: apply 2D equivariant conv independently per z-slice
        xy_kernel = self._build_xy_kernel()                         # (C_out, C_in, K, K)
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * D, C_in, H, W)
        h_flat = F.conv2d(x_flat, xy_kernel, padding=self.padding)  # (B*D, C_out, H, W)
        h = h_flat.reshape(B, D, self.irreps_out.dim, H, W).permute(0, 2, 1, 3, 4)
        # h: (B, C_out, D, H, W)

        # z part: residual equivariant axial refinement
        h = h + self.z_conv(h)

        # Batch norm
        if self.bn is not None:
            B2, C2, Dv, Hv, Wv = h.shape
            h = h.permute(0, 2, 3, 4, 1).reshape(-1, C2)
            h = self.bn(h)
            h = h.reshape(B2, Dv, Hv, Wv, C2).permute(0, 4, 1, 2, 3)

        # Nonlinearity
        if self.activation is not None:
            h = self.activation(h)

        return h


# ---------------------------------------------------------------------------
# SO(3)-Equivariant U-Net
# ---------------------------------------------------------------------------

class EquivariantUNet3D(nn.Module):
    """
    Equivariant 3D U-Net for protein centroid detection.

    Encoder–decoder architecture with skip connections.  Intermediate features
    are typed as SO(3) irreducible representations (scalars + vectors + tensors).
    The final output is C scalar channels (one per protein class), which is
    automatically rotation-invariant by construction.

    The `conv_cls` parameter selects the convolution type:
      - LightEquivariantConv  (default) → full SO(3) equivariance
      - SO2EquivariantConv              → SO(2) equivariance (beam-axis only)
    """

    def __init__(self, cfg: ModelConfig = ModelConfig(), conv_cls=None):
        super().__init__()
        self.cfg = cfg
        if conv_cls is None:
            conv_cls = LightEquivariantConv

        # Parse irreps
        input_irreps = Irreps(f"{cfg.in_channels}x0e")
        encoder_irreps = [Irreps(s) for s in cfg.encoder_irreps]
        decoder_irreps = [Irreps(s) for s in cfg.decoder_irreps]
        output_irreps = Irreps(f"{cfg.num_classes}x0e")

        # ----- Encoder -----
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Input projection: scalar input → first encoder irreps
        self.input_conv = conv_cls(
            input_irreps, encoder_irreps[0],
            kernel_size=cfg.kernel_size, lmax=cfg.lmax,
            num_radial_basis=cfg.num_radial_basis,
            use_batch_norm=cfg.use_batch_norm, activation=cfg.activation,
        )

        for i in range(len(encoder_irreps)):
            if i == 0:
                block = EquivariantResBlock(
                    encoder_irreps[0], cfg.kernel_size, cfg.lmax,
                    cfg.num_radial_basis, cfg.use_batch_norm, cfg.activation,
                    conv_cls=conv_cls,
                )
            else:
                block = nn.Sequential(
                    conv_cls(
                        encoder_irreps[i - 1], encoder_irreps[i],
                        kernel_size=cfg.kernel_size, lmax=cfg.lmax,
                        num_radial_basis=cfg.num_radial_basis,
                        use_batch_norm=cfg.use_batch_norm, activation=cfg.activation,
                    ),
                    EquivariantResBlock(
                        encoder_irreps[i], cfg.kernel_size, cfg.lmax,
                        cfg.num_radial_basis, cfg.use_batch_norm, cfg.activation,
                        conv_cls=conv_cls,
                    ),
                )
            self.enc_blocks.append(block)
            if i < len(encoder_irreps) - 1:
                self.pools.append(EquivariantAvgPool3d(kernel_size=2))

        # ----- Decoder -----
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(len(decoder_irreps)):
            self.ups.append(EquivariantUpsample3d(scale_factor=2))

            enc_level = len(encoder_irreps) - 2 - i
            skip_irreps = encoder_irreps[enc_level]
            from_irreps = encoder_irreps[enc_level + 1] if i == 0 else decoder_irreps[i - 1]
            cat_irreps = from_irreps + skip_irreps

            self.up_convs.append(
                conv_cls(
                    cat_irreps, decoder_irreps[i],
                    kernel_size=cfg.kernel_size, lmax=cfg.lmax,
                    num_radial_basis=cfg.num_radial_basis,
                    use_batch_norm=cfg.use_batch_norm, activation=cfg.activation,
                )
            )
            self.up_blocks.append(
                EquivariantResBlock(
                    decoder_irreps[i], cfg.kernel_size, cfg.lmax,
                    cfg.num_radial_basis, cfg.use_batch_norm, cfg.activation,
                    conv_cls=conv_cls,
                )
            )

        # ----- Output Head (Invariant) -----
        # Final conv to scalar irreps: scalar output is invariant under any
        # rotation (SO(3) or SO(2)) since D(R) = 1 for l=0 irreps.
        final_irreps = decoder_irreps[-1]
        self.output_conv = conv_cls(
            final_irreps, output_irreps,
            kernel_size=3, lmax=cfg.lmax,
            num_radial_basis=cfg.num_radial_basis,
            use_batch_norm=False, activation="none",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 1, D, H, W) input density volume

        Returns:
            heatmaps: (batch, num_classes, D, H, W) per-class Gaussian heatmaps
        """
        # Input projection
        x = self.input_conv(x)

        # Encoder
        skips = []
        for i, enc_block in enumerate(self.enc_blocks):
            x = enc_block(x)
            if i < len(self.pools):
                skips.append(x)
                x = self.pools[i](x)

        # Decoder
        for i in range(len(self.up_blocks)):
            x = self.ups[i](x)

            # Handle spatial size mismatch with skip connection
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)
            x = self.up_convs[i](x)
            x = self.up_blocks[i](x)

        # Output head: project to scalar (invariant) heatmaps
        heatmaps = self.output_conv(x)
        heatmaps = torch.sigmoid(heatmaps)

        return heatmaps


# ---------------------------------------------------------------------------
# Standard 3D U-Net baseline (non-equivariant, for comparison)
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Standard 3D convolution block: Conv → BN → ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Standard 3D U-Net baseline (non-equivariant)."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
    ):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock3D(in_channels, c)
        self.enc2 = ConvBlock3D(c, c * 2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.enc4 = ConvBlock3D(c * 4, c * 8)

        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock3D(c * 2, c)

        self.head = nn.Conv3d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))


# ---------------------------------------------------------------------------
# SO(2)-Equivariant U-Net (convenience alias)
# ---------------------------------------------------------------------------

class SO2EquivariantUNet3D(EquivariantUNet3D):
    """
    SO(2)-Equivariant 3D U-Net — same architecture as EquivariantUNet3D but
    uses SO2EquivariantConv throughout.

    Enforces equivariance only under rotations about the beam (z) axis,
    allowing the network to learn anisotropic responses along z — correctly
    matching cryo-ET data whose SO(3) symmetry is broken by the missing wedge.
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__(cfg, conv_cls=SO2EquivariantConv)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

class SegResNetWrapper(nn.Module):
    """MONAI SegResNet + sigmoid head, matching the equivariant model's
    (B, C, D, H, W) heatmap-in-[0,1] interface.

    This is the Rank-0 integrity-check baseline. The premise is that a
    standard 3D CNN trained with heavy augmentation (heavy_augmentation=True
    in DataConfig) will learn approximate equivariance from data and may
    outperform the hard-coded equivariant model. If it doesn't, equivariance
    is paying its way; if it does, the equivariant variant is research-only.

    Uses InstanceNorm internally (MONAI's default), matching nnU-Net's
    convention for small-batch 3D segmentation.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        try:
            from monai.networks.nets import SegResNet
        except ImportError as exc:
            raise ImportError(
                "MONAI is required for the SegResNet baseline. "
                "Install with: pip install 'monai>=1.3.0'"
            ) from exc

        self.backbone = SegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=in_channels,
            out_channels=num_classes,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            norm="instance",
            use_conv_final=True,
            dropout_prob=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


def build_model(cfg: ModelConfig, use_equivariant: bool = True) -> nn.Module:
    """
    Build the requested model variant.

    Equivariant variants (cfg.model_variant):
      "so3"       — full SO(3)-equivariant U-Net, isotropic targets
      "so3_aniso" — full SO(3)-equivariant U-Net, anisotropic z targets
      "so2"       — SO(2)-equivariant U-Net,     anisotropic z targets
      "so2_iso"   — SO(2)-equivariant U-Net,     isotropic targets

    use_equivariant=False builds a non-equivariant baseline. cfg.baseline_arch
    selects between:
      "segresnet" — MONAI 3D SegResNet (default, recommended for the
                    rank-0 integrity check; matches CZII top-5 architectures)
      "unet3d"    — the original plain 3D U-Net (kept for backwards compat)
    """
    if not use_equivariant:
        arch = getattr(cfg, "baseline_arch", "segresnet")
        if arch == "segresnet":
            return SegResNetWrapper(
                in_channels=cfg.in_channels, num_classes=cfg.num_classes,
            )
        elif arch == "unet3d":
            return UNet3D(in_channels=cfg.in_channels, num_classes=cfg.num_classes)
        else:
            raise ValueError(
                f"Unknown baseline_arch '{arch}'. Choose from: segresnet, unet3d"
            )

    variant = getattr(cfg, "model_variant", "so3")
    if variant in ("so3", "so3_aniso"):
        return EquivariantUNet3D(cfg)
    elif variant in ("so2", "so2_iso"):
        return SO2EquivariantUNet3D(cfg)
    else:
        raise ValueError(
            f"Unknown model_variant '{variant}'. Choose from: "
            "so3, so3_aniso, so2, so2_iso"
        )
