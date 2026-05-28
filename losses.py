"""
Loss Functions for Gaussian Heatmap-Based Centroid Detection.

Three loss strategies:
  1. Weighted MSE  — upweights positive (near-centroid) voxels
  2. Focal Loss    — adapted from CenterNet; handles extreme class imbalance
  3. Combined      — weighted sum of both
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import (
    TrainConfig, CLASS_WEIGHTS, CLASS_SIGMAS_VOXELS, NUM_CLASSES,
)


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error for Gaussian heatmap regression.

    Loss = mean( (pred - target)^2 * (target + alpha) )

    The (target + alpha) weighting ensures that:
      - Voxels near centroids (target ≈ 1) get full weight
      - Background voxels (target ≈ 0) get reduced weight alpha
      - This combats the extreme foreground/background imbalance
    """

    def __init__(self, alpha: float = 0.1, class_weights: Optional[list] = None):
        super().__init__()
        self.alpha = alpha
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, C, D, H, W) predicted heatmaps in [0, 1]
        target: (B, C, D, H, W) Gaussian heatmap targets in [0, 1]
        """
        weight = target + self.alpha
        mse = (pred - target) ** 2 * weight

        if self.class_weights is not None:
            # Apply per-class weights: (C,) → (1, C, 1, 1, 1)
            cw = self.class_weights.reshape(1, -1, 1, 1, 1)
            mse = mse * cw

        return mse.mean()


class FocalHeatmapLoss(nn.Module):
    """
    Focal loss for heatmap regression, adapted from CornerNet/CenterNet.

    For positive locations (target = 1):
        loss = -(1 - pred)^alpha * log(pred)

    For negative locations (target < 1):
        loss = -(1 - target)^beta * pred^alpha * log(1 - pred)

    This heavily downweights easy negatives (background voxels far from any
    centroid) and focuses learning on hard examples near the boundary.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 4.0,
        class_weights: Optional[list] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, C, D, H, W) in [0, 1]
        target: (B, C, D, H, W) in [0, 1]
        """
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)

        # Positive locations: target close to 1
        pos_mask = target.ge(0.99)
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred)

        # Negative locations: target < 1
        neg_loss = (
            -((1 - target) ** self.gamma)
            * (pred ** self.alpha)
            * torch.log(1 - pred)
        )

        loss = torch.where(pos_mask, pos_loss, neg_loss)

        if self.class_weights is not None:
            cw = self.class_weights.reshape(1, -1, 1, 1, 1)
            loss = loss * cw

        # Normalise by total voxel count so the loss stays in [0, ~2] regardless
        # of how many positive voxels exist.  The focal weights still push the
        # optimiser to focus on hard examples; the absolute scale is just fixed.
        # Normalising by num_pos alone makes the loss scale with patch fill rate
        # (~50 positives / 884K voxels = ×17K amplification) which overflows
        # float16 during AMP backward and stalls GradScaler indefinitely.
        return loss.mean()


class LearnableZBlurLoss(nn.Module):
    """
    Wraps any heatmap loss with a differentiable per-class z-direction
    Gaussian blur applied to the (xy-only-blurred) targets.

    Motivation
    ----------
    The missing-wedge PSF elongates particles along z, but the exact
    elongation depends on the particle (size, ice thickness, defocus, ...).
    Hard-coding z_sigma_scale = 1.5 for every class is a guess.  This wrapper
    treats sigma_z / sigma_xy as a learnable per-class scalar, parameterised
    in log-space and clamped to a safe range.

    Pipeline
    --------
    1. Dataloader emits xy-only-blurred targets (target_mode == "xy_blur").
       Each centroid is a 2D Gaussian along (y, x) at a single z-slice.
    2. This module convolves the targets along z with a Gaussian of
       sigma_z = sigma_xy[c] * exp(log_z_scale[c]) for each class c.
    3. The wrapped loss then compares the predictions to the z-blurred targets.

    The z-blur is implemented as a depthwise 1D conv whose kernel is rebuilt
    each forward pass from the current log_z_scale, so gradients flow into the
    learnable parameter.  The 2D xy peak is normalised to 1 in the dataloader,
    so re-normalising after the z-blur keeps an isolated centroid at peak 1.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        sigma_xy_per_class: Optional[list] = None,
        init_z_scale: float = 1.5,
        z_scale_min: float = 0.5,
        z_scale_max: float = 4.0,
        truncate: float = 3.0,
    ):
        super().__init__()
        self.base_loss = base_loss

        if sigma_xy_per_class is None:
            sigma_xy_per_class = CLASS_SIGMAS_VOXELS
        self.register_buffer(
            "sigma_xy",
            torch.tensor(sigma_xy_per_class, dtype=torch.float32),
        )
        # Parameterise in log-space so sigma_z stays positive.
        self.log_z_scale = nn.Parameter(
            torch.full(
                (len(sigma_xy_per_class),),
                math.log(init_z_scale),
                dtype=torch.float32,
            )
        )
        self.z_scale_min = float(z_scale_min)
        self.z_scale_max = float(z_scale_max)
        self.truncate = float(truncate)
        # Kernel half-width sized for the largest reachable sigma_z at the
        # upper clamp — guarantees the kernel never needs to grow at runtime.
        max_sigma_z = float(max(sigma_xy_per_class)) * self.z_scale_max
        self.kernel_radius = max(1, int(math.ceil(self.truncate * max_sigma_z)))

    def current_z_scales(self) -> torch.Tensor:
        """Current per-class sigma_z / sigma_xy ratio, clamped to bounds."""
        return torch.exp(self.log_z_scale).clamp(self.z_scale_min, self.z_scale_max)

    def _build_z_kernel(self, device, dtype) -> torch.Tensor:
        """
        Build a (C, 1, K, 1, 1) depthwise z-Gaussian kernel from current sigmas.
        Each channel's kernel is normalised so a unit impulse stays at peak 1
        after convolution (peak preservation).
        """
        z_scales = self.current_z_scales().to(device=device, dtype=dtype)
        sigma_xy = self.sigma_xy.to(device=device, dtype=dtype)
        sigma_z = sigma_xy * z_scales                       # (C,)

        K = 2 * self.kernel_radius + 1
        offsets = torch.arange(K, device=device, dtype=dtype) - self.kernel_radius  # (K,)
        # (C, K): exp(-d^2 / (2 sigma_z^2))
        kernel = torch.exp(-(offsets[None, :] ** 2) / (2 * sigma_z[:, None] ** 2))
        # Normalise so the centre tap = 1 / sum(kernel) of mass distribution.
        # We want a unit impulse to stay at peak 1, so divide by max.
        kernel = kernel / kernel.max(dim=-1, keepdim=True).values
        # Depthwise conv needs (C_out, C_in/groups, K, 1, 1) = (C, 1, K, 1, 1).
        return kernel.view(-1, 1, K, 1, 1)

    def z_blur_target(self, target: torch.Tensor) -> torch.Tensor:
        """Apply per-class z-Gaussian blur to a (B, C, D, H, W) target."""
        kernel = self._build_z_kernel(device=target.device, dtype=target.dtype)
        # Depthwise conv: groups == C
        blurred = F.conv3d(
            target,
            kernel,
            bias=None,
            stride=1,
            padding=(self.kernel_radius, 0, 0),
            groups=target.shape[1],
        )
        return blurred.clamp(0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, C, D, H, W) model predictions in [0, 1]
        target: (B, C, D, H, W) xy-only-blurred targets in [0, 1]
        """
        target_z = self.z_blur_target(target)
        return self.base_loss(pred, target_z)

    def extra_repr(self) -> str:
        return (f"sigma_xy={self.sigma_xy.tolist()}, "
                f"z_scale_init=exp(log_z_scale), "
                f"clamp=[{self.z_scale_min}, {self.z_scale_max}], "
                f"kernel_radius={self.kernel_radius}")


class CombinedLoss(nn.Module):
    """Weighted combination of Weighted MSE and Focal Heatmap Loss."""

    def __init__(
        self,
        wmse_weight: float = 1.0,
        focal_weight: float = 1.0,
        wmse_alpha: float = 0.1,
        focal_alpha: float = 2.0,
        focal_gamma: float = 4.0,
        class_weights: Optional[list] = None,
    ):
        super().__init__()
        self.wmse_weight = wmse_weight
        self.focal_weight = focal_weight
        self.wmse = WeightedMSELoss(alpha=wmse_alpha, class_weights=class_weights)
        self.focal = FocalHeatmapLoss(
            alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_wmse = self.wmse(pred, target)
        l_focal = self.focal(pred, target)
        return self.wmse_weight * l_wmse + self.focal_weight * l_focal


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(cfg: TrainConfig) -> nn.Module:
    """Build the loss function from config.

    If cfg.learnable_z_sigma is True, the base loss is wrapped in
    LearnableZBlurLoss, which expects xy-only-blurred targets from the
    dataloader (DataConfig.target_mode = "xy_blur") and applies a
    differentiable per-class z-Gaussian blur with a learnable sigma_z.
    """
    cw = CLASS_WEIGHTS if cfg.loss_class_weights else None

    if cfg.loss_type == "wmse":
        base = WeightedMSELoss(alpha=cfg.wmse_alpha, class_weights=cw)
    elif cfg.loss_type == "focal":
        base = FocalHeatmapLoss(
            alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=cw,
        )
    elif cfg.loss_type == "combined":
        base = CombinedLoss(
            wmse_alpha=cfg.wmse_alpha,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            class_weights=cw,
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss_type}")

    if getattr(cfg, "learnable_z_sigma", False):
        base = LearnableZBlurLoss(
            base_loss=base,
            init_z_scale=cfg.learnable_z_sigma_init,
            z_scale_min=cfg.learnable_z_sigma_min,
            z_scale_max=cfg.learnable_z_sigma_max,
        )
    return base
