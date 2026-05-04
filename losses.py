"""
Loss Functions for Gaussian Heatmap-Based Centroid Detection.

Three loss strategies:
  1. Weighted MSE  — upweights positive (near-centroid) voxels
  2. Focal Loss    — adapted from CenterNet; handles extreme class imbalance
  3. Combined      — weighted sum of both
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import TrainConfig, CLASS_WEIGHTS, NUM_CLASSES


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
    """Build the loss function from config."""
    cw = CLASS_WEIGHTS if cfg.loss_class_weights else None

    if cfg.loss_type == "wmse":
        return WeightedMSELoss(alpha=cfg.wmse_alpha, class_weights=cw)
    elif cfg.loss_type == "focal":
        return FocalHeatmapLoss(
            alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, class_weights=cw,
        )
    elif cfg.loss_type == "combined":
        return CombinedLoss(
            wmse_alpha=cfg.wmse_alpha,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            class_weights=cw,
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss_type}")
