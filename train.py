"""
Training Pipeline for SO(3)/SO(2)-Equivariant Protein Centroid Detection.

Supports three model variants (--model-variant):
  so3       — full SO(3)-equivariant U-Net, isotropic heatmap targets
  so3_aniso — full SO(3)-equivariant U-Net, anisotropic z heatmap targets
  so2       — SO(2)-equivariant U-Net,      anisotropic z heatmap targets

Multi-GPU (2 GPUs):
  torchrun --nproc_per_node=2 train.py --model-variant so3
  torchrun --nproc_per_node=2 train.py --model-variant so3_aniso
  torchrun --nproc_per_node=2 train.py --model-variant so2

Single-GPU:
  python train.py --model-variant so3

Checkpoints are saved to ./checkpoints/{variant}/ so all three can be
trained without clobbering each other.  Use compare_variants.py afterwards
to compare val F-beta across variants.
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import Config, TrainConfig
from dataset import build_dataloaders
from model import build_model
from losses import build_loss
from utils import (
    extract_centroids,
    compute_weighted_fbeta,
    nms_3d,
    CLASS_NAMES,
    PROTEIN_CLASSES,
)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple:
    """
    Initialise torch.distributed from the environment variables set by torchrun.

    Returns (is_ddp, rank, local_rank, world_size).
    When not launched with torchrun all four return (False, 0, 0, 1).
    """
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean_scalar(value: float, world_size: int) -> float:
    """Average a Python float across all DDP ranks."""
    if world_size == 1:
        return value
    t = torch.tensor(value, dtype=torch.float32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int, rank: int = 0):
    """Set random seed, offset by rank so each GPU sees different data order."""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay skipped for bias / norm parameters."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW([
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.learning_rate)


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup + cosine decay."""
    total_steps  = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler,
    scaler, device, use_amp, epoch, rank, world_size,
    writer=None,
) -> dict:
    model.train()
    total_loss   = 0.0
    total_gnorm  = 0.0
    n_batches    = 0
    global_step  = epoch * len(loader)

    for batch_idx, (volumes, targets) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            preds = model(volumes)
            loss  = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss  += loss.item()
        total_gnorm += float(gnorm)
        n_batches   += 1

        step = global_step + batch_idx
        if is_main(rank) and writer is not None:
            writer.add_scalar("train/grad_norm", float(gnorm), step)

        if is_main(rank) and batch_idx % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.5f} | GradNorm: {float(gnorm):.3f} | LR: {lr:.2e}"
            )

    avg_loss  = reduce_mean_scalar(total_loss  / max(n_batches, 1), world_size)
    avg_gnorm = reduce_mean_scalar(total_gnorm / max(n_batches, 1), world_size)
    return {"train_loss": avg_loss, "train_grad_norm": avg_gnorm}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model, loader, criterion, device, use_amp,
    score_thresholds, world_size,
    nms_kernel_sizes=None,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    all_pred = {cls: [] for cls in CLASS_NAMES}
    all_gt   = {cls: [] for cls in CLASS_NAMES}

    for volumes, targets in loader:
        volumes = volumes.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            preds = model(volumes)
            loss  = criterion(preds, targets)

        total_loss += loss.item()
        n_batches  += 1

        for b in range(preds.shape[0]):
            pred_np   = preds[b].cpu().numpy()
            target_np = targets[b].cpu().numpy()
            pred_dets = extract_centroids(
                pred_np, score_thresholds,
                nms_kernel_sizes=nms_kernel_sizes,
            )
            gt_dets = extract_centroids(
                target_np, {c: 0.5 for c in CLASS_NAMES},
                nms_kernel_sizes=nms_kernel_sizes,
                subvoxel=False,  # GT peaks are already integer-placed
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
    fbeta    = compute_weighted_fbeta(merged_pred, merged_gt)
    val_loss = reduce_mean_scalar(total_loss / max(n_batches, 1), world_size)

    return {"val_loss": val_loss, "val_fbeta": fbeta}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Unwrap DDP before saving so checkpoints load on any number of GPUs
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":    scaler.state_dict() if scaler else None,
        "metrics":             metrics,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    # Handle both plain and DDP-wrapped models
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train equivariant protein detector")
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--baseline",      action="store_true",
                        help="Use standard (non-equivariant) U-Net baseline")
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        choices=["so3", "so3_aniso", "so2"],
        help=(
            "so3: SO(3)-equivariant, isotropic targets (default). "
            "so3_aniso: SO(3)-equivariant, anisotropic z targets. "
            "so2: SO(2)-equivariant, anisotropic z targets."
        ),
    )
    args = parser.parse_args()

    # DDP setup — must happen before any CUDA calls
    is_ddp, rank, local_rank, world_size = setup_ddp()

    # Build config
    cfg = Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.learning_rate = args.lr

    # Apply model variant — also controls z_sigma_scale for anisotropic targets
    variant = args.model_variant or ("baseline" if args.baseline else "so3")
    if not args.baseline:
        cfg.model.model_variant = variant
    if variant in ("so3_aniso", "so2"):
        cfg.data.z_sigma_scale = 1.5   # anisotropic heatmap targets
    else:
        cfg.data.z_sigma_scale = 1.0   # isotropic heatmap targets

    train_cfg = cfg.train
    set_seed(train_cfg.seed, rank)

    # Checkpoint and log dirs are variant-specific so all three can coexist
    ckpt_base  = Path(train_cfg.checkpoint_dir) / variant
    log_base   = Path(train_cfg.log_dir) / variant
    cfg.train.checkpoint_dir = str(ckpt_base)
    cfg.train.log_dir        = str(log_base)

    # Device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main(rank):
        print(f"Variant: {variant} | DDP: {is_ddp} | World size: {world_size} | Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU[{local_rank}]: {torch.cuda.get_device_name(local_rank)}")

    # Data
    if is_main(rank):
        print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg, rank=rank, world_size=world_size)
    if is_main(rank):
        print(f"Train batches: {len(train_loader)}, "
              f"Val batches: {len(val_loader) if val_loader else 0}")

    # Model
    use_equivariant = not args.baseline
    if is_main(rank):
        print(f"Building model variant: {variant}...")
    model = build_model(cfg.model, use_equivariant=use_equivariant)
    model = model.to(device)

    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    # Loss, optimizer, scheduler, scaler
    criterion = build_loss(train_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, len(train_loader))
    scaler    = GradScaler("cuda", enabled=train_cfg.use_amp)

    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    # TensorBoard — only on rank 0
    writer = SummaryWriter(log_dir=str(log_base)) if is_main(rank) else None

    score_thresholds  = cfg.inference.score_thresholds
    nms_kernel_sizes  = cfg.inference.nms_kernel_sizes
    best_fbeta        = 0.0

    if is_main(rank):
        print(f"\nStarting training for {train_cfg.epochs} epochs...")
        print("=" * 70)

    for epoch in range(start_epoch, train_cfg.epochs):
        t0 = time.time()

        # DDP sampler must be told the epoch for correct shuffling
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, train_cfg.use_amp, epoch, rank, world_size,
            writer=writer if is_main(rank) else None,
        )

        if is_main(rank) and writer:
            writer.add_scalar("train/loss",      train_metrics["train_loss"],      epoch)
            writer.add_scalar("train/grad_norm", train_metrics["train_grad_norm"], epoch)
            writer.add_scalar("train/lr",        optimizer.param_groups[0]["lr"],  epoch)

        # Validate (only main process; val_loader has no sampler)
        val_metrics = {}
        if val_loader and (epoch + 1) % train_cfg.val_every == 0:
            if is_main(rank):
                val_metrics = validate(
                    model, val_loader, criterion, device,
                    train_cfg.use_amp, score_thresholds,
                    world_size=1,
                    nms_kernel_sizes=nms_kernel_sizes,
                )
                if writer:
                    writer.add_scalar("val/loss",  val_metrics["val_loss"],  epoch)
                    writer.add_scalar("val/fbeta", val_metrics["val_fbeta"], epoch)

                if val_metrics["val_fbeta"] > best_fbeta:
                    best_fbeta = val_metrics["val_fbeta"]
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch,
                        {**train_metrics, **val_metrics},
                        os.path.join(str(ckpt_base), "best.pt"),
                    )

        # Periodic checkpoint — rank 0 only
        if is_main(rank) and (epoch + 1) % train_cfg.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {**train_metrics, **val_metrics},
                os.path.join(str(ckpt_base), f"epoch_{epoch:04d}.pt"),
            )

        if is_main(rank):
            elapsed = time.time() - t0
            msg = (
                f"Epoch {epoch:3d}/{train_cfg.epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.5f}"
            )
            if val_metrics:
                msg += (f" | Val Loss: {val_metrics['val_loss']:.5f}"
                        f" | F-beta: {val_metrics['val_fbeta']:.4f}")
            msg += f" | {elapsed:.1f}s"
            print(msg)

    # Final checkpoint — rank 0 only
    if is_main(rank):
        save_checkpoint(
            model, optimizer, scheduler, scaler, train_cfg.epochs - 1,
            train_metrics,
            os.path.join(str(ckpt_base), "final.pt"),
        )
        if writer:
            writer.close()
        print(f"\nTraining complete. Best F-beta: {best_fbeta:.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
