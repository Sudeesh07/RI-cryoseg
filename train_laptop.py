"""
Laptop-scale training script for the SO(3)/SO(2) equivariant model.

Designed for GPUs with 4–8 GB VRAM (e.g. RTX 4050 6 GB).

Key reductions vs the full train.py:
  - "tiny" irrep config: ~850 K params vs ~10 M in full model
  - kernel_size 3 instead of 5  (3^3=27 vs 5^3=125 kernel positions)
  - patch_size (64, 64, 64) instead of (96, 96, 96)
  - batch_size 2
  - AMP always on
  - 3-level encoder/decoder instead of 4

Purpose: verify the full training loop works end-to-end, debug data
         loading, spot-check F-beta behaviour, and do quick ablations.
         NOT a replacement for the full-scale training run.

Usage:
    python train_laptop.py --model-variant so3 --epochs 20
    python train_laptop.py --model-variant so2 --epochs 20

After training both variants you can compare them on the validation set:
    python compare_variants.py \\
        --so3 checkpoints_laptop/so3/best.pt \\
        --so2 checkpoints_laptop/so2/best.pt
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
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import Config, ModelConfig, TrainConfig, DataConfig
from dataset import build_dataloaders
from model import build_model
from losses import build_loss
from utils import (
    extract_centroids,
    compute_weighted_fbeta,
    CLASS_NAMES,
)

# ---------------------------------------------------------------------------
# Tiny model config that fits in 6 GB VRAM with patch 64^3, batch 2
# ---------------------------------------------------------------------------

TINY_ENCODER_IRREPS = [
    "8x0e + 4x1o + 2x2e",    # level 0 — full resolution
    "16x0e + 8x1o + 4x2e",   # level 1
    "24x0e + 12x1o + 6x2e",  # level 2 — bottleneck
]
TINY_DECODER_IRREPS = [
    "16x0e + 8x1o + 4x2e",   # level 1
    "8x0e + 4x1o + 2x2e",    # level 0
]


def make_laptop_config(variant: str) -> Config:
    """Return a Config pre-tuned for laptop-scale training."""
    cfg = Config()

    # Tiny model
    cfg.model.encoder_irreps = TINY_ENCODER_IRREPS
    cfg.model.decoder_irreps = TINY_DECODER_IRREPS
    cfg.model.kernel_size = 3       # 3^3=27 positions vs 5^3=125
    cfg.model.lmax = 2
    cfg.model.model_variant = variant

    # Smaller patches — fits 6 GB VRAM with batch=2 and AMP
    cfg.data.patch_size = (64, 64, 64)
    cfg.data.patches_per_tomo = 32  # fewer patches → faster epoch
    cfg.data.z_sigma_scale = 1.5 if variant in ("so3_aniso", "so2") else 1.0

    # Training
    cfg.train.epochs = 50
    cfg.train.batch_size = 2
    cfg.train.learning_rate = 3e-4
    cfg.train.warmup_epochs = 3
    cfg.train.use_amp = True
    cfg.train.val_every = 5
    cfg.train.save_every = 10
    cfg.train.num_workers = 2        # laptop: keep low to avoid RAM pressure

    # Variant-specific checkpoint/log dirs
    cfg.train.checkpoint_dir = f"./checkpoints_laptop/{variant}"
    cfg.train.log_dir = f"./logs_laptop/{variant}"

    return cfg


# ---------------------------------------------------------------------------
# Training / validation — minimal versions from train.py
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, use_amp, epoch, writer=None) -> dict:
    model.train()
    total_loss = 0.0
    total_gnorm = 0.0
    n = 0
    global_step = epoch * len(loader)

    for i, (vols, targets) in enumerate(loader):
        vols    = vols.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            preds = model(vols)
            loss  = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        total_loss  += loss.item()
        total_gnorm += float(gnorm)
        n += 1

        if writer:
            writer.add_scalar("train/grad_norm", float(gnorm), global_step + i)

        if i % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Ep{epoch} [{i}/{len(loader)}] loss={loss.item():.4f}"
                  f"  gnorm={float(gnorm):.3f}  lr={lr:.2e}")

    return {"train_loss": total_loss / max(n, 1),
            "train_grad_norm": total_gnorm / max(n, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp,
             score_thresholds, nms_kernel_sizes) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred = {c: [] for c in CLASS_NAMES}
    all_gt   = {c: [] for c in CLASS_NAMES}

    for vols, targets in loader:
        vols    = vols.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            preds = model(vols)
            loss  = criterion(preds, targets)

        total_loss += loss.item()
        n += 1

        for b in range(preds.shape[0]):
            pred_np = preds[b].cpu().numpy()
            tgt_np  = targets[b].cpu().numpy()
            pd = extract_centroids(pred_np, score_thresholds,
                                   nms_kernel_sizes=nms_kernel_sizes)
            gd = extract_centroids(tgt_np, {c: 0.5 for c in CLASS_NAMES},
                                   nms_kernel_sizes=nms_kernel_sizes,
                                   subvoxel=False)
            for c in CLASS_NAMES:
                if len(pd[c]) > 0:
                    all_pred[c].append(pd[c][:, :3])
                if len(gd[c]) > 0:
                    all_gt[c].append(gd[c][:, :3])

    merged_pred = {c: np.concatenate(all_pred[c]) if all_pred[c] else np.empty((0, 3))
                   for c in CLASS_NAMES}
    merged_gt   = {c: np.concatenate(all_gt[c])   if all_gt[c]  else np.empty((0, 3))
                   for c in CLASS_NAMES}

    return {
        "val_loss":  total_loss / max(n, 1),
        "val_fbeta": compute_weighted_fbeta(merged_pred, merged_gt),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":    scaler.state_dict(),
        "metrics":             metrics,
    }, path)
    print(f"  Checkpoint: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Laptop-scale training")
    parser.add_argument("--model-variant", default="so3",
                        choices=["so3", "so3_aniso", "so2"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    cfg = make_laptop_config(args.model_variant)
    if args.epochs:
        cfg.train.epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM={props.total_memory//1024**2} MB")

    # ---- Data ----
    print("Loading data...")
    train_loader, val_loader = build_dataloaders(cfg, rank=0, world_size=1)
    print(f"Train batches: {len(train_loader)}"
          f"  Val batches: {len(val_loader) if val_loader else 0}")

    # ---- Model ----
    print(f"Building {args.model_variant} model (tiny config)...")
    model = build_model(cfg.model, use_equivariant=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # ---- Optimizer / scheduler / scaler ----
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.train.learning_rate,
                                  weight_decay=1e-5)
    total_steps  = cfg.train.epochs * len(train_loader)
    warmup_steps = cfg.train.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler("cuda", enabled=cfg.train.use_amp)
    criterion = build_loss(cfg.train).to(device)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    ckpt_dir = cfg.train.checkpoint_dir
    writer   = SummaryWriter(log_dir=cfg.train.log_dir)
    score_thresholds = cfg.inference.score_thresholds
    nms_kernel_sizes = cfg.inference.nms_kernel_sizes
    best_fbeta = 0.0

    print(f"\nStarting training ({cfg.train.epochs} epochs, "
          f"patch={cfg.data.patch_size}, batch={cfg.train.batch_size})...")
    print("=" * 60)

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, cfg.train.use_amp, epoch, writer=writer,
        )
        writer.add_scalar("train/loss",      train_metrics["train_loss"],      epoch)
        writer.add_scalar("train/grad_norm", train_metrics["train_grad_norm"], epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        val_metrics = {}
        if val_loader and (epoch + 1) % cfg.train.val_every == 0:
            val_metrics = validate(
                model, val_loader, criterion, device,
                cfg.train.use_amp, score_thresholds, nms_kernel_sizes,
            )
            writer.add_scalar("val/loss",  val_metrics["val_loss"],  epoch)
            writer.add_scalar("val/fbeta", val_metrics["val_fbeta"], epoch)

            if val_metrics["val_fbeta"] > best_fbeta:
                best_fbeta = val_metrics["val_fbeta"]
                save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                                {**train_metrics, **val_metrics},
                                os.path.join(ckpt_dir, "best.pt"))

        if (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            {**train_metrics, **val_metrics},
                            os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))

        elapsed = time.time() - t0
        msg = (f"Epoch {epoch:3d}/{cfg.train.epochs}"
               f"  loss={train_metrics['train_loss']:.5f}")
        if val_metrics:
            msg += (f"  val_loss={val_metrics['val_loss']:.5f}"
                    f"  F-beta={val_metrics['val_fbeta']:.4f}")
        msg += f"  {elapsed:.1f}s"
        print(msg)

    save_checkpoint(model, optimizer, scheduler, scaler, cfg.train.epochs - 1,
                    train_metrics,
                    os.path.join(ckpt_dir, "final.pt"))
    writer.close()
    print(f"\nDone. Best F-beta: {best_fbeta:.4f}")


if __name__ == "__main__":
    main()
