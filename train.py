"""
Training Pipeline for SO(3)/SO(2)-Equivariant Protein Centroid Detection.

Supports four equivariant variants — a 2x2 ablation of architecture
(so3 / so2) x targets (iso / aniso):
  so3       — SO(3)-equivariant U-Net, isotropic heatmap targets
  so3_aniso — SO(3)-equivariant U-Net, anisotropic z heatmap targets
  so2       — SO(2)-equivariant U-Net, anisotropic z heatmap targets
  so2_iso   — SO(2)-equivariant U-Net, isotropic heatmap targets

Plus a non-equivariant baseline (--baseline) — the Rank-0 integrity check:
  --baseline                              # default arch: segresnet
  --baseline --baseline-arch segresnet    # MONAI 3D SegResNet (recommended)
  --baseline --baseline-arch unet3d       # plain 3D U-Net

Recommended SegResNet-baseline command (matches the CZII Kaggle setup that
beat equivariant networks empirically):
  torchrun --nproc_per_node=2 train.py --baseline --heavy-aug \\
      --tomo-dirs 100  # add more comma-separated dirs if available

Other useful flags:
  --learnable-z-sigma   per-class learnable z-Gaussian in the loss
  --heavy-aug           MONAI-style augmentation (gamma/blur/low-res/...)
  --no-ema              disable EMA shadow model (default: enabled)
  --tomo-dirs DIRS      comma-separated tomogram subdirs to stack as channels

Multi-GPU (2 GPUs):
  torchrun --nproc_per_node=2 train.py --model-variant so2
  torchrun --nproc_per_node=2 train.py --baseline --heavy-aug

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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter

from config import Config, TrainConfig
from dataset import (
    build_dataloaders,
    load_tomogram_stack,
    load_run_annotations, discover_run_names,
    CZIIInferenceDataset,
)
from torch.utils.data import DataLoader
from model import build_model
from losses import build_loss, LearnableZBlurLoss
from utils import (
    aggregate_predictions,
    extract_centroids,
    compute_fbeta_per_class,
    compute_weighted_fbeta,
    nms_3d,
    CLASS_NAMES,
    NUM_CLASSES,
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

def build_ema_model(model: nn.Module, decay: float = 0.999) -> AveragedModel:
    """Create an EMA (exponential moving average) shadow of `model`.

    theta_ema = decay * theta_ema + (1 - decay) * theta, applied after every
    optimizer step. The EMA weights typically generalise 0.5-1.5% better in
    F-beta than the raw weights, and were used by the 4th and 13th-place
    CZII Kaggle solutions. AveragedModel handles DDP-unwrapping internally.
    """
    inner = model.module if isinstance(model, DDP) else model

    def _ema_fn(avg_param, param, num_averaged):
        return decay * avg_param + (1.0 - decay) * param

    ema = AveragedModel(inner, avg_fn=_ema_fn)
    return ema


def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
    extra_modules: Optional[list] = None,
) -> torch.optim.Optimizer:
    """AdamW with weight decay skipped for bias / norm parameters.

    extra_modules: optional list of nn.Modules whose parameters should also
    be optimised (e.g. a loss with learnable per-class sigma_z).  These
    parameters get weight_decay=0 because they are interpretable scalars,
    not feature weights.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    extra_params: list = []
    if extra_modules:
        for m in extra_modules:
            for p in m.parameters():
                if p.requires_grad:
                    extra_params.append(p)

    return torch.optim.AdamW([
        {"params": decay,        "weight_decay": cfg.weight_decay},
        {"params": no_decay,     "weight_decay": 0.0},
        {"params": extra_params, "weight_decay": 0.0},
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
    writer=None, ema_model=None,
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

        # EMA update happens after the optimizer step so the shadow tracks
        # the post-step parameters. AveragedModel takes the underlying module,
        # but it was constructed from `model.module` if DDP — passing the live
        # DDP-wrapped model would walk a different parameter tree, so unwrap.
        if ema_model is not None:
            ema_model.update_parameters(
                model.module if isinstance(model, DDP) else model
            )

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
# Full-volume validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_full_volume(
    model,
    dataset_dir: str,
    run_names,
    data_cfg,
    inference_cfg,
    device,
    use_amp: bool,
    batch_size: int = 4,
    overlap: float = 0.5,
) -> dict:
    """
    Run sliding-window inference on each full validation tomogram and compute
    F-beta on the aggregated centroids.

    This matches deployment behaviour (no positive-patch bias, no patch-cut
    artefacts), unlike the patch-level validate() function whose patches are
    drawn uniformly at random from a distribution that does not match training.

    Returns: {
        "full_val_fbeta":       weighted overall F-beta across all val runs,
        "full_val_fbeta_per_class": dict[class -> F-beta],
        "full_val_runs":        list of run names actually scored,
    }
    """
    model.eval()

    inner_model = model.module if isinstance(model, DDP) else model

    score_thresholds = inference_cfg.score_thresholds
    nms_kernel_sizes = inference_cfg.nms_kernel_sizes
    fbeta_beta = inference_cfg.fbeta_beta
    distance_scale = inference_cfg.distance_threshold_scale

    merged_pred = {c: [] for c in CLASS_NAMES}
    merged_gt   = {c: [] for c in CLASS_NAMES}
    actual_runs = []

    for run_name in run_names:
        try:
            # Load all configured tomogram channels (denoised / wbp / etc.)
            volume = load_tomogram_stack(
                dataset_dir=dataset_dir,
                run_name=run_name,
                voxel_spacing_dir=data_cfg.voxel_spacing_dir,
                tomo_dirs=getattr(data_cfg, "tomo_dirs", ("100",)),
                normalize=data_cfg.normalize,
                clip_percentile=data_cfg.clip_percentile,
            )
        except FileNotFoundError:
            continue

        ds = CZIIInferenceDataset(
            volume,
            patch_size=inference_cfg.patch_size,
            overlap=overlap,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )

        patch_preds = []
        for patches, origins in loader:
            patches = patches.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                preds = inner_model(patches)
            preds = preds.float().cpu()
            for i in range(preds.shape[0]):
                origin = tuple(origins[i].numpy().tolist())
                patch_preds.append((preds[i].numpy(), origin))

        full_heatmap = aggregate_predictions(
            patch_preds, volume_shape=volume.shape, num_classes=NUM_CLASSES,
        )

        pred_dets = extract_centroids(
            full_heatmap, score_thresholds, nms_kernel_sizes=nms_kernel_sizes,
            use_soft_nms=getattr(inference_cfg, "use_soft_nms", False),
            soft_nms_sigma_factor=getattr(inference_cfg, "soft_nms_sigma_factor", 0.7),
        )
        gt_dict = load_run_annotations(
            os.path.join(dataset_dir, run_name), data_cfg.voxel_spacing_dir,
        )

        for c in CLASS_NAMES:
            if len(pred_dets[c]) > 0:
                merged_pred[c].append(pred_dets[c][:, :3])
            if c in gt_dict and len(gt_dict[c]) > 0:
                merged_gt[c].append(gt_dict[c])
        actual_runs.append(run_name)

    merged_pred = {
        c: np.concatenate(merged_pred[c], axis=0) if merged_pred[c] else np.empty((0, 3))
        for c in CLASS_NAMES
    }
    merged_gt = {
        c: np.concatenate(merged_gt[c], axis=0) if merged_gt[c] else np.empty((0, 3))
        for c in CLASS_NAMES
    }

    per_class = compute_fbeta_per_class(
        merged_pred, merged_gt, beta=fbeta_beta, distance_scale=distance_scale,
    )
    overall = compute_weighted_fbeta(
        merged_pred, merged_gt, beta=fbeta_beta, distance_scale=distance_scale,
    )

    return {
        "full_val_fbeta":           overall,
        "full_val_fbeta_per_class": per_class,
        "full_val_runs":            actual_runs,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch, metrics, path,
    ema_model=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Unwrap DDP before saving so checkpoints load on any number of GPUs
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    ema_state = None
    if ema_model is not None:
        # AveragedModel stores the shadow as .module
        ema_state = ema_model.module.state_dict()
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    state_dict,
        "ema_state_dict":      ema_state,
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
        "--learnable-z-sigma", action="store_true",
        help=(
            "Use per-class learnable sigma_z applied inside the loss "
            "(targets are xy-only-blurred). Overrides z_sigma_scale."
        ),
    )
    parser.add_argument(
        "--no-ema", action="store_true",
        help="Disable EMA shadow model (default: enabled)",
    )
    parser.add_argument(
        "--heavy-aug", action="store_true",
        help="Enable MONAI-style heavy augmentation (gamma/blur/low-res/...)",
    )
    parser.add_argument(
        "--baseline-arch", type=str, default=None,
        choices=["segresnet", "unet3d"],
        help="When --baseline: which non-equivariant arch to build",
    )
    parser.add_argument(
        "--tomo-dirs", type=str, default=None,
        help="Comma-separated tomogram subdirs to stack as channels, e.g. '100,wbp,ctfdeconvolved'",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        choices=["so3", "so3_aniso", "so2", "so2_iso"],
        help=(
            "so3: SO(3)-equivariant, isotropic targets (default). "
            "so3_aniso: SO(3)-equivariant, anisotropic z targets. "
            "so2: SO(2)-equivariant, anisotropic z targets. "
            "so2_iso: SO(2)-equivariant, isotropic targets."
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

    # Apply model variant — also controls z_sigma_scale for anisotropic targets.
    # The 2x2 ablation: architecture (so3 / so2) x targets (iso / aniso).
    if args.baseline:
        # Tag baseline checkpoints with the architecture so segresnet and
        # unet3d don't collide. Default architecture is segresnet.
        baseline_arch = args.baseline_arch or cfg.model.baseline_arch or "segresnet"
        variant = f"baseline_{baseline_arch}"
    else:
        variant = args.model_variant or "so3"
        cfg.model.model_variant = variant

    if variant in ("so3_aniso", "so2"):
        cfg.data.z_sigma_scale = 1.5   # anisotropic z heatmap targets
    else:
        # "so3", "so2_iso", and all baselines use isotropic targets
        cfg.data.z_sigma_scale = 1.0

    # Learnable per-class sigma_z overrides the static z_sigma_scale.
    # Dataloader emits xy-only-blurred targets; the loss applies a
    # differentiable z-Gaussian with a learnable per-class sigma.
    if args.learnable_z_sigma:
        cfg.train.learnable_z_sigma = True
        cfg.data.target_mode = "xy_blur"
        # Tag the run so checkpoints/logs don't collide with the static variant.
        variant = f"{variant}_lz"

    # CLI overrides for the new flags.
    if args.no_ema:
        cfg.train.use_ema = False
    if args.heavy_aug:
        cfg.data.heavy_augmentation = True
    if args.baseline_arch is not None:
        cfg.model.baseline_arch = args.baseline_arch
    if args.tomo_dirs is not None:
        cfg.data.tomo_dirs = tuple(
            t.strip() for t in args.tomo_dirs.split(",") if t.strip()
        )

    # Keep ModelConfig.in_channels in sync with the number of stacked tomogram
    # reconstructions. The dataset emits (C, D, H, W) where C == len(tomo_dirs).
    cfg.model.in_channels = len(cfg.data.tomo_dirs)

    train_cfg = cfg.train
    set_seed(train_cfg.seed, rank)

    # Checkpoint and log dirs are variant-specific so all variants coexist.
    # train_kfold.py overrides these via CZII_CKPT_DIR / CZII_LOG_DIR so each
    # fold gets its own subdirectory; otherwise default to ./{base}/{variant}/.
    ckpt_override = os.environ.get("CZII_CKPT_DIR")
    log_override  = os.environ.get("CZII_LOG_DIR")
    ckpt_base = Path(ckpt_override) if ckpt_override else Path(train_cfg.checkpoint_dir) / variant
    log_base  = Path(log_override)  if log_override  else Path(train_cfg.log_dir) / variant
    cfg.train.checkpoint_dir = str(ckpt_base)
    cfg.train.log_dir        = str(log_base)

    # train_kfold.py communicates the held-out run(s) via CZII_VAL_RUN.
    val_run_env = os.environ.get("CZII_VAL_RUN")
    if val_run_env:
        cfg.data.val_run_names = tuple(
            r.strip() for r in val_run_env.split(",") if r.strip()
        )
        if is_main(rank):
            print(f"  (fold override) val_run_names = {cfg.data.val_run_names}")

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
        # No SyncBatchNorm: equivariant blocks use e3nn BatchNorm with
        # instance=True (per-sample stats), and the SegResNet baseline
        # uses InstanceNorm. Both are independent of DDP world size, so
        # converting to SyncBN here would be a no-op or harmful.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    # Loss, optimizer, scheduler, scaler.  When learnable_z_sigma is on,
    # the criterion owns parameters (per-class log_z_scale) that must be
    # registered with the optimizer; pass it as extra_modules.
    criterion = build_loss(train_cfg).to(device)
    extra_loss_modules = [criterion] if train_cfg.learnable_z_sigma else None
    optimizer = build_optimizer(model, train_cfg, extra_modules=extra_loss_modules)

    # EMA shadow model (rank 0 only — only used for validation/checkpointing).
    # Built BEFORE the first training step so it captures the init.
    ema_model: Optional[AveragedModel] = None
    if is_main(rank) and train_cfg.use_ema:
        ema_model = build_ema_model(model, decay=train_cfg.ema_decay).to(device)
        print(f"EMA shadow model enabled with decay={train_cfg.ema_decay}")
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

        # Train (ema_model is rank-0 only; other ranks pass None)
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, train_cfg.use_amp, epoch, rank, world_size,
            writer=writer if is_main(rank) else None,
            ema_model=ema_model,
        )

        if is_main(rank) and writer:
            writer.add_scalar("train/loss",      train_metrics["train_loss"],      epoch)
            writer.add_scalar("train/grad_norm", train_metrics["train_grad_norm"], epoch)
            writer.add_scalar("train/lr",        optimizer.param_groups[0]["lr"],  epoch)
            # Log per-class learned z_scale when active so we can watch
            # the missing-wedge anisotropy emerge during training.
            if train_cfg.learnable_z_sigma and isinstance(criterion, LearnableZBlurLoss):
                with torch.no_grad():
                    z_scales = criterion.current_z_scales().detach().cpu().tolist()
                for cls_name, z in zip(CLASS_NAMES, z_scales):
                    writer.add_scalar(
                        f"learnable_z_sigma/{cls_name}", float(z), epoch,
                    )

        # Validate (only main process; val_loader has no sampler).
        # Evaluate the EMA shadow when available — that is the deployment model.
        val_metrics = {}
        if val_loader and (epoch + 1) % train_cfg.val_every == 0:
            if is_main(rank):
                eval_model = ema_model if ema_model is not None else model
                val_metrics = validate(
                    eval_model, val_loader, criterion, device,
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
                        ema_model=ema_model,
                    )

        # Full-volume validation — sliding-window inference on each val run,
        # F-beta on aggregated centroids.  Matches deployment behaviour.
        # Rank 0 only because it does I/O and would otherwise duplicate work.
        fv_every = getattr(train_cfg, "full_volume_val_every", 0)
        if (is_main(rank) and val_loader is not None and fv_every > 0
                and (epoch + 1) % fv_every == 0):
            val_runs = list(cfg.data.val_run_names)
            t_fv = time.time()
            fv_eval_model = ema_model if ema_model is not None else model
            fv_metrics = validate_full_volume(
                fv_eval_model,
                dataset_dir=cfg.data.dataset_dir,
                run_names=val_runs,
                data_cfg=cfg.data,
                inference_cfg=cfg.inference,
                device=device,
                use_amp=train_cfg.use_amp,
                batch_size=train_cfg.full_volume_val_batch_size,
                overlap=train_cfg.full_volume_val_overlap,
            )
            fv_elapsed = time.time() - t_fv
            print(
                f"  Full-volume val on {fv_metrics['full_val_runs']}: "
                f"F-beta={fv_metrics['full_val_fbeta']:.4f} "
                f"({fv_elapsed:.1f}s)"
            )
            if writer:
                writer.add_scalar("val_full/fbeta", fv_metrics["full_val_fbeta"], epoch)
                for c, fb in fv_metrics["full_val_fbeta_per_class"].items():
                    writer.add_scalar(f"val_full/fbeta_{c}", float(fb), epoch)
            # Track best by full-volume F-beta so checkpoint selection matches
            # deployment metric, not the biased patch-level F-beta.
            if fv_metrics["full_val_fbeta"] > best_fbeta:
                best_fbeta = fv_metrics["full_val_fbeta"]
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    {**train_metrics, **val_metrics, **fv_metrics},
                    os.path.join(str(ckpt_base), "best.pt"),
                    ema_model=ema_model,
                )

        # Periodic checkpoint — rank 0 only
        if is_main(rank) and (epoch + 1) % train_cfg.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {**train_metrics, **val_metrics},
                os.path.join(str(ckpt_base), f"epoch_{epoch:04d}.pt"),
                ema_model=ema_model,
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
            ema_model=ema_model,
        )
        if writer:
            writer.close()
        print(f"\nTraining complete. Best F-beta: {best_fbeta:.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
