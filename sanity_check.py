"""
Quick sanity check — no dataset required.

Runs a mini training loop (5 steps) and a validation step with synthetic
random data, verifying that every component works end-to-end:
  - model forward pass (SO3 and SO2)
  - loss backward
  - optimizer step
  - centroid extraction  (Hungarian + subvoxel)
  - F-beta computation

Usage:
    python sanity_check.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import math, time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config import Config, CLASS_NAMES
from model import build_model
from losses import build_loss
from utils import extract_centroids, compute_weighted_fbeta, match_predictions
from dataset import generate_gaussian_heatmap


def make_tiny_cfg(variant="so3"):
    cfg = Config()
    cfg.model.encoder_irreps = ["8x0e + 4x1o + 2x2e", "16x0e + 8x1o + 4x2e"]
    cfg.model.decoder_irreps = ["8x0e + 4x1o + 2x2e"]
    cfg.model.kernel_size = 3
    cfg.model.lmax = 2
    cfg.model.model_variant = variant
    return cfg


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check(condition, label):
    mark = "[PASS]" if condition else "[FAIL]"
    print(f"  {mark}  {label}")
    if not condition:
        sys.exit(1)


# -----------------------------------------------------------------------
section("Environment")
# -----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU    : {p.name}  VRAM={p.total_memory//1024**2} MB")

import e3nn
print(f"  e3nn   : {e3nn.__version__}")
print(f"  PyTorch: {torch.__version__}")

# -----------------------------------------------------------------------
section("Model forward pass — SO3 and SO2")
# -----------------------------------------------------------------------
PATCH = 32
BATCH = 2

for variant in ["so3", "so2"]:
    cfg = make_tiny_cfg(variant)
    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(BATCH, 1, PATCH, PATCH, PATCH, device=device)
    with torch.no_grad():
        y = model(x)
    check(y.shape == (BATCH, 6, PATCH, PATCH, PATCH),
          f"{variant} output shape {tuple(y.shape)}")
    check(0 <= y.min() <= y.max() <= 1,
          f"{variant} output in [0,1]: min={y.min():.3f} max={y.max():.3f}")
    print(f"    params={n_params:,}")
    del model, x, y

# -----------------------------------------------------------------------
section("Loss computation and backward")
# -----------------------------------------------------------------------
cfg = make_tiny_cfg("so3")
model = build_model(cfg.model).to(device)
criterion = build_loss(cfg.train).to(device)
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
t0 = time.time()
for step in range(5):
    x   = torch.randn(BATCH, 1, PATCH, PATCH, PATCH, device=device)
    tgt = torch.rand(BATCH, 6, PATCH, PATCH, PATCH, device=device) * 0.1  # sparse targets
    opt.zero_grad(set_to_none=True)
    with autocast("cuda", enabled=torch.cuda.is_available()):
        y    = model(x)
        loss = criterion(y, tgt)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)  # must precede clip_grad_norm_ so grads are in real scale
    gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    losses.append(loss.item())
    print(f"  step {step}: loss={loss.item():.4f}  gnorm={float(gnorm):.3f}")

elapsed = time.time() - t0
check(not math.isnan(losses[-1]), "loss is not NaN")
check(all(math.isfinite(l) for l in losses), "all losses finite")
check(losses[-1] < losses[0], f"loss decreased: {losses[0]:.2f} -> {losses[-1]:.2f}")
print(f"  {elapsed/5:.2f}s/step on {device}")

# -----------------------------------------------------------------------
section("Centroid extraction — subvoxel + per-class NMS")
# -----------------------------------------------------------------------
vol_shape = (64, 64, 64)
GT_POS = {
    "ferritin_complex": np.array([[20.0, 21.0, 22.0]]),
    "cytosolic_ribosome": np.array([[44.0, 44.0, 44.0]]),
}
heatmaps = np.zeros((6, *vol_shape), dtype=np.float32)
for cls_name, pts in GT_POS.items():
    from config import PROTEIN_CLASSES, CLASS_SIGMAS_VOXELS
    cls_id = PROTEIN_CLASSES[cls_name]["id"]
    sigma  = CLASS_SIGMAS_VOXELS[cls_id]
    heatmaps[cls_id] = generate_gaussian_heatmap(vol_shape, pts, sigma=sigma)

score_thresholds = cfg.inference.score_thresholds
nms_kernel_sizes = cfg.inference.nms_kernel_sizes
dets = extract_centroids(heatmaps, score_thresholds,
                         nms_kernel_sizes=nms_kernel_sizes, subvoxel=True)

for cls_name, gt_pts in GT_POS.items():
    det = dets[cls_name]
    check(len(det) >= 1, f"{cls_name}: at least 1 detection (got {len(det)})")
    z, y_c, x_c, score = det[0]
    gt_z, gt_y, gt_x = gt_pts[0]
    err = math.sqrt((z-gt_z)**2 + (y_c-gt_y)**2 + (x_c-gt_x)**2)
    check(err < 0.5, f"{cls_name}: subvoxel error {err:.3f} < 0.5 vox")
    print(f"    {cls_name}: detected at ({z:.2f},{y_c:.2f},{x_c:.2f}) "
          f"GT=({gt_z},{gt_y},{gt_x}) err={err:.3f}")

# -----------------------------------------------------------------------
section("Hungarian matching")
# -----------------------------------------------------------------------
pred = np.array([[0,0,0],[10,10,10],[5,5,5]], dtype=float)
gt   = np.array([[0.4,0,0],[10,11,10]], dtype=float)
tp, fp, fn = match_predictions(pred, gt, distance_threshold=2.0)
check(tp == 2 and fp == 1 and fn == 0,
      f"Hungarian: tp={tp} fp={fp} fn={fn}  (expected 2,1,0)")

# -----------------------------------------------------------------------
section("F-beta metric end-to-end")
# -----------------------------------------------------------------------
pred_centroids = {c: dets[c][:, :3] if len(dets[c]) > 0 else np.empty((0,3))
                  for c in CLASS_NAMES}
gt_centroids   = {c: GT_POS.get(c, np.empty((0,3))) for c in CLASS_NAMES}
fbeta = compute_weighted_fbeta(pred_centroids, gt_centroids)
check(fbeta > 0, f"weighted F-beta = {fbeta:.4f} (> 0)")
print(f"  weighted F-beta: {fbeta:.4f}")

# -----------------------------------------------------------------------
section("VRAM usage summary")
# -----------------------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.synchronize()
    used  = torch.cuda.max_memory_allocated() // 1024**2
    total = torch.cuda.get_device_properties(0).total_memory // 1024**2
    print(f"  Peak VRAM used: {used} / {total} MB  "
          f"({100*used/total:.1f}%)")
    check(used < total * 0.9, f"VRAM usage under 90%")

# -----------------------------------------------------------------------
print()
print("=" * 55)
print("  ALL SANITY CHECKS PASSED")
print("=" * 55)
