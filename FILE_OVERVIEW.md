# Project File Overview

This document walks through every Python file at the project root, explains what it does, how it fits with the others, and flags any issues found during review.

The project trains and evaluates **SO(3)** / **SO(2)** rotation-equivariant 3D U-Nets (built on `e3nn`) for detecting the 3D centroids of 6 protein classes in cryo-ET tomograms (CZII dataset 10440).

---

## High-level architecture

```
config.py ──────────────► single source of truth for all hyper-params
   │
   ├─ dataset.py       (tomogram I/O, Gaussian heatmap targets, patch sampling, DataLoader)
   ├─ model.py         (LightEquivariantConv, SO2EquivariantConv, U-Net, baseline U-Net)
   ├─ losses.py        (WeightedMSE, FocalHeatmap, Combined)
   ├─ utils.py         (3D NMS, sub-voxel peak refinement, Hungarian F-β, sliding-window aggregation, submission CSV)
   │
   ├─ train.py         (full 2-GPU DDP training loop)
   ├─ train_laptop.py  (single-GPU "tiny" variant for 4–8 GB VRAM)
   │
   ├─ inference.py     (sliding-window inference + optional rotation-TTA + submission)
   ├─ compare_variants.py (load all three trained variants, evaluate on TS_99_9, print F-β table)
   │
   ├─ test_equivariance.py (correctness tests + 4 visualisations)
   ├─ sanity_check.py  (end-to-end smoke test on synthetic data — no dataset needed)
   ├─ new.py           (dataset verification — particle counts + heatmap sanity)
   │
   └─ dataset_download.py (one-liner to fetch dataset 10440 from CryoET portal)
```

---

## File-by-file

### [config.py](config.py)
The central config (dataclasses: `DataConfig`, `ModelConfig`, `TrainConfig`, `InferenceConfig`, `Config`). Also defines:

- `PROTEIN_CLASSES` — 6 target proteins with id, radius (Å), Gaussian σ (vox), class weight, colour.
- `ANNOTATION_DIR_TO_CLASS` — maps CryoET-portal annotation sub-directory IDs (`100`–`105`) to class names. Directory `106` (membrane segmentation mask) is intentionally not mapped.
- `ModelConfig.model_variant` ∈ {`so3`, `so3_aniso`, `so2`} — selects the equivariance group and target shape.
- `InferenceConfig.nms_kernel_sizes` — per-class NMS kernel ≈ `2·floor(radius) + 1`.

**Status: correct.** No imports needed besides stdlib.

---

### [dataset.py](dataset.py)
Loads tomograms (MRC / Zarr / NPY), parses CryoET-portal `*_point.ndjson` annotations, generates per-class 3D Gaussian heatmap targets, samples training patches, and constructs `DataLoader`s.

Key points:
- **Coordinate convention**: ndjson stores `{x, y, z}` voxel coords; internally everything is stored as `[z, y, x]` to match numpy `(D, H, W)`.
- `generate_gaussian_heatmap` normalises the heatmap so an isolated peak = 1.0 regardless of σ; supports anisotropic σ_z for missing-wedge-aware targets. Norm factor is cached.
- `CZIIDataset` pre-loads tomograms once (~2 GB total), generates heatmaps **per patch on the fly** with a `border` margin so particles just outside the patch still contribute Gaussian tails inside.
- `augment_patch` flips only in-plane (axes 1, 2) — flipping z would reverse the beam direction and create out-of-distribution data. 90° rotations are XY-only.
- `build_dataloaders` constructs train + val loaders, builds a `DistributedSampler` when `world_size > 1`, and **forces `num_workers=0` on Windows** to avoid the well-known spawn-pickling truncation bug with large numpy arrays.

**Status: correct.** Minor cosmetic issue: `from config import CLASS_SIGMAS_VOXELS` is re-imported inside `CZIIDataset.__init__` even though it's already imported at the top of the file. Harmless.

---

### [model.py](model.py)
All neural-net classes.

- `GaussianRadialBasis` — encodes voxel-grid distances as Gaussian basis features.
- `GatedNonlinearity` / `NormNonlinearity` — equivariant nonlinearities (scalars via SiLU; higher-l features either gated by a learned sigmoid scalar or rescaled by a function of their norm).
- `LightEquivariantConv` — the workhorse SO(3)-equivariant 3D conv. Pre-computes spherical harmonics on the K³ kernel grid, then builds the full `(C_out, C_in, K, K, K)` kernel with a **single batched `FullyConnectedTensorProduct` call** (rather than the original Python loop over K³ positions). Radial MLP output is normalised by `1/√weight_numel` for stable variance.
- `EquivariantZConv` — 1D conv along the z (beam) axis. Treats each component of each irrep as an independent "batch element" and applies a `(mul, mul)` weight matrix only across multiplicities — never across irrep components. This preserves SO(2) equivariance about the z-axis (each m-mode picks up the same `e^{imφ}` phase, which commutes with a per-component conv). `bias=False` (a non-zero bias would break equivariance for l > 0).
- `SO2EquivariantConv` — combines a K×K xy-only equivariant 2D conv (positions at z = 0 → circular harmonics) with an `EquivariantZConv` residual along z. Equivariant under z-rotations only; intentionally NOT equivariant under x/y rotations (matches cryo-ET's missing-wedge symmetry breaking).
- `EquivariantUNet3D` — 4-level encoder, 3-level decoder, skip connections; final `output_conv` projects to `num_classes × 0e` (scalar = rotation-invariant by construction), then `sigmoid`.
- `SO2EquivariantUNet3D` — same architecture but with `conv_cls=SO2EquivariantConv`.
- `UNet3D` — standard non-equivariant baseline for comparison.
- `build_model(cfg, use_equivariant)` — factory.

**Status: correct.** I traced the kernel-reshape index ordering carefully: in `_build_kernel`, the batch axis `K³·C_in` decomposes as `k·C_in + i`, so `kernel_flat.reshape(K3, C_in, C_out)` puts position outer and basis-vector inner, matching the broadcast pattern of `sh_batch` and `w_batch`. The subsequent `.permute(0, 2, 1).reshape(K, K, K, C_out, C_in).permute(3, 4, 0, 1, 2)` then yields the `(C_out, C_in, D, H, W)` layout expected by `F.conv3d`. The same logic in `_build_xy_kernel` is right for 2D conv. The trilinear-interpolate fallback in the decoder (when a skip's spatial size differs by ≤ 1 voxel) does not trigger for cube-shaped patches that divide cleanly by 8, so it doesn't break equivariance in practice.

---

### [losses.py](losses.py)
Three heatmap-regression losses:

- `WeightedMSELoss` — `mean( (pred - target)² · (target + α) )`. The `(target + α)` weight up-weights near-centroid voxels relative to the vast background.
- `FocalHeatmapLoss` — CornerNet/CenterNet-style focal loss with separate positive / negative formulations. Critically, it normalises by **total voxels** rather than positive count: the comment explains that normalising by `num_pos` produces ~17 K× amplification on sparsely-populated patches that overflows fp16 and stalls `GradScaler`.
- `CombinedLoss` — additive combination.
- `build_loss(cfg)` — factory dispatched on `cfg.loss_type` ∈ {`wmse`, `focal`, `combined`}. Per-class weights from `CLASS_WEIGHTS` applied when `loss_class_weights` is True.

**Status: correct.**

---

### [utils.py](utils.py)
Post-processing and evaluation helpers.

- `nms_3d` / `nms_3d_numpy` — 3D non-maximum suppression via max-pool / `maximum_filter`.
- `_subvoxel_centroid` — refines integer-peak coordinates with a weighted centroid over a small neighbourhood (compensates for the discrete grid).
- `extract_centroids` — full pipeline: per-class NMS (kernel chosen from `nms_kernel_sizes`), threshold, peak collection, sub-voxel refinement, sort by score.
- `match_predictions` — **Hungarian** (optimal-assignment) matching of predictions to GTs. The comment explains why greedy nearest-neighbour was rejected: it can lock a prediction onto a sub-optimal GT and lose a valid match elsewhere.
- `compute_fbeta_per_class` / `compute_weighted_fbeta` — CZII competition metric with per-class distance threshold = `0.5 · radius_voxels` and class weights from `CLASS_WEIGHTS`.
- `aggregate_predictions` — Gaussian-weighted stitching of patch-level heatmaps into a full-volume prediction; reduces seam artefacts.
- `generate_submission` — writes a CSV in the CZII competition format (coords converted voxel → Å).

**Status: correct.**

---

### [train.py](train.py)
Production training loop. Supports `torchrun --nproc_per_node=2 train.py --model-variant {so3, so3_aniso, so2}` or single-GPU fallback.

- `setup_ddp()` initialises the `nccl` process group from torchrun env vars; returns `(False, 0, 0, 1)` for single-GPU runs.
- Sets the seed offset by rank so each GPU sees a different data order.
- Sets `cfg.data.z_sigma_scale = 1.5` automatically for `so3_aniso` and `so2` variants (anisotropic targets); 1.0 for `so3`.
- Checkpoint / log directories are variant-suffixed (`./checkpoints/so2/best.pt`) so all three variants coexist without clobbering each other.
- Optimiser groups: AdamW with weight-decay skipped for bias / batch-norm parameters.
- LR schedule: linear warmup then cosine decay.
- AMP via `torch.amp` (`GradScaler`, `autocast`), with `scaler.unscale_` before `clip_grad_norm_` (correct order).
- DDP wraps the model after `SyncBatchNorm.convert_sync_batchnorm`; checkpoint saving unwraps `.module` so saved weights load on any number of GPUs.
- Validation runs only on rank 0 (val loader has no `DistributedSampler`).
- Saves `best.pt` whenever val F-β improves; periodic `epoch_NNNN.pt` checkpoints by `save_every`.

**Status: correct.** One DDP nit worth knowing about: while rank 0 is running validation, ranks 1+ have already moved on to the next training epoch — they will block at the first backward-pass all-reduce until rank 0 catches up. In practice validation is short and this is fine, but for long val passes it can risk a NCCL timeout. Not a bug per se, just a known cost of the simplified val-on-rank-0-only design.

---

### [train_laptop.py](train_laptop.py)
Stripped-down single-GPU variant of `train.py` for laptop GPUs (4–8 GB VRAM).

Differences from the full pipeline:
- Tiny irrep config (~850 K params) — encoder/decoder 3 levels instead of 4.
- `kernel_size = 3` (27 positions vs 125).
- `patch_size = (64, 64, 64)`, `batch_size = 2`, fewer patches per tomo.
- AMP always on.
- No DDP, no `DistributedSampler`, no rank gating.
- Checkpoints to `./checkpoints_laptop/{variant}/`, logs to `./logs_laptop/{variant}/`.

**Status: correct.** Re-uses `build_loss`, `extract_centroids`, `compute_weighted_fbeta` from the rest of the codebase, so it always tracks any fixes made elsewhere.

---

### [inference.py](inference.py)
Sliding-window inference on full tomograms.

- `predict_tomogram` builds a `CZIIInferenceDataset`, runs the model on each patch under AMP, then calls `aggregate_predictions` (Gaussian-weighted stitching).
- `detect_proteins` calls `extract_centroids` and prints per-class counts.
- `predict_with_rotation_tta` — optional 4× rotation-augmented inference (rotate input, predict, inverse-rotate the heatmap, average). Useful for the non-equivariant baseline; the equivariant model gets a smaller boost (mostly cleans up sliding-window seam effects).
- `main` loops over an input tomogram or a directory of them; writes `{tomo_id}_heatmap.npy`, `{tomo_id}_detections.npy`, and a final `submission.csv`.

**Status: correct.** Minor code smell: `np.save(det_path, detections)` is called with a Python dict, which numpy persists via pickle (emits an `allow_pickle` warning on load); switching to `np.savez(det_path, **detections)` would be cleaner but is not a correctness issue.

---

### [compare_variants.py](compare_variants.py)
Loads the `best.pt` checkpoint for each of the three variants, evaluates on the held-out validation run (`TS_99_9`), and prints a per-class + weighted F-β comparison table. Skips any variant whose checkpoint is missing. Saves results to `compare_results.json`.

**Status: correct.** Imports `DataLoader` from `dataset` rather than `torch.utils.data` — works because `dataset.py` re-exports it via its own top-level import. Unconventional but functional.

---

### [test_equivariance.py](test_equivariance.py)
The most carefully built file. Two test families plus four visualisations.

Tests:
1. `verify_grid_rotations()` — places a unit delta in a cube, applies `torch.rot90`, and checks the delta lands at the position predicted by the analytical `R · r` (the `GRID_ROTATIONS` table at the top).
2. `verify_d_for_l1(R)` — checks that the Wigner D-matrix for `1x1o` equals R itself (a fundamental e3nn convention check).
3. **Output invariance** (`test_output_invariance`): `model(R·X) ≈ R·model(X)` for all 9 non-trivial 90° grid rotations. Threshold 5e-3.
4. **Feature equivariance** (`test_feature_equivariance`): `f_level(R·X) = D(R) · spatial_rot(f_level(X), R⁻¹)` at encoder levels 0–2.
5. **Baseline non-equivariance**: standard `UNet3D` is expected to FAIL most rotations (excluding the 180° cases which can pass by symmetry).
6. **SO(2) tests**: z-rotations must PASS (~5e-3), x/y rotations must FAIL (>1e-3). This makes the SO(2) variant's intentional symmetry breaking explicit.

Visualisations (`equivariance_error_vs_angle.png`, `heatmap_comparison.png`, `wigner_d_structure.png`, `feature_decomposition.png`) show the equivariance error curve across arbitrary angles, side-by-side heatmap comparisons, the block-diagonal D-matrix structure, and the magnitudes of the scalar/vector/tensor feature channels.

Critical config in `build_test_model`: `use_batch_norm=False` and `activation="norm"`. The comment explains why — BN's running stats are uninitialised at eval-time on tiny test volumes and would corrupt the equivariance check.

**Status: correct.** Comprehensive and well thought through.

---

### [sanity_check.py](sanity_check.py)
End-to-end smoke test — **no dataset required**. Runs:

1. Environment dump (Torch, e3nn, GPU).
2. SO(3) and SO(2) forward pass — checks output shape `(B, 6, 32, 32, 32)` and that the sigmoid output is in `[0, 1]`.
3. A 5-step training loop on synthetic data; checks losses are finite and decreasing.
4. Centroid extraction — places known GT Gaussians, runs `extract_centroids`, asserts sub-voxel error < 0.5 voxels.
5. Hungarian matching — three predictions, two GTs, distance threshold 2.0 → expected `(tp=2, fp=1, fn=0)`.
6. Full F-β pipeline.
7. VRAM usage report (asserts under 90 %).

A failure exits with `sys.exit(1)`, making it suitable as a pre-flight check.

**Status: correct.**

---

### [new.py](new.py)
Dataset-verification script. Prints:
- The 7 runs discovered under `./dataset/10440/`.
- A per-run × per-class table of annotation counts.
- The σ used per class.
- For one chosen run (default `TS_5_4`): tomogram shape, dtype, value range, and a heatmap-generation sanity check (max value per class channel).

Useful as the first thing to run after downloading the data.

**Status: correct.** (The filename is unfortunately generic — `verify_dataset.py` would be clearer, but the docstring is unambiguous.)

---

### [dataset_download.py](dataset_download.py)
Two-line script that uses the `cryoet_data_portal` SDK to download dataset 10440 in full.

**Status: correct** (assuming `cryoet_data_portal` is installed and an auth token is available if required).

---

### [requirements.txt](requirements.txt)
PyTorch ≥ 2.1, `e3nn` ≥ 0.5.1, `mrcfile`, `zarr`, `ome-zarr`, `copick`, `scikit-image/learn`, `tensorboard`, `monai`, etc. Looks consistent with everything imported across the project.

---

## Issues / observations

Nothing that breaks correctness. The only items worth flagging:

1. **`train.py` validation-on-rank-0-only** can in principle hold up the other ranks at the next backward pass; benign for short val passes but worth knowing if validation ever grows much longer.
2. **`inference.py` saves a dict via `np.save`** — works (pickle), but `np.savez(**detections)` is the idiomatic replacement.
3. **`compare_variants.py` imports `DataLoader` from `dataset`** rather than from `torch.utils.data` directly — works via re-export, slightly unconventional.
4. **`new.py`** is named after no particular concept; consider renaming to `verify_dataset.py` (it's already mentioned in [report.md](report.md) and the memory file under its current name though).
5. **Redundant import** in `dataset.py` `CZIIDataset.__init__` (`from config import CLASS_SIGMAS_VOXELS`) — already imported at the top of the file.

Everything else — equivariant kernel construction, SO(2) component-wise z-conv, DDP setup, Hungarian F-β matching, anisotropic heatmap generation, missing-wedge-aware augmentation, sliding-window aggregation — checks out.
