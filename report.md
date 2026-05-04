# SO(3)-Equivariant Protein Detection in Cryo-ET — Technical Report

## 1. Problem Statement

Cryo-electron tomography (cryo-ET) reconstructs 3D electron-density volumes (tomograms) of biological samples vitrified in ice. The task is to locate the 3D centroids of specific protein complexes inside these noisy, low-contrast volumes.

The key scientific insight driving the architecture: **proteins can appear in any orientation** inside a cell. A convolutional network that is not rotation-equivariant must learn all orientations from data; an SO(3)-equivariant network learns them once and generalises for free, which matters greatly when labelled data is scarce.

---

## 2. Dataset: CZII CryoET Object Identification Challenge (ID 10440)

### 2.1 Acquisition

| Property | Value |
|---|---|
| Instrument | Krios G4, Falcon 4i detector, SelectrisX energy filter |
| Sample | HEK293T cell lysate spiked with 6 purified proteins |
| Reconstruction | AreTomo3 v1.0.23 / v2.0.10 (WBP), denoised with DenoisET |
| Primary voxel spacing | **10.012 Å/voxel** |
| Tomogram size | 184 × 630 × 630 voxels (z × y × x) |

### 2.2 Runs (7 tomograms)

`TS_5_4`, `TS_6_4`, `TS_6_6`, `TS_69_2`, `TS_73_6`, `TS_86_3`, `TS_99_9`

### 2.3 Protein Classes (6 targets)

| Class name | Approx. diameter | Gaussian σ (vox) | Class weight |
|---|---|---|---|
| `ferritin_complex` | 120 Å | 2.5 | 1.0 |
| `beta_amylase` | 75 Å | 1.5 | 2.0 |
| `beta_galactosidase` | 160 Å | 3.5 | 1.5 |
| `cytosolic_ribosome` | 250–300 Å | 6.0 | 1.0 |
| `thyroglobulin` | ~180 Å mean | 4.0 | 2.0 |
| `pp7_vlp` | 270 Å | 6.0 | 1.0 |

Membrane (annotation dir 106) is a segmentation mask, not a point annotation — it is **excluded** from training.

### 2.4 Annotation Format

Point annotations are stored as `.ndjson` files under:
```
{run}/Reconstructions/VoxelSpacing10.012/Annotations/{id}/{class}-{ver}_point.ndjson
```

Each line:  `{"type": "point", "location": {"x": float, "y": float, "z": float}}`

**Coordinate space**: voxel indices at VoxelSpacing10.012.  Confirmed by range analysis:
- x, y ∈ [42, 584] ⊂ [0, 630) → voxel units ✓
- z ∈ [55, 177] ⊂ [0, 184) → voxel units ✓ (edges avoided, as expected in cryo-ET)

Loaded with ordering reordered to `[z, y, x]` to match numpy `(D, H, W)` convention.

---

## 3. Architecture Overview

```
Input (1, D, H, W) float32 density
        │
        ▼
  [Input Projection: LightEquivariantConv  1x0e → 32x0e+16x1o+8x2e]
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   ENCODER (3 downsamplings)             │
  │  Level 0: EquivariantResBlock  32x0e+16x1o+8x2e        │
  │  AvgPool3d(2)                                          │
  │  Level 1: Trans+Res            48x0e+24x1o+12x2e       │
  │  AvgPool3d(2)                                          │
  │  Level 2: Trans+Res            64x0e+32x1o+16x2e       │
  │  AvgPool3d(2)                                          │
  │  Level 3 (bottleneck): Trans+Res  96x0e+48x1o+24x2e   │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   DECODER (3 upsamplings)               │
  │  Upsample → cat(skip L2) → LightConv+Res               │
  │  Upsample → cat(skip L1) → LightConv+Res               │
  │  Upsample → cat(skip L0) → LightConv+Res               │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  [Output Head: LightEquivariantConv → 6x0e (scalars only)]
        │
  sigmoid
        │
        ▼
  Output (6, D, H, W) heatmaps ∈ [0, 1]
```

The output layer outputs **only scalar (l=0) irreps**. Scalars are automatically invariant under SO(3) rotations, so the output heatmap is rotation-invariant by construction — not by training.

---

## 4. Files

### 4.1 `config.py`

Central configuration dataclasses.  All hyperparameters live here.

| Symbol | Description |
|---|---|
| `PROTEIN_CLASSES` | Dict mapping class names → id, radius, sigma, weight, color |
| `NUM_CLASSES` | 6 |
| `CLASS_NAMES` | Ordered list of class names |
| `CLASS_WEIGHTS` | Per-class loss weights |
| `CLASS_RADII_VOXELS` | Radii in voxels at 10 Å/vox |
| `CLASS_SIGMAS_VOXELS` | Per-class Gaussian heatmap sigma (voxels) |
| `ANNOTATION_DIR_TO_CLASS` | Maps annotation subdirectory ID → class name |
| `DataConfig` | Paths, patch size, augmentation, Gaussian parameters |
| `ModelConfig` | Irrep specs, kernel size, lmax, activation type |
| `TrainConfig` | Epochs, LR, loss type, checkpointing |
| `InferenceConfig` | Patch overlap, NMS, score thresholds, submission output |
| `Config` | Master config bundling all four sub-configs |

**Key design choice**: per-class `sigma_voxels` is stored in `PROTEIN_CLASSES` and used in heatmap generation. Small proteins (beta_amylase, σ=1.5) get tight Gaussians; large ones (ribosome, VLP, σ=6) get broad Gaussians — matching their physical extent.

---

### 4.2 `dataset.py`

| Function / Class | Purpose |
|---|---|
| `load_tomogram(path)` | Load `.mrc`, `.zarr`, or `.npy` → `(D,H,W)` float32 ndarray |
| `normalize_tomogram(vol, clip_pct)` | Percentile-clip then z-score normalise |
| `load_ndjson_points(path)` | Parse one `*_point.ndjson` file → `(N,3)` [z,y,x] voxel coords |
| `load_run_annotations(run_dir, vs_dir)` | Load all point annotations for one run |
| `load_dataset_annotations(dataset_dir, vs_dir)` | Load annotations for all runs |
| `get_tomogram_path(dataset_dir, run_name, ...)` | Return path to default denoised MRC |
| `discover_run_names(dataset_dir)` | List all `TS_*` directories |
| `_gaussian_peak_value(sigma, truncate)` | Compute normalisation factor for isolated Gaussian |
| `generate_gaussian_heatmap(shape, centroids, sigma, truncate)` | Single-class Gaussian heatmap |
| `generate_multiclass_heatmap(shape, annotations, ...)` | `(C,D,H,W)` per-class heatmaps |
| `extract_random_patch(...)` | Biased patch sampling (positive_ratio near particles) |
| `augment_patch(vol, hm, cfg)` | Flips, 90° rotations, noise, contrast |
| `_sliding_origins(size, patch, stride)` | Origin positions with guaranteed edge coverage |
| `CZIIDataset` | PyTorch Dataset — patch-based training, pre-loads all tomos |
| `CZIIInferenceDataset` | Sliding-window dataset for full-tomogram inference |
| `build_dataloaders(cfg)` | Factories for train + val DataLoaders |

#### Heatmap normalisation

A key correction from the original code:

```
heatmap = gaussian_filter(delta_at_centroids, sigma=sigma)
heatmap /= _gaussian_peak_value(sigma)   # isolated peak → 1.0
heatmap = clip(heatmap, 0, 1)           # overlapping particles → 1.0
```

This is the CenterNet convention: each isolated centroid peak is exactly 1.0 regardless of sigma; overlapping particles are clamped. The original code divided by the *global max* of the filtered heatmap, which meant clustered particles got artificially reduced peak values.

---

### 4.3 `model.py`

#### `GaussianRadialBasis`

Maps scalar distances `||r||` to a `num_basis`-dimensional feature vector using evenly-spaced Gaussian basis functions:

```
φ_i(r) = exp(-(r - μ_i)² / (2σ²))
```

Used to encode the radial dependence of kernel weights.

#### `LightEquivariantConv` (primary convolution)

An SO(3)-equivariant 3D convolution layer. The kernel is constructed as:

```
K(r) = TP(·, Y_l(r/||r||), w(||r||))
```

where:
- `Y_l` are real spherical harmonics up to degree `lmax` — encode the **angular** dependence
- `w(||r||) = MLP(φ(||r||))` — learned **radial** weights from the radial basis
- `TP` is a Clebsch-Gordan tensor product that couples input and angular irreps to output irreps

**Equivariance proof sketch**: Under rotation R, input irreps transform as `D(R)·f`, and `Y_l(Rr) = D_l(R)·Y_l(r)`. The TP is constructed to satisfy `TP(D(R)·f, D(R)·Y, w) = D_out(R)·TP(f, Y, w)`, so the full convolution output transforms correctly. The final output projects to 0e (scalars), which are invariant.

**Kernel extraction** (`_build_kernel`): Since TP is linear in its first argument, the full `(C_out, C_in, K, K, K)` kernel matrix can be extracted by applying the TP to each basis vector of the input space.  The vectorised implementation batches all K³ × C_in calls into a single TP forward pass (replaces original K³-iteration Python loop which was ~125× slower).

#### `GatedNonlinearity`

Equivariant gating for mixed-irrep feature tensors:
- `l=0` (scalars): standard SiLU
- `l>0` (vectors, tensors): gated by learned scalar features via sigmoid

Preserves equivariance: the gate is invariant (scalar), so multiplying a vector by it does not change its transformation law.

#### `NormNonlinearity`

Alternative: `silu(||f||) / ||f|| · f`. Preserves equivariance because the norm is invariant.

#### `EquivariantAvgPool3d` / `EquivariantUpsample3d`

Average pooling and trilinear interpolation are both **linear** operations → they commute with the SO(3) action and preserve equivariance exactly. (Max pooling would break equivariance and is not used.)

#### `EquivariantResBlock`

Two `LightEquivariantConv` layers with skip connection. Second layer has no activation to avoid non-linearity before the residual add (standard pre-activation ResNet design).

#### `EquivariantUNet3D`

Full encoder-decoder U-Net. All intermediate features are typed as SO(3) irreps. Skip connections concatenate (direct sum) encoder and decoder features of the same resolution, preserving the irrep type structure. The final `output_conv` projects to `6x0e` (six scalar channels) — by construction rotation-invariant.

#### `UNet3D`

Standard 3D U-Net baseline (non-equivariant) for comparison. Uses Conv3d + BatchNorm + ReLU blocks.

---

### 4.4 `losses.py`

| Class | Formula | Use case |
|---|---|---|
| `WeightedMSELoss` | `mean((pred-target)² · (target + α))` | Regression baseline, α downweights background |
| `FocalHeatmapLoss` | CenterNet focal: pos `-(1-p)^α log p`, neg `-(1-t)^β p^α log(1-p)` | Extreme fg/bg imbalance |
| `CombinedLoss` | `λ_wmse · WMSE + λ_focal · Focal` | Default; best of both |

`build_loss(cfg)` constructs the appropriate loss from config.

**Class weighting**: beta_amylase and thyroglobulin get `weight=2.0` since they are harder to detect (small size / elongated shape). Applied as a `(1, C, 1, 1, 1)` multiplicative factor on the per-channel loss.

---

### 4.5 `utils.py`

| Function | Description |
|---|---|
| `nms_3d(heatmap, kernel_size)` | Torch 3D NMS via max-pool; keeps local maxima |
| `nms_3d_numpy(heatmap, kernel_size)` | NumPy/scipy version |
| `extract_centroids(heatmaps, thresholds, nms_kernel)` | Threshold + NMS → per-class `(N,4)` [z,y,x,score] |
| `match_predictions(pred, gt, distance_threshold)` | Greedy nearest-neighbour TP/FP/FN |
| `compute_fbeta_per_class(pred, gt, β, dist_scale)` | Per-class F_β score |
| `compute_weighted_fbeta(pred, gt, β, dist_scale)` | Weighted mean F_β (competition metric) |
| `aggregate_predictions(patch_preds, shape, n_classes)` | Gaussian-weighted stitching of patch heatmaps |
| `_gaussian_window_3d(d, h, w)` | 3D Gaussian window for stitching |
| `voxel_to_angstrom` / `angstrom_to_voxel` | Coordinate conversion |
| `generate_submission(all_detections, path)` | Write competition submission CSV |

**F_β metric**: the competition uses β=4, which weights recall 16× more than precision (i.e. missing a particle is penalised much more than a false detection). Distance threshold per class = 0.5 × radius_voxels.

---

### 4.6 `train.py`

Full training pipeline.

| Function | Description |
|---|---|
| `set_seed(seed)` | Reproducible RNG across Python / NumPy / PyTorch |
| `build_optimizer(model, cfg)` | AdamW; no weight decay on biases / norms |
| `build_scheduler(optimizer, cfg, steps)` | Linear warmup → cosine annealing |
| `train_one_epoch(...)` | AMP forward + backward + grad-clip + step |
| `validate(...)` | Patch-level loss + F_β from heatmap peak extraction |
| `save_checkpoint(...)` | Saves model, optimiser, scheduler, scaler, metrics |
| `load_checkpoint(...)` | Resumes training from saved state |
| `main()` | CLI entry point; arg-parses and orchestrates training |

**Mixed precision**: `torch.cuda.amp.GradScaler` with gradient clipping at norm=1.0 prevents fp16 overflow in the equivariant tensor products.

**Validation note**: GT centroids during validation are approximated by extracting peaks from the Gaussian target heatmap at threshold 0.5. This is a reasonable proxy for training monitoring. For final evaluation, run `inference.py` against the full tomograms and compare with the original ndjson annotations.

---

### 4.7 `inference.py`

| Function | Description |
|---|---|
| `predict_tomogram(model, vol, cfg, ...)` | Sliding-window inference + Gaussian-weighted aggregation |
| `detect_proteins(heatmap, cfg)` | NMS + threshold → per-class centroid list |
| `predict_with_rotation_tta(...)` | Test-time augmentation: average 4 × 90° XY rotations |
| `main()` | CLI: load checkpoint, run inference, save submission CSV |

**TTA note**: For a perfectly equivariant model, TTA over rotations that ARE in the equivariance group (90° multiples) should give identical results and is unnecessary. TTA is most useful for the non-equivariant baseline, or to average out discretisation artifacts at patch boundaries.

---

### 4.8 `new.py` (dataset verification)

Standalone script to verify the dataset structure and annotation loading.

```bash
python new.py                        # use default ./dataset/10440
python new.py --dataset /path/to/10440 --run TS_69_2
```

Prints: particle counts per run × class, tomogram shapes, heatmap peak values.

---

### 4.9 `dataset_download.py`

Minimal download script using the CryoET Data Portal Python client:

```python
from cryoet_data_portal import Client, Dataset
client = Client()
Dataset.get_by_id(client, 10440).download_everything()
```

---

## 5. Data Flow: Training

```
dataset/10440/TS_*/
    Tomograms/100/TS_*.mrc   ──→  load_tomogram  ──→  normalize  ──→  vol (D,H,W)
    Annotations/10*/..ndjson ──→  load_ndjson_points ──→  [z,y,x] coords
                                                    ──→  generate_multiclass_heatmap
                                                    ──→  hm (C,D,H,W)

CZIIDataset.__getitem__:
    (vol, hm) ──→ extract_random_patch (biased toward particles)
              ──→ augment_patch (flips, 90° rot, noise)
              ──→ (B,1,D,H,W) tensor + (B,C,D,H,W) heatmap target

train_one_epoch:
    batch ──→ EquivariantUNet3D.forward ──→ (B,C,D,H,W) pred
    CombinedLoss(pred, target)  ──→  scalar loss
    AMP backward + AdamW step + cosine LR
```

---

## 6. Data Flow: Inference

```
vol (D,H,W) ──→ CZIIInferenceDataset ──→ sliding patches (1,pd,ph,pw)
             ──→ EquivariantUNet3D    ──→ (B,C,pd,ph,pw) patch heatmaps
             ──→ aggregate_predictions (Gaussian-weighted stitching)
             ──→ (C,D,H,W) full-volume heatmap
             ──→ nms_3d + threshold
             ──→ {class: (N,4) [z,y,x,score]}
             ──→ submission.csv (z*10.012, y*10.012, x*10.012 Å)
```

---

## 7. Bugs Fixed (vs. Original Code)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `config.py` | 5 wrong protein class names (`apo-ferritin`, `ribosome`, `virus-like-particle`, missing `beta_amylase`) | Updated to 6 correct names matching dataset |
| 2 | `config.py` | Single global sigma=5 for all classes | Per-class `sigma_voxels` (1.5–6.0) in `PROTEIN_CLASSES` |
| 3 | `dataset.py` | `load_annotations` expected a CSV that does not exist | Replaced with `load_ndjson_points` + `load_run_annotations` + `load_dataset_annotations` |
| 4 | `dataset.py` | No coordinate reordering from ndjson {x,y,z} to numpy [z,y,x] | Explicit `[loc["z"], loc["y"], loc["x"]]` on load |
| 5 | `dataset.py` | `build_dataloaders` hard-coded CSV path, ignored actual dataset structure | Now traverses `dataset/10440/TS_*/Reconstructions/...` |
| 6 | `dataset.py` | `CZIIInferenceDataset` modified loop variable `z0/y0/x0 = min(...)` mid-loop — caused duplicate and missing origins | Rewrote with `_sliding_origins` helper; last origin always = `max(size-patch, 0)` |
| 7 | `dataset.py` | Heatmap normalised by `global_max` — overlapping particles got reduced peak values | Normalise by the peak value of a single isolated Gaussian; clip at 1.0 |
| 8 | `model.py` | `_build_kernel` looped K³=125 times, one TP call per iteration | Single vectorised TP call over K³×C_in batch |
| 9 | `model.py` | `activation="none"` string (semantically unclear) in `EquivariantResBlock` | Changed to `"linear"` (also falls to `None` activation via else-branch) |
| 10 | `train.py` | `os.makedirs(os.path.dirname(path))` crashes for bare filename | `os.makedirs(os.path.dirname(os.path.abspath(path)))` |

---

## 8. Theoretical Correctness Notes

### SO(3) Equivariance on a Cartesian Grid

The architecture is **exactly** equivariant to the following operations:
- The continuous SO(3) action on the feature irreps (by construction of the TP kernel)
- The discrete 90° rotations of the 3D voxel grid (since the kernel positions are symmetric)

It is **approximately** equivariant to arbitrary continuous rotations because:
- The tomogram lives on a fixed Cartesian grid; rotation by an arbitrary angle requires resampling
- Trilinear interpolation (used in upsampling) is equivariant only at the grid discretisation error level
- This approximation is standard practice in all 3D equivariant CNNs (SE3-DNN, SEGNN, 3D-IEConv)

For the application (cryo-ET, 10 Å/voxel), this approximation is accurate enough: the effective resolution of the tomogram (~20–30 Å after denoising) is larger than the discretisation error.

### Kernel Correctness

The equivariant kernel `K(r) = TP(·, Y_l(r̂), w(||r||))` satisfies:

```
K(R·r)   =  D_out(R) · K(r) · D_in(R)^T
```

for all R ∈ SO(3), where `D_l(R)` is the Wigner D-matrix for degree l. This is guaranteed by the Clebsch-Gordan structure of `FullyConnectedTensorProduct` from e3nn (Batzner et al. 2022, NeurIPS; Thomas et al. 2018).

Extracting this as a standard `F.conv3d` kernel is valid because the TP is **linear** in its first argument: `TP(f_in, sh, w) = M(sh, w) · f_in`. The matrix `M` is extracted column-by-column by evaluating `TP(e_i, sh_k, w_k)` for each basis vector `e_i`.

### Gated Nonlinearity

The gate `g = sigmoid(W_s · scalars)` is a scalar (l=0, invariant). Multiplying a degree-l feature by a scalar does not change its transformation law under SO(3). Equivariance is therefore preserved.

---

## 9. Training Guide

### Setup

```bash
pip install torch e3nn mrcfile zarr scipy numpy pandas tqdm tensorboard
```

### Download dataset (if not already present)

```bash
pip install cryoet-data-portal
python dataset_download.py
```

### Verify dataset

```bash
python new.py
```

### Train equivariant model

```bash
python train.py --epochs 200 --batch-size 2
```

### Train standard U-Net baseline

```bash
python train.py --baseline --epochs 200
```

### Resume from checkpoint

```bash
python train.py --resume ./checkpoints/epoch_0099.pt
```

### Run inference

```bash
python inference.py \
    --checkpoint ./checkpoints/best.pt \
    --input ./dataset/10440/TS_99_9/Reconstructions/VoxelSpacing10.012/Tomograms/100/TS_99_9.mrc \
    --output ./predictions
```

### With rotation TTA

```bash
python inference.py --checkpoint ./checkpoints/best.pt --input ... --tta
```

---

## 10. References

1. **e3nn** — Batzner et al. (2022). *e3nn: Euclidean Neural Networks*. NeurIPS 2022. — equivariant tensor product implementation
2. **CenterNet** — Zhou et al. (2019). *Objects as Points*. arXiv:1904.07850. — Gaussian heatmap target and focal loss design
3. **CornerNet** — Law & Deng (2018). *CornerNet: Detecting Objects as Paired Keypoints*. ECCV 2018. — focal heatmap loss origin
4. **SE3-DNN** — Thomas et al. (2018). *Tensor Field Networks*. arXiv:1802.08219. — equivariant kernel via TP+SH
5. **SEGNN** — Brandstetter et al. (2022). *Geometric and Physical Quantities Improve E(3) Equivariant Message Passing*. ICLR 2022. — graph-level equivariant networks
6. **DeepFindET** — Moebel et al. (2021). *Deep learning improves macromolecule identification in 3D cellular cryo-electron tomograms*. Nature Methods. — cryo-ET detection baseline
7. **CZII Dataset** — Peck, Yu, Schwartz et al. (2024). *CryoET Object Identification Challenge*. bioRxiv 10.1101/2024.11.04.621686
