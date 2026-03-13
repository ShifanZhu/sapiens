# Training & Inference Pipeline

## Overview

This project fine-tunes a Sapiens ViT backbone (pretrained on 300M human images) to predict 3D human poses from RGB + depth input. The model outputs **root-relative** joint positions plus **pelvis localization** (depth and 2D position), enabling multi-person 3D pose estimation via a top-down pipeline.

```
Training:
  BEDLAM2 labels → Dataset (per-person, per-frame) → Transforms → Model → Multi-task Loss

Inference:
  RGB + Depth → Detect people → Crop each → Model → Root-relative joints + pelvis → Absolute 3D
```

---

## 1. Data (`data/`)

### Source: BEDLAM2

Synthetic dataset of humans in diverse scenes with ground-truth annotations.

- ~1800 sequences, 24-96 frames each at 30fps (downsampled to 6fps, stride=5)
- Each sequence NPZ label contains:
  - `joints_cam`: `(n_body, n_frames, 127, 3)` — 3D joints in camera space (metres)
  - `joints_2d`: `(n_body, n_frames, 127, 2)` — projected 2D positions (pixels)
  - `intrinsic_matrix`: `(3, 3)` — camera calibration
  - `folder_name`, `seq_name`, `n_frames`, `rotate_flag`

### Coordinate System (BEDLAM2 camera space)

**Non-standard** convention — differs from OpenCV:

| Axis | BEDLAM2 | OpenCV |
|------|---------|--------|
| X    | **Forward** (depth) | Right |
| Y    | Left | Down |
| Z    | Up | **Forward** (depth) |

Projection equations:
```
u = fx * (-Y / X) + cx
v = fy * (-Z / X) + cy
```

### Depth Maps

BEDLAM2 depth maps store **forward distance** (planar/Z-buffer depth), which is the X coordinate in BEDLAM2's camera space. This was verified empirically: at 236 surface-joint samples, `depth_map_value / X_forward` has median = 0.994, vs `depth_map_value / euclidean_distance` has median = 0.827.

This is consistent with rendering engines (Blender/Unreal) that output Z-buffer depth.

**Real depth cameras** vary:
- Structured light (RealSense D400) and stereo (ZED) → forward distance
- Time-of-Flight (iPhone LiDAR, some Azure Kinect modes) → Euclidean distance
- Conversion: `Z_forward = Z_euclidean * cos(angle_from_optical_axis)`
- At inference time, ensure depth input matches BEDLAM2 convention (forward distance)

### Dataset Indexing (`dataset.py`)

The flat index is `(label_path, body_idx, frame_idx)`. For a multi-person sequence with 3 bodies and 50 frames, that's 150 samples. Each sample is one person in one frame.

Samples with tiny bounding boxes (< 32px) are retried with a random different index.

### Splits (`splits.py`)

Sequence-level splits (not frame-level) to prevent data leakage. Seed=2026.

---

## 2. Transform Pipeline (`data/transforms.py`)

### Training: `NoisyBBox → CropPerson → SubtractRoot → ToTensor`

### Validation: `CropPerson → SubtractRoot → ToTensor`

### Step A: `NoisyBBox` (train only)

Simulates an imperfect person detector. Randomly jitters:
- Center position: ±10% of box size
- Scale: ±15%

Result clamped to image bounds. Skipped if jitter produces a box < 2px.

### Step B: `CropPerson`

Crops the image to the person bounding box and resizes to 640x384.

1. Expand bbox to match target aspect ratio (640:384 = 5:3)
2. Pad with zeros if expanded box extends beyond image bounds
3. Crop and resize RGB (bilinear) and depth (nearest — avoids edge bleeding)
4. **Update intrinsic K** to maintain geometric consistency:
   ```
   fx' = fx * sx         fy' = fy * sy
   cx' = (cx - x0) * sx  cy' = (cy - y0) * sy
   ```
   where `sx = out_w / crop_w`, `x0` = crop origin in original image pixels

If no `bbox` key exists (backward compat), falls back to plain `Resize`.

### Step C: `SubtractRoot`

Makes predictions translation-invariant by converting to root-relative coordinates.

1. Save pelvis absolute position: `pelvis_abs = joints[0].copy()`
2. Subtract pelvis from all joints: `joints -= pelvis` (pelvis becomes origin)
3. Compute GT supervision targets for pelvis localization:
   - `pelvis_depth = pelvis_abs[0]` — forward distance in **raw metres** (X coordinate, NOT normalized). Typical range 1-10m.
   - `pelvis_uv` — project pelvis through **crop K** (`u_px = fx*(-Y/X)+cx`, `v_px = fy*(-Z/X)+cy`), then **normalize to [-1, 1]**:
     ```
     u_norm = u_px / crop_w * 2 - 1
     v_norm = v_px / crop_h * 2 - 1
     ```
     (0, 0) = crop center. The pelvis is typically near the center of the person crop, so values cluster around 0, which is friendly for linear layers with zero-initialized bias.

Must run **after** `CropPerson` so that K is the crop K and `rgb` has the crop dimensions.

**Unit design:** All three regression targets are now in similar numeric ranges:
- `joints`: root-relative metres, typically ±0.5
- `pelvis_depth`: raw metres, typically 1-10
- `pelvis_uv`: normalized [-1, 1], typically ±0.3

This avoids the need for aggressive loss down-weighting. All lambdas default to 1.0 and all SmoothL1 betas are 0.05. The depth *image* (model input channel 3) is separately normalized to [0, 1], but that is unrelated to the regression targets.

**Denormalization at inference:** To recover crop pixels from model output: `u_px = (u_norm + 1) / 2 * crop_w`.

### Step D: `ToTensor`

| Field | Before | After |
|-------|--------|-------|
| `rgb` | `(H,W,3) uint8` | `(3,H,W) float32`, `/255`, ImageNet mean/std normalized |
| `depth` | `(H,W) float32` metres | `(1,H,W) float32`, clipped to [0, 20m], `/20` → **[0, 1] unitless** |
| `joints` | `(127,3) float32` metres | `(127,3) float32` tensor, root-relative **metres** |
| `intrinsic` | `(3,3) float32` | `(3,3) float32` tensor |
| `pelvis_depth` | `(1,) float32` metres | `(1,) float32` tensor, **raw metres** (not normalized) |
| `pelvis_uv` | `(2,) float32` normalized | `(2,) float32` tensor, **[-1, 1]** (0 = crop center) |

Note: the depth *image* is normalized to [0,1] for the model input, but `pelvis_depth` stays in raw metres as a regression target. `pelvis_uv` is normalized to [-1, 1] so all outputs are in similar numeric ranges.

---

## 3. Model Architecture (`model/`)

### Input

`(B, 4, H, W)` where channels 0-2 are ImageNet-normalized RGB and channel 3 is depth normalized to [0, 1] (unitless).

Constructed by: `x = torch.cat([rgb, depth], dim=1)`

### Output units

| Output | Shape | Unit |
|--------|-------|------|
| `joints` | `(B, 127, 3)` | Root-relative, **metres** |
| `pelvis_depth` | `(B, 1)` | Forward distance, **raw metres** |
| `pelvis_uv` | `(B, 2)` | Pelvis position, **normalized [-1, 1]** (0 = crop center) |

The model input depth is normalized [0,1], but the model **predicts** pelvis_depth in raw metres. There is no explicit connection between the two — the network learns the mapping internally.

### Backbone (`backbone.py`)

Sapiens ViT with 4-channel patch embedding (instead of 3). Architecture configs:

| Arch | embed_dim | layers | params |
|------|-----------|--------|--------|
| sapiens_0.3b | 1024 | 24 | ~300M |
| sapiens_0.6b | 1280 | 32 | ~600M |
| sapiens_1b | 1536 | 40 | ~1B |
| sapiens_2b | 1920 | 48 | ~2B |

- Patch size = 16x16
- Input 640x384 → 40x24 grid = 960 tokens
- No CLS token, `out_type="featmap"`
- Output: `(B, embed_dim, 40, 24)`

### Head (`head.py`)

Shared-trunk MLP with 3 task-specific output branches:

```
(B, embed_dim, 40, 24)
  → AdaptiveAvgPool2d(1)                    → (B, embed_dim)
  → Linear(embed_dim, 2048) + LN + GELU + Dropout   [shared trunk]
  ├→ Linear(2048, 127*3) → reshape           → joints      (B, 127, 3)
  ├→ Linear(2048, 1)                         → pelvis_depth (B, 1)
  └→ Linear(2048, 2)                         → pelvis_uv    (B, 2)
```

Returns a dict: `{"joints": ..., "pelvis_depth": ..., "pelvis_uv": ...}`

### Pretrained Weight Loading (`weights.py`)

Three conversions when loading a Sapiens RGB-pretrained checkpoint:

1. **Key prefix**: add `backbone.vit.` to all keys
2. **Patch embed expansion**: `(C, 3, 16, 16)` → `(C, 4, 16, 16)`, depth channel = mean(RGB)
3. **Pos embed interpolation**: `(1, 4097, D)` [64x64 + CLS] → `(1, 960, D)` [40x24, no CLS], bicubic

The head is left randomly initialized.

---

## 4. Training Loop (`train.py`)

### Optimizer

AdamW with two param groups:
- Backbone: lr = 1e-5 (pretrained, fine-tune slowly)
- Head: lr = 1e-4 (random init, learn fast)
- Weight decay = 0.03

### LR Schedule

Linear warmup (3 epochs) → cosine decay to 0.

### Loss

Multi-task loss with configurable weights:

```
L_total = L_pose + λ_depth * L_depth + λ_uv * L_uv
```

| Component | Formula | Default weight | Beta |
|-----------|---------|----------------|------|
| `L_pose` | SmoothL1(pred_joints, gt_joints_rel) | 1.0 | 0.05m (5cm) |
| `L_depth` | SmoothL1(pred_depth, gt_pelvis_depth) | λ_depth = 1.0 | 0.05m |
| `L_uv` | SmoothL1(pred_uv, gt_pelvis_uv) | λ_uv = 1.0 | 0.05 |

SmoothL1 with beta: L2 below beta (smooth gradient for small errors), L1 above (robust to outliers).

All three targets are in similar numeric ranges (metres ~1-5, normalized UV ~±0.3), so all lambdas default to 1.0 and all betas to 0.05.

### Metrics

MPJPE (Mean Per-Joint Position Error) in mm:
- **Body** (joints 0:22): core kinematic joints — used for best model selection and early stopping
- **Hand** (joints 25:55): left + right hand joints
- **All** (joints 0:127): all SMPL-X joints

### Mixed Precision

AMP enabled by default (float16 forward/backward, float32 optimizer). Disable with `--no-amp`.

### Checkpointing

- `best.pth`: lowest val MPJPE (body)
- `epoch_XXXX.pth`: every N epochs (default 5)
- Early stopping: configurable patience (default 5 val checks without improvement)

---

## 5. Inference Pipeline (`scripts/demo_multiperson.py`)

Top-down multi-person pipeline: detect people, process each independently.

### Step 1: Load Frame

```
RGB:   JPEG frame (H, W, 3) uint8
Depth: NPY mmap or NPZ (H, W) float32 metres
```

### Step 2: Get Person Bounding Boxes

Currently: GT bboxes from labels (computed from joints_2d with 10% padding).
Production: person detector (YOLOv8, etc.).

### Step 3: Per-Person Crop + Forward Pass

For each person:

```python
# Same crop logic as training CropPerson
rgb_crop, depth_crop, K_crop = crop_person(rgb, depth, K_orig, bbox, 640, 384)

# Normalize: ImageNet for RGB, [0,1] for depth
x = normalize_for_model(rgb_crop, depth_crop)  # (4, 640, 384)

# Forward pass
out = model(x.unsqueeze(0))
pred_rel   = out["joints"][0]         # (127, 3) root-relative
pred_depth = out["pelvis_depth"][0,0] # scalar — forward distance in metres
pred_uv    = out["pelvis_uv"][0]      # (2,) — normalized [-1, 1]
```

### Step 4: Recover Absolute 3D Pelvis

The model predicts root-relative poses. To place each person in the scene:

```
1. pred_uv is normalized [-1, 1]. Denormalize to crop pixels:
   u_crop = (u_norm + 1) / 2 * crop_w
   v_crop = (v_norm + 1) / 2 * crop_h

2. Invert the crop transform to get original image pixel:
   u_orig = u_crop / sx + x0
   v_orig = v_crop / sy + y0
   (sx, sy, x0, y0 are the crop-resize parameters)

3. Use pred_depth as X (forward distance in metres)

4. Unproject with original K (BEDLAM2 convention):
   Y = -(u_orig - cx) * X / fx    (left)
   Z = -(v_orig - cy) * X / fy    (up)

5. pelvis_abs = [X, Y, Z]
```

### Step 5: Absolute Skeleton

```python
joints_abs = pred_rel + pelvis_abs[np.newaxis, :]  # (127, 3)
```

### Step 6: Visualize

Project absolute joints to 2D via original K, draw skeleton on the original image. Each person gets a distinct color.

---

## 6. Why Root-Relative + Pelvis Recovery?

Predicting absolute 3D positions directly is harder because:

1. **Scale ambiguity**: the same pose at 2m vs 5m looks identical in relative coords but very different in absolute coords
2. **High variance**: absolute depth ranges 1-20m while relative joint offsets are ±0.5m
3. **Clean factorization**: "what pose" (joints branch) is separated from "where in the scene" (pelvis_depth + pelvis_uv branches)

The pelvis_uv branch learns where the pelvis projects in the crop image. The pelvis_depth branch learns the forward distance. Together they fully determine the 3D pelvis position, which anchors all root-relative joints into absolute camera space.
