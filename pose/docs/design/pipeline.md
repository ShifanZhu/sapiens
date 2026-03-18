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
  - `joints_cam`: `(n_body, n_frames, 127, 3)` — 3D joints in camera space (metres); 70-joint active subset selected at load time
  - `joints_2d`: `(n_body, n_frames, 127, 2)` — projected 2D positions (pixels); all 127 used for OOB visibility filter
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

### Active Joint Subset

The raw BEDLAM2 labels contain 127 SMPL-X joints. We use only a **70-joint active subset** that excludes the dense face mesh:

| Group | Original indices | Active indices | Count |
|-------|-----------------|----------------|-------|
| Body (pelvis → wrists) | 0-21 | 0-21 | 22 |
| Eyes (left_eye_smplhf, right_eye_smplhf) | 23-24 | 22-23 | 2 |
| Hands (left + right) | 25-54 | 24-53 | 30 |
| Non-face surface (toes, heels, fingertips) | 60-75 | 54-69 | 16 |

**Excluded** (57 joints): jaw (22), nose/eye/ear surface (55-59), eyebrows (76-85), nose mesh (86-94), eye mesh (95-106), mouth (107-118), lips (119-126).

The active joint subset is defined in `constants.ACTIVE_JOINT_INDICES` and applied in `dataset.py` at load time. The model head outputs 70 joints.

### Dataset Indexing (`dataset.py`)

The flat index is `(label_path, body_idx, frame_idx)`. For a multi-person sequence with 3 bodies and 50 frames, that's 150 samples. Each sample is one person in one frame.

**Filtering** — a sample is skipped (retried with a random index) if:
1. Bounding box is smaller than 32×32 px.
2. More than **70%** of the 127 raw joints have a 2D projection outside the image (x < 0, x ≥ W, y < 0, or y ≥ H). This removes frames where the person is mostly off-screen.

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
| `joints` | `(70,3) float32` metres | `(70,3) float32` tensor, root-relative **metres** |
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
| `joints` | `(B, 70, 3)` | Root-relative, **metres** (active joint subset) |
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

> **Status: NOT YET IMPLEMENTED.** The transformer decoder design below is the planned replacement for the current global-average-pooling MLP head. The current code still uses the old head (AdaptiveAvgPool2d → shared MLP → 3 branches). Implementation is planned for a future session.

Transformer decoder head with per-joint query tokens and single-layer linear output projections.

#### Why not global average pooling?

The current head collapses the spatial feature map via `AdaptiveAvgPool2d(1)` into a single vector before regressing 210 coordinates (70 joints × 3). This **destroys all spatial information** — the network cannot know *where* in the image each joint appears. The transformer decoder approach preserves spatial structure: each joint query attends to the relevant image region (e.g., the "left knee" query looks at the left knee area of the feature map).

#### Baseline (current GAP+MLP head)

| Model | Body MPJPE | Hand MPJPE | All MPJPE |
|-------|-----------|-----------|----------|
| sapiens_0.3b | 80.6 mm | 130.5 mm | 117.6 mm |
| sapiens_2b | 70.5 mm | 113.7 mm | 102.1 mm |

Hands are dramatically worse than body — consistent with spatial information loss from GAP, since hands are small, spatially localized, and span only 1-2 patches. However, other hypotheses (patch resolution, data distribution, occlusion) have not been ruled out.

**Success criterion:** body MPJPE ≤ 75mm on 0.3b (≥5mm improvement, ~7% relative).

#### Architecture

```
Backbone output: (B, embed_dim, 40, 24)

Step 1 — Prepare spatial tokens:
  Flatten to (B, 960, embed_dim)
  Add 2D sine/cosine positional encoding (DETR-style)     ← encodes (row, col) position
  Result: spatial_tokens (B, 960, embed_dim)

Step 2 — Joint query tokens:
  70 learnable query embeddings (70, embed_dim), broadcast to (B, 70, embed_dim)

Step 3 — Transformer decoder (1 layer):
  (a) Self-attention over 70 query tokens                  ← learns implicit kinematics
  (b) Cross-attention: queries attend to spatial_tokens     ← each joint finds its image region
  (c) FFN: Linear → GELU → Dropout → Linear + residual
  Output: (B, 70, embed_dim)

Step 4 — Joint output:
  Linear(embed_dim, 3) applied per token (shared weights)
  Output: joints (B, 70, 3) — root-relative metres

Step 5 — Pelvis branches (from pelvis query token, index 0):
  decoder_out[:, 0, :]  (B, embed_dim)
  ├→ Linear(embed_dim, 1)    → pelvis_depth (B, 1)
  └→ Linear(embed_dim, 2)    → pelvis_uv    (B, 2)
```

Returns a dict: `{"joints": ..., "pelvis_depth": ..., "pelvis_uv": ...}`

#### Design decisions

**Start minimal, add complexity only with evidence.** We use 1 decoder layer (not 2+) and single linear output projections (not multi-layer MLPs). This isolates the impact of cross-attention itself. If results are insufficient, we can add depth (more layers, larger MLPs) and know exactly what each change bought.

**2D positional encodings on spatial tokens.** Flattening the feature map from `(B, 1024, 40, 24)` to `(B, 960, 1024)` loses explicit spatial structure. Although the ViT backbone's positional embeddings leave implicit position information in the features, adding DETR-style 2D sine/cosine encodings gives cross-attention a clean, explicit spatial signal. Zero learnable parameters, negligible compute, no downside.

**Self-attention enables implicit kinematics.** The decoder layer runs self-attention over the 70 query tokens *before* cross-attention. This lets joints exchange information with anatomically related joints — e.g., the "left hand" query learns to coordinate with the "left elbow" query, producing anatomically plausible predictions without an explicit kinematic tree.

**Multi-head attention (8 heads).** With `embed_dim=1024`, each of the 8 heads has 128 dimensions. Different heads can specialize — e.g., when locating a knee, one head might attend to the visible kneecap while another attends to the thigh angle to infer depth.

**Pelvis branches from query token 0.** The pelvis is joint index 0 among the 70 queries. Its decoded representation has already attended to the pelvis region via cross-attention, so it is the natural source for pelvis depth and UV prediction. This is simpler and more semantically coherent than mean-pooling all 70 query outputs (which mixes signals from unrelated body parts).

**Joint queries predict root-relative directly.** Ground truth is root-subtracted in the `SubtractRootJoint` data transform (before the model). The joint queries directly output root-relative coordinates — no explicit subtraction inside the head. This preserves the clean factorization: "what pose" (joint queries) vs "where in the scene" (pelvis branches from query 0).

**Single linear output projections.** The decoder's cross-attention already routes spatial information to each query token. A single `Linear(1024, 3)` per token should suffice for mapping to 3D coordinates — DETR uses similarly lightweight output heads. If this proves insufficient, adding MLP depth is a straightforward follow-up.

#### Parameter count (sapiens_0.3b, embed_dim=1024)

| Component | Parameters |
|-----------|-----------|
| 2D positional encoding | 0 (fixed sine/cosine) |
| Joint query embeddings | 70 × 1024 = 72K |
| 1 decoder layer (self-attn + cross-attn + FFN) | ~12.6M |
| Joint projection Linear(1024, 3) | ~3K |
| Pelvis depth Linear(1024, 1) + UV Linear(1024, 2) | ~3K |
| **Total head** | **~13M** |

This replaces the old head (~4M params) but remains small relative to the 300M backbone.

#### Training plan

Train from scratch (pretrained Sapiens backbone, randomly initialized head) with the same recipe as baseline: AdamW, 1e-4 head / 1e-5 backbone, 3-epoch linear warmup, cosine decay, 50 epochs. Purely a head swap — no changes to backbone, data pipeline, estimator, or loss functions.

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

Linear warmup (3 epochs, `by_epoch=True`) → cosine annealing decay to 0 (`eta_min=0`).

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
- **Body** (active indices 0:22): core kinematic joints — used for best model selection and early stopping
- **Hand** (active indices 24:54): left + right hand joints (was 25:55 in raw 127-joint space)
- **All** (active indices 0:70): all active joints

### Mixed Precision

AMP enabled by default (float16 forward/backward, float32 optimizer). Disable with `--no-amp`.
Uses `AmpOptimWrapper` with dynamic loss scaling.

### Checkpointing

- `best.pth`: lowest val MPJPE (body) — selected via `save_best='bedlam/mpjpe/body'`
- `epoch_XXXX.pth`: every 5 epochs (`interval=5` in `CheckpointHook`)
- Early stopping via `EarlyStoppingHook`: patience=5, monitors `bedlam/mpjpe/body`

---

## 5. Inference Pipeline (`demo/demo_bedlam2.py`)

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
pred_rel   = out["joints"][0]         # (70, 3) root-relative
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
joints_abs = pred_rel + pelvis_abs[np.newaxis, :]   # (70, 3)
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
