# Pipeline: Model Architecture & Inference

End-to-end overview of the BEDLAM2 RGBD 3D pose system: what the model is and how inference works.

- **Data & transforms:** [data_transforms.md](data_transforms.md)
- **Training loop (loss, metrics, optimizer):** [training_loop.md](training_loop.md)

---

## Overview

Fine-tunes a Sapiens ViT backbone (pretrained on 300M human images) to predict 3D human poses from RGB + depth input. The model outputs **root-relative** joint positions plus **pelvis localization** (depth and 2D position), enabling multi-person 3D pose estimation via a top-down pipeline.

```
Training:
  BEDLAM2 labels → Dataset (per-person, per-frame) → Transforms → Model → Multi-task Loss

Inference:
  RGB + Depth → Detect people → Crop each → Model → Root-relative joints + pelvis → Absolute 3D
```

---

## Model Architecture

### Input / Output

Input: `(B, 4, H, W)` — channels 0-2 are ImageNet-normalized RGB, channel 3 is depth normalized to [0, 1].

| Output | Shape | Unit |
|--------|-------|------|
| `joints` | `(B, 70, 3)` | Root-relative, **metres** |
| `pelvis_depth` | `(B, 1)` | Forward distance, **raw metres** |
| `pelvis_uv` | `(B, 2)` | Pelvis position, **normalized [-1, 1]** (0 = crop center) |

The model input depth is normalized [0,1], but the model **predicts** pelvis_depth in raw metres. The network learns the mapping internally.

### Backbone (`SapiensBackboneRGBD`)

Sapiens ViT with 4-channel patch embedding (instead of 3).

| Arch | embed_dim | layers | params |
|------|-----------|--------|--------|
| sapiens_0.3b | 1024 | 24 | ~300M |
| sapiens_0.6b | 1280 | 32 | ~600M |
| sapiens_1b | 1536 | 40 | ~1B |
| sapiens_2b | 1920 | 48 | ~2B |

- Patch size = 16×16; Input 640×384 → 40×24 grid = 960 tokens
- No CLS token, `out_type="featmap"`; Output: `(B, embed_dim, 40, 24)`

### Head

Two heads are available; switch via `head.type` in the config.

#### `Pose3dTransformerHead` (recommended)

Transformer decoder with per-joint query tokens and cross-attention to the spatial feature map.

```
Backbone output: (B, embed_dim, 40, 24)

Step 1 — Spatial tokens:
  Flatten → (B, 960, embed_dim)
  Add 2D sine/cosine positional encoding (DETR-style)

Step 2 — Joint query tokens:
  70 learnable query embeddings → (B, 70, embed_dim)

Step 3 — Transformer decoder (1 layer):
  (a) Self-attention over 70 query tokens     ← implicit kinematics
  (b) Cross-attention to spatial_tokens        ← each joint finds its region
  (c) FFN: Linear → GELU → Dropout → Linear + residual
  Output: (B, 70, embed_dim)

Step 4 — Outputs:
  Linear(embed_dim, 3) per token  → joints (B, 70, 3)
  Linear(embed_dim, 1) on token 0 → pelvis_depth (B, 1)
  Linear(embed_dim, 2) on token 0 → pelvis_uv (B, 2)
```

Parameter count (0.3b): ~13M head vs ~300M backbone.

See [prd/transformer_decoder_head.md](../prd/transformer_decoder_head.md) for full design rationale. See [bedlam2/training_results.md](../bedlam2/training_results.md) for A/B results.

#### `Pose3dRegressionHead` (baseline)

`AdaptiveAvgPool2d(1)` → flatten → `Linear+LN+GELU+Dropout` → three branches. Simpler and faster, but global average pooling destroys spatial information.

Baseline results:

| Model | Body MPJPE | Hand MPJPE | All MPJPE |
|-------|-----------|-----------|----------|
| sapiens_0.3b | 80.6 mm | 130.5 mm | 117.6 mm |
| sapiens_2b | 70.5 mm | 113.7 mm | 102.1 mm |

### Pretrained Weight Loading (`rgbd_weight_utils.py`)

Three conversions when loading a Sapiens RGB-pretrained checkpoint:

1. **Key prefix**: add `backbone.vit.` to all keys
2. **Patch embed expansion**: `(C, 3, 16, 16)` → `(C, 4, 16, 16)`, depth channel = mean(RGB)
3. **Pos embed interpolation**: `(1, 4097, D)` [64×64 + CLS] → `(1, 960, D)` [40×24, no CLS], bicubic

The head is left randomly initialized.

---

## Inference Pipeline (`demo/demo_bedlam2.py`)

Top-down multi-person pipeline.

### Step 1: Load Frame

```
RGB:   JPEG frame (H, W, 3) uint8
Depth: NPY mmap or NPZ (H, W) float32 metres
```

### Step 2: Get Person Bounding Boxes

Currently: GT bboxes from labels (10% padding). Production: person detector (YOLOv8, etc.).

### Step 3: Per-Person Crop + Forward Pass

```python
# Same crop logic as training CropPerson
rgb_crop, depth_crop, K_crop = crop_person(rgb, depth, K_orig, bbox, 640, 384)
x = normalize_for_model(rgb_crop, depth_crop)  # (4, 640, 384)

out = model(x.unsqueeze(0))
pred_rel   = out["joints"][0]         # (70, 3) root-relative metres
pred_depth = out["pelvis_depth"][0,0] # scalar — forward distance in metres
pred_uv    = out["pelvis_uv"][0]      # (2,) — normalized [-1, 1]
```

### Step 4: Recover Absolute 3D Pelvis

```
1. Denormalize pelvis UV to crop pixels:
   u_crop = (u_norm + 1) / 2 * crop_w
   v_crop = (v_norm + 1) / 2 * crop_h

2. Invert crop transform to original image pixels:
   u_orig = u_crop / sx + x0
   v_orig = v_crop / sy + y0

3. Unproject with original K (BEDLAM2 convention):
   X = pred_depth
   Y = -(u_orig - cx) * X / fx
   Z = -(v_orig - cy) * X / fy

4. pelvis_abs = [X, Y, Z]
```

### Step 5: Absolute Skeleton

```python
joints_abs = pred_rel + pelvis_abs[np.newaxis, :]   # (70, 3)
```

### Step 6: Visualize

Project absolute joints to 2D via original K, draw skeleton. Each person gets a distinct color.

---

## Why Root-Relative + Pelvis Recovery?

Predicting absolute 3D positions directly is harder because:

1. **Scale ambiguity**: the same pose at 2m vs 5m looks identical in relative coords but very different in absolute coords
2. **High variance**: absolute depth ranges 1-20m while relative joint offsets are ±0.5m
3. **Clean factorization**: "what pose" (joints branch) is separated from "where in the scene" (pelvis_depth + pelvis_uv branches)

The pelvis_uv branch learns where the pelvis projects in the crop image. The pelvis_depth branch learns the forward distance. Together they fully determine the 3D pelvis position, which anchors all root-relative joints into absolute camera space.
