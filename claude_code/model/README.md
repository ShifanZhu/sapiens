# SapiensPose3D — Model Structure

## Overview

```
Input: (B, 4, H, W)          RGB (3ch) + Depth (1ch), e.g. H=384, W=640
         │
    ┌────▼────────────────────────────────────────────┐
    │  SapiensBackboneRGBD  (304.3M params)           │
    │                                                  │
    │  Patch Embedding                                 │
    │  Conv2d(4→1024, 16×16, stride=16, pad=2)        │
    │  → (B, 1024, 24, 40)                            │
    │                                                  │
    │  ViT Transformer  ×24 blocks                    │
    │  Each block:                                     │
    │    LayerNorm → Multi-Head Self-Attention         │
    │    LayerNorm → FFN (1024→4096→1024)              │
    │  → (B, 1024, 24, 40)                            │
    │                                                  │
    │  Final LayerNorm                                 │
    │  → (B, 1024, 24, 40)   feature map              │
    └────────────────────┬────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────┐
    │  Pose3DHead  (2.9M params)                       │
    │                                                  │
    │  AdaptiveAvgPool2d(1)   → (B, 1024)             │
    │  Linear(1024 → 2048)                            │
    │  LayerNorm + GELU                               │
    │  Linear(2048 → 127×3)                           │
    │  reshape → (B, 127, 3)                          │
    └────────────────────┬────────────────────────────┘
                         │
Output: (B, 127, 3)      Camera-space XYZ (X=fwd, Y=left, Z=up), metres
```

## Key Design Choices

| Aspect | Choice | Reason |
|---|---|---|
| Input fusion | Early fusion — concat RGB+D before patch embed | Simple; lets ViT learn cross-modal features from scratch |
| Depth channel init | Mean of 3 RGB pretrained channels | Warm start; avoids random noise disrupting pretrained features |
| Pos embed | Bicubic interpolation 64×64 → 24×40 | Adapts 1024×1024 pretrain to our 384×640 input |
| Head | GAP + MLP | Lightweight; suitable baseline for 3D regression |
| Output space | Camera-space XYZ (metres) | Directly supervised against BEDLAM2 `joints_cam` labels |

## Pretrained Weight Loading

```
checkpoints/sapiens_0.3b_epoch_1600_clean.pth  (1.2 GB)
Pretrained on 300M in-the-wild human images at 1024×1024.

Conversions applied at load time:
  1. Key remap    : flat keys → add 'backbone.vit.' prefix
  2. patch_embed  : (1024, 3, 16, 16) → (1024, 4, 16, 16)
                    depth channel = mean of 3 RGB channels
  3. pos_embed    : (1, 4097, 1024) 64×64+CLS → (1, 960, 1024) 24×40
                    bicubic interpolation, CLS token dropped
  4. cls_token    : dropped  (with_cls_token=False)
  5. head         : randomly initialised (new task-specific module)

Result: 293/293 backbone tensors loaded, 0 missing.
```

## Model Variants

| Variant | Embed dim | ViT blocks | Backbone params | Total params |
|---|---|---|---|---|
| `sapiens_0.3b` | 1024 | 24 | 304 M | 307 M |
| `sapiens_0.6b` | 1280 | 32 | 618 M | 621 M |
| `sapiens_1b`   | 1536 | 40 | 982 M | 985 M |
| `sapiens_2b`   | 1920 | 48 | 1.9 B | 1.9 B |

Start with **0.3b** for fast iteration.

## File Reference

| File | Description |
|---|---|
| `backbone.py` | `SapiensBackboneRGBD` — mmpretrain ViT with `in_channels=4` |
| `head.py` | `Pose3DHead` — GAP + LayerNorm MLP regression head |
| `sapiens_pose3d.py` | `SapiensPose3D` — combines backbone + head |
| `weights.py` | `load_sapiens_pretrained()` — handles all weight conversions |

## Coordinate Convention (BEDLAM2)

- Camera space: **X = forward (depth), Y = left, Z = up**, right-handed, unit = metres
- Projection:  `u = fx·(-Y/X) + cx`,  `v = fy·(-Z/X) + cy`
- Label intrinsic `K` and `joints_2d` are always in the upright (post-rotation) image frame
