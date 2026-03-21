# Training Loop

Optimizer, LR schedule, loss, metrics, and checkpointing configuration.

See also: [pipeline.md](pipeline.md) (model architecture), [data_transforms.md](data_transforms.md) (data pipeline).

---

## Optimizer

AdamW with two param groups:
- Backbone: lr = 1e-5 (pretrained, fine-tune slowly)
- Head: lr = 1e-4 (random init, learn fast)
- Weight decay = 0.03

## LR Schedule

Linear warmup (3 epochs, `by_epoch=True`) → cosine annealing decay to 0 (`eta_min=0`).

## Loss

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

## Metrics

MPJPE (Mean Per-Joint Position Error) in mm:
- **`mpjpe/body/val`** (active indices 0:22): core kinematic joints — used for best model selection and early stopping
- **`mpjpe/hand/val`** (active indices 24:54): left + right hand joints
- **`mpjpe/rel/val`** (active indices 0:70): all active joints, root-relative
- **`mpjpe/abs/val`**: absolute MPJPE after recovering pelvis position via unprojection

Training epoch averages (logged by `TrainMPJPEAveragingHook`):
- **`mpjpe/rel/train`**: epoch-averaged root-relative MPJPE (train set, with augmentation)
- **`mpjpe/abs/train`**: epoch-averaged absolute MPJPE (train set)

## TensorBoard Tag Hierarchy

```
loss/joints/train    — per-iteration joint SmoothL1 loss
loss/depth/train     — per-iteration pelvis depth loss
loss/uv/train        — per-iteration pelvis UV loss
mpjpe/rel/train      — epoch-averaged root-relative MPJPE (train)
mpjpe/abs/train      — epoch-averaged absolute MPJPE (train)
mpjpe/rel/val        — validation root-relative MPJPE (all joints)
mpjpe/body/val       — validation body MPJPE
mpjpe/hand/val       — validation hand MPJPE
mpjpe/abs/val        — validation absolute MPJPE
video/{split}/{scene}/gt_pelvis   — visualization
video/{split}/{scene}/pred_pelvis — visualization
```

## Mixed Precision

AMP enabled by default (float16 forward/backward, float32 optimizer). Disable with `--no-amp`.
Uses `AmpOptimWrapper` with dynamic loss scaling.

## Checkpointing

- `best.pth`: lowest val MPJPE (body) — selected via `save_best='mpjpe/body/val'`
- `epoch_XXXX.pth`: every 5 epochs (`interval=5` in `CheckpointHook`)
- Early stopping via `EarlyStoppingHook`: patience=5, monitors `mpjpe/body/val`
