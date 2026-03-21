## Parent PRD

pose/docs/prd/tensorboard_restructure.md

## What to build

Add absolute MPJPE metric (using predicted pelvis) to both training and validation, and add epoch-averaged training MPJPE scalars so train/val can be compared on the same TensorBoard plot.

End-to-end:
- Head `loss()` computes `mpjpe_abs` by unprojecting predicted pelvis via crop K, reconstructing absolute joints, and comparing to GT absolute joints
- New epoch-averaging hook accumulates per-batch `mpjpe` and `mpjpe_abs`, writes `mpjpe/rel/train` and `mpjpe/abs/train` at epoch end
- `BedlamMPJPEMetric` stores predicted pelvis + crop K during `process()`, reconstructs absolute positions in `compute_metrics()`, returns `mpjpe/abs/val`
- TensorBoard shows train and val on the same `mpjpe/rel/` and `mpjpe/abs/` plots

### Unprojection formula (BEDLAM2 convention)

```
u_px = (pelvis_uv[0] + 1) / 2 * crop_w
v_px = (pelvis_uv[1] + 1) / 2 * crop_h
X = pelvis_depth
Y = -(u_px - cx) * X / fx
Z = -(v_px - cy) * X / fy
pelvis_abs = [X, Y, Z]
abs_joints = rel_joints + pelvis_abs
```

### Key constraint

Training MPJPE is computed with dropout on and augmentation applied. This is acceptable — purpose is gap monitoring, not exact measurement.

**Status: COMPLETE** (commit 1800b43)

## Acceptance criteria

- [x] Head `loss()` returns `mpjpe_abs` (absolute MPJPE in mm, no gradient) alongside existing `mpjpe`
- [x] Absolute MPJPE uses crop K from `data_sample.metainfo['K']` for unprojection
- [x] New epoch-averaging hook registered in config
- [x] Hook writes `mpjpe/rel/train` and `mpjpe/abs/train` to TensorBoard at each epoch end
- [x] Hook resets accumulators at epoch start
- [x] `BedlamMPJPEMetric.process()` stores predicted pelvis_depth, pelvis_uv, and crop K
- [x] `BedlamMPJPEMetric.compute_metrics()` returns `mpjpe/abs/val` (all joints only, no body/hand split)
- [x] Training runs for 2+ epochs without errors
- [x] TensorBoard `mpjpe/rel/` panel shows both train and val curves
- [x] TensorBoard `mpjpe/abs/` panel shows both train and val curves
- [x] Absolute MPJPE values are larger than root-relative (sanity check)

## Blocked by

- Blocked by Issue 1 (tag restructure must be in place)

## User stories addressed

- User story 1: train/val root-relative MPJPE on same plot
- User story 2: train/val absolute MPJPE on same plot
- User story 3: absolute MPJPE uses predicted pelvis
- User story 4: unprojection via crop K matches inference pipeline
- User story 8: epoch-averaged from per-batch values, no extra forward pass
- User story 9: val body/hand breakdown preserved (from Slice 1)
- User story 12: absolute MPJPE reports all-joints only
