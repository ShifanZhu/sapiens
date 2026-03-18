# PRD: Restructure TensorBoard Logging — Unified Train/Val MPJPE + Absolute MPJPE

## Problem Statement

The current TensorBoard logging for BEDLAM2 3D pose training makes it difficult to monitor overfitting and model quality:

1. **No apples-to-apples train/val comparison.** Training logs per-iteration loss values, while validation logs per-epoch MPJPE from the evaluator. These are different metrics at different granularities, making it impossible to directly compare train vs val in TensorBoard.

2. **No absolute MPJPE metric.** The current MPJPE is root-relative only (pelvis subtracted). There is no metric that captures the full pipeline accuracy including pelvis localization — i.e., how well the model places joints in absolute camera space using the predicted pelvis.

3. **Disorganized TensorBoard tags.** Loss keys (`loss_joints`, `loss_depth`, `loss_uv`), metrics (`bedlam/mpjpe/all`, etc.), and videos are scattered without consistent grouping, making the TensorBoard UI cluttered.

## Solution

Restructure all TensorBoard scalar tags into a clean hierarchy and add two new capabilities:

1. **Epoch-averaged training MPJPE.** Accumulate the per-batch root-relative and absolute MPJPE values computed during training (inside the head's `loss()` method), average them at epoch end, and log as `mpjpe/rel/train` and `mpjpe/abs/train`.

2. **Absolute MPJPE metric.** Compute absolute MPJPE by reconstructing absolute joint positions from predicted root-relative joints + predicted pelvis (depth + UV unprojected via crop K), and comparing against GT absolute joints. Available for both train (epoch-averaged) and val (evaluator).

3. **Unified tag scheme.** All tags grouped so TensorBoard auto-organizes into logical panels.

### Final tag scheme

| Tag | Source | Granularity |
|-----|--------|-------------|
| `loss/joints/train` | head `loss()` | per-iter |
| `loss/depth/train` | head `loss()` | per-iter |
| `loss/uv/train` | head `loss()` | per-iter |
| `mpjpe/rel/train` | head `loss()`, epoch-averaged | per-epoch |
| `mpjpe/rel/val` | `BedlamMPJPEMetric` | per-epoch |
| `mpjpe/abs/train` | head `loss()`, epoch-averaged | per-epoch |
| `mpjpe/abs/val` | `BedlamMPJPEMetric` | per-epoch |
| `mpjpe/body/val` | `BedlamMPJPEMetric` | per-epoch |
| `mpjpe/hand/val` | `BedlamMPJPEMetric` | per-epoch |
| `video/...` | vis hook | per-epoch |
| `lr`, `grad_norm`, etc. | MMEngine (unchanged) | per-iter |

Checkpoint `save_best` and early stopping `monitor` updated to `mpjpe/body/val`.

## User Stories

1. As a researcher, I want to see train and val root-relative MPJPE on the same TensorBoard plot, so that I can monitor overfitting at a glance.
2. As a researcher, I want to see train and val absolute MPJPE on the same TensorBoard plot, so that I can assess full pipeline accuracy including pelvis localization.
3. As a researcher, I want the absolute MPJPE to use the predicted pelvis (not GT), so that it reflects true inference-time accuracy.
4. As a researcher, I want absolute MPJPE computed by unprojecting predicted pelvis_uv + pelvis_depth through crop K to recover 3D pelvis, then adding root-relative joints, so that the metric matches the actual inference pipeline.
5. As a researcher, I want loss tags grouped under `loss/` with a `/train` suffix, so that TensorBoard auto-groups them into a clean panel.
6. As a researcher, I want all MPJPE variants grouped under `mpjpe/`, so that I can compare them side-by-side in one TensorBoard panel.
7. As a researcher, I want video tags prefixed with `video/`, so that they don't clutter the scalar panels.
8. As a researcher, I want the epoch-averaged training MPJPE to be computed from the per-batch values already calculated in the head's `loss()` method, so that no extra forward pass is needed.
9. As a researcher, I want val body/hand MPJPE breakdown preserved, so that I can track which joint groups are improving.
10. As a researcher, I want `save_best` and early stopping to use `mpjpe/body/val`, so that the best checkpoint is selected by val body MPJPE.
11. As a researcher, I want MMEngine-managed scalars (lr, grad_norm, data_time, time) left unchanged, so that there is no risk of breaking MMEngine internals.
12. As a researcher, I want the absolute MPJPE for val to report only the all-joints aggregate (no body/hand split), since the pelvis error is shared across all joints and the split adds noise.

## Implementation Decisions

### Modules to modify

1. **Head `loss()` method** (`Pose3dRegressionHead` and `Pose3dTransformerHead` if applicable):
   - Rename returned loss keys: `loss_joints` → `loss/joints/train`, `loss_depth` → `loss/depth/train`, `loss_uv` → `loss/uv/train`
   - Add absolute MPJPE computation: unproject pred pelvis_uv + pelvis_depth via crop K (from `batch_data_samples` metainfo), reconstruct absolute pred and GT joints, compute MPJPE in mm
   - Return both `mpjpe` (root-relative) and `mpjpe_abs` (absolute) for logging
   - Crop K accessed via `data_sample.metainfo['K']`; crop dimensions are fixed (640x384 from config, or read from input tensor shape)

2. **Epoch-averaging hook** (new custom hook or extend existing):
   - Accumulate per-batch `mpjpe` and `mpjpe_abs` values during training
   - At epoch end, compute averages and write `mpjpe/rel/train` and `mpjpe/abs/train` to TensorBoard
   - Reset accumulators at epoch start

3. **`BedlamMPJPEMetric`**:
   - Store predicted pelvis_depth, pelvis_uv, and crop K alongside pred/gt joints in `process()`
   - In `compute_metrics()`, reconstruct absolute positions and compute absolute MPJPE
   - Change `default_prefix` to `''` (empty) or remove it
   - Return keys: `mpjpe/rel/val`, `mpjpe/abs/val`, `mpjpe/body/val`, `mpjpe/hand/val`

4. **Config file**:
   - `save_best='mpjpe/body/val'`
   - `monitor='mpjpe/body/val'`
   - Video tag prefix updated to `video/`

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

Where fx, fy, cx, cy come from crop K.

### Key constraint

Training MPJPE values are computed with dropout on and augmentation applied, so they won't be identical to a clean evaluation pass. This is acceptable — the purpose is overfitting detection (gap monitoring), not exact measurement.

## Testing Decisions

- **No new test files required.** The changes are to logging/metrics, not core model logic.
- If tests exist for `BedlamMPJPEMetric`, update them to reflect new return key names.
- Manual verification: run a short training (2-3 epochs) and check TensorBoard for correct tag structure and reasonable values.

## Out of Scope

- Running full evaluator on the training set (too expensive)
- Adding val-time losses
- Restructuring MMEngine-managed tags (lr, grad_norm, etc.)
- Body/hand breakdown for absolute MPJPE
- Changes to the visualization hook logic (only tag prefix changes)
- Changes outside the `pose/` directory

## Further Notes

- The `mpjpe` key returned by the head's `loss()` is a non-gradient tensor used only for logging. MMEngine's runner logs all keys from the loss dict automatically.
- For the epoch-averaging hook, MMEngine's message hub or a simple list accumulator can be used. The hook should use `after_train_iter` to accumulate and `after_train_epoch` to flush.
- The pelvis recovery logic already exists in the visualization hook (`_bedlam2_recover_pelvis`). The same formula should be used in the head and metric for consistency.
