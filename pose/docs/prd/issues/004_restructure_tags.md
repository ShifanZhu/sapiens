## Parent PRD

pose/docs/prd/tensorboard_restructure.md

## What to build

Restructure all existing TensorBoard tags into a clean hierarchy without adding new metrics. This is a rename-only change that establishes the tag naming convention for subsequent work.

End-to-end:
- Head `loss()` returns keys with `loss/` prefix and `/train` suffix
- `BedlamMPJPEMetric` returns keys under `mpjpe/` with `/val` suffix
- Config references updated to match new key names
- Visualization hook video tags prefixed with `video/`

### Tag mapping

| Old | New |
|-----|-----|
| `loss_joints` | `loss/joints/train` |
| `loss_depth` | `loss/depth/train` |
| `loss_uv` | `loss/uv/train` |
| `mpjpe` (training) | `mpjpe` (unchanged, will be consumed by epoch hook in Slice 2) |
| `bedlam/mpjpe/all` | `mpjpe/rel/val` |
| `bedlam/mpjpe/body` | `mpjpe/body/val` |
| `bedlam/mpjpe/hand` | `mpjpe/hand/val` |
| `{split}/{scene}/gt_pelvis` | `video/{split}/{scene}/gt_pelvis` |
| `{split}/{scene}/pred_pelvis` | `video/{split}/{scene}/pred_pelvis` |

**Status: COMPLETE** (commit 1800b43)

## Acceptance criteria

- [x] Head `loss()` returns `loss/joints/train`, `loss/depth/train`, `loss/uv/train` instead of `loss_joints`, `loss_depth`, `loss_uv`
- [x] `BedlamMPJPEMetric` returns `mpjpe/rel/val`, `mpjpe/body/val`, `mpjpe/hand/val`
- [x] Config `save_best='mpjpe/body/val'` and `monitor='mpjpe/body/val'`
- [x] Visualization hook video tags prefixed with `video/`
- [x] Training runs for 1 epoch without errors
- [x] TensorBoard shows all scalars under new tag names
- [x] MMEngine-managed tags (lr, grad_norm, etc.) unchanged

## Blocked by

None - can start immediately

## User stories addressed

- User story 5: loss tags grouped under `loss/`
- User story 6: MPJPE variants grouped under `mpjpe/`
- User story 7: video tags prefixed with `video/`
- User story 10: save_best and early stopping use `mpjpe/body/val`
- User story 11: MMEngine tags unchanged
