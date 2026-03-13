# Validation Visualization Pipeline

Visualization runs every epoch after validation and logs pose-overlay videos to TensorBoard.

## Sequence Selection

**4 val sequences + 4 train sequences** are fixed at startup and reused every epoch.

Within each split, the 4 sequences are chosen as:
- 1 sequence with `rotate_flag=True` (portrait video, rotated CCW 90° at extraction)
- 2 sequences with `rotate_flag=False` (landscape / already-upright video)
- 1 sequence drawn **randomly** from the full split each epoch (any `rotate_flag`)

The fixed 3 ensure stable comparisons across epochs. The random slot provides variety — showing the model on unseen sequences each epoch.

`rotate_flag` is read from each sequence's label NPZ at startup for the fixed slots.

## Two Videos Per Sequence

Each sequence produces **two side-by-side TensorBoard videos** per epoch:

| Tag suffix | Pelvis source | What it shows |
|---|---|---|
| `gt_pelvis` | GT `pelvis_abs` from dataset labels | Quality of predicted **relative** joint layout |
| `pred_pelvis` | Recovered from model's `pelvis_depth` + `pelvis_uv` | Full end-to-end **absolute** pose quality |

TensorBoard tags:
- `val/scene_0/gt_pelvis`, `val/scene_0/pred_pelvis`, ..., `val/scene_3/...`
- `train/scene_0/gt_pelvis`, `train/scene_0/pred_pelvis`, ..., `train/scene_3/...`

## Absolute Pelvis Recovery (`pred_pelvis` videos)

Follows the inference pipeline from `docs/pipeline.md` §5, adapted for the crop image space:

```
1. Denormalize pred_uv from [-1, 1] to crop pixels:
   u_crop = (u_norm + 1) / 2 * crop_w      (crop_w = 384)
   v_crop = (v_norm + 1) / 2 * crop_h      (crop_h = 640)

2. Use pred_depth as X (forward distance in metres, raw metres — not normalized)

3. Unproject with crop K (BEDLAM2 convention: X=forward, Y=left, Z=up):
   Y = -(u_crop - cx) * X / fx
   Z = -(v_crop - cy) * X / fy

4. pelvis_pred_abs = [X, Y, Z]

5. joints_abs = pred_rel + pelvis_pred_abs[np.newaxis, :]   # (127, 3)
```

Using crop K directly (rather than inverting to original image then applying original K) is equivalent and consistent with `draw_pose_frame`, which also projects onto the crop image.

## Functions (all in `train.py`)

### `select_vis_indices(dataset, n_rotate_true=1, n_rotate_false=2)` — called once at startup

Scans the dataset index and picks the first frame from sequences satisfying the `rotate_flag` distribution. Returns a fixed list of 3 indices: `[rotate_true_idx, rotate_false_idx_0, rotate_false_idx_1]`. `rotate_flag` is read from the label NPZ for each new sequence key `(label_path, body_idx)`.

### `sample_random_vis_index(dataset)` — called each visualization epoch

Picks a random sequence from the dataset (uniform over unique `(label_path, body_idx)` keys) and returns the flat index of its first frame. Called fresh each visualization epoch so the random slot differs across epochs.

### `recover_pelvis_from_pred(pred_depth, pred_uv, K, crop_hw=(640, 384))` — helper

Implements steps 1–4 above. Returns `(3,)` float32 array `[X, Y, Z]` in camera space metres.

### `visualize_fixed_samples(model, dataset, indices, device, val_tf)` — called every epoch, indices = fixed_3 + [random_1]

For each starting index, walks forward through consecutive frames of the same sequence until `_VIS_FRAMES=16` frames are collected. For each frame:
1. Temporarily swaps dataset transform to `val_tf` (no augmentation)
2. Runs a forward pass: `out = model(x)` where `x = (B, 4, H, W)` [RGB | depth]
3. Collects:
   - `pred_joints` `(127, 3)` root-relative from `out["joints"]`
   - `pred_depth` scalar from `out["pelvis_depth"]`
   - `pred_uv` `(2,)` from `out["pelvis_uv"]`
   - GT `pelvis_abs` `(3,)` from `sample["pelvis_abs"]`
   - Crop K from `sample["intrinsic"]`
4. Returns **two** `(1, T, 3, H, W)` uint8 video arrays per sequence: one GT-anchored, one pred-anchored.

### `build_val_video(rgb_frames, pred_frames, K_frames, pelvis_frames)` — assembles video

Calls `draw_pose_frame` on each frame and stacks into `(1, T, 3, H, W)`.

### `draw_pose_frame(rgb_chw, joints, K, pelvis_abs, color)` — draws one frame

1. **Root → absolute:** adds provided `pelvis_abs` (GT or predicted) to root-relative joints
2. **Projection** (BEDLAM2 convention):
   ```
   u = fx * (-Y / X) + cx
   v = fy * (-Z / X) + cy
   ```
   Joints with `X <= 0.01 m` are skipped as invalid.
3. **Skeleton:** draws bones as lines between joint pairs from `SMPLX_SKELETON`
4. **Joints:** draws each of 127 joints as a 3 px dot
5. Returns `(3, H, W)` uint8 RGB

## TensorBoard Logging

Videos are logged via `writer.add_video(tag, vid, global_step=epoch+1, fps=4)` at the end of each validation epoch. Each tag produces a 16-frame clip at 4 fps.

## Key Difference Between the Two Videos

| | `gt_pelvis` | `pred_pelvis` |
|---|---|---|
| Pelvis XYZ | From GT label (`pelvis_abs` in dataset sample) | Recovered from `pred_depth` + `pred_uv` + crop K |
| Useful for | Diagnosing relative pose quality in isolation | Diagnosing full end-to-end absolute pose quality |
| Skeleton misalignment cause | Errors in predicted relative joints only | Errors in relative joints **and** pelvis localization |
