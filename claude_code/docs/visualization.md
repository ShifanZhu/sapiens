# Validation Visualization Pipeline

Visualization runs every epoch after validation and logs pose-overlay videos to TensorBoard.

## Sequence Selection

**4 val sequences + 4 train sequences** are used each epoch.

| Scene tag | Type | Description |
|---|---|---|
| `scene_0` | fixed | `rotate_flag=True` (portrait video, rotated CCW 90° at extraction) |
| `scene_1` | fixed | `rotate_flag=False` (landscape / already-upright video) |
| `scene_2` | fixed | Multi-person sequence (`n_body > 1`) |
| `scene_3_random` | random | Drawn randomly from the full split each epoch (any type) |

The 3 fixed scenes ensure stable comparisons across epochs. The random slot provides variety.

`rotate_flag` and `n_body` are read from each sequence's label NPZ at startup for the fixed slots.

## Two Videos Per Sequence

Each sequence produces **two TensorBoard videos** per epoch:

| Tag suffix | People shown | Joints | Image | K used | What it shows |
|---|---|---|---|---|---|
| `gt_pelvis` | Selected body only | pred relative + GT pelvis | **Crop** (384×640) | Crop K | Quality of predicted **relative** joint layout |
| `pred_pelvis` | **All people in scene** | pred relative + pred pelvis per person | **Original uncropped image** | Original K | Full end-to-end **absolute** pose quality |

For `scene_2` (multi-person), `pred_pelvis` runs a separate forward pass for every person in the scene and overlays all skeletons on the original image, each in a distinct color (`_PERSON_COLORS`: green, orange, blue, magenta, cyan, cycling if n_body > 5).

TensorBoard tags:
- `val/scene_0/gt_pelvis`, `val/scene_0/pred_pelvis`
- `val/scene_1/gt_pelvis`, `val/scene_1/pred_pelvis`
- `val/scene_2/gt_pelvis`, `val/scene_2/pred_pelvis`
- `val/scene_3_random/gt_pelvis`, `val/scene_3_random/pred_pelvis`
- (same pattern for `train/`)

## Absolute Pelvis Recovery (`pred_pelvis` videos)

Steps 1–4 unproject the predicted 2D pelvis position into camera-space 3D using crop K. Step 5 then projects the recovered absolute joints onto the **original uncropped image** using original K.

```
1. Denormalize pred_uv from [-1, 1] to crop pixels:
   u_crop = (u_norm + 1) / 2 * crop_w      (crop_w = 384)
   v_crop = (v_norm + 1) / 2 * crop_h      (crop_h = 640)

2. Use pred_depth as X (forward distance in metres, raw metres — not normalized)

3. Unproject with crop K (BEDLAM2 convention: X=forward, Y=left, Z=up):
   Y = -(u_crop - cx_crop) * X / fx_crop
   Z = -(v_crop - cy_crop) * X / fy_crop

4. pelvis_pred_abs = [X, Y, Z]

5. joints_abs = pred_rel + pelvis_pred_abs[np.newaxis, :]   # (127, 3)

6. Project joints_abs onto the original image using original K:
   u = fx_orig * (-Y / X) + cx_orig
   v = fy_orig * (-Z / X) + cy_orig
```

Drawing on the original image shows where the predicted skeleton lands in the full scene, making it easy to spot absolute localization errors.

## Functions (all in `train.py`)

### `select_vis_indices(dataset, n_rotate_true=1, n_rotate_false=1, n_multi_person=1)` — called once at startup

Scans the dataset index and picks the first frame from sequences satisfying the slot distribution. Returns a fixed list of up to 3 indices: `[rotate_true_idx, rotate_false_idx, multi_person_idx]`. `rotate_flag` and `n_body` are loaded from each sequence's label NPZ (cached in `label_meta` dict to avoid redundant IO). If no multi-person sequence is available (e.g. frames not yet extracted), that slot is silently omitted.

### `sample_random_vis_index(dataset)` — called each visualization epoch

Picks a random sequence from the dataset (uniform over unique `(label_path, body_idx)` keys) and returns the flat index of its first frame. Called fresh each visualization epoch so the random slot differs across epochs.

### `recover_pelvis_from_pred(pred_depth, pred_uv, K, crop_hw=(640, 384))` — helper

Implements steps 1–4 above. Returns `(3,)` float32 array `[X, Y, Z]` in camera space metres.

### `visualize_fixed_samples(model, dataset, indices, device, val_tf)` — called every epoch, indices = fixed_3 + [random_1]

For each starting index, detects whether the sequence is multi-person (`n_body > 1`), then walks forward through consecutive frames of the same `(label_path, body_idx)` until `_VIS_FRAMES=16` frames are collected. For each frame:
1. Temporarily swaps dataset transform to `val_tf` (no augmentation)
2. Runs a forward pass for the selected body: `out = model(x)` where `x = (B, 4, H, W)` [RGB | depth]
3. Collects for `gt_pelvis`:
   - `pred_joints` `(127, 3)` root-relative, GT `pelvis_abs` `(3,)`, crop K
4. Collects for `pred_pelvis`:
   - Original uncropped RGB and original K from `dataset._label_cache` / `dataset._read_frame`
   - **Single-person:** recovers pelvis from `pred_depth` + `pred_uv` + crop K; one `(joints, pelvis, color)` tuple
   - **Multi-person:** uses `frame_body_map` to run a separate forward pass for every `body_idx`; collects one `(joints, pelvis, color)` tuple per person (colors from `_PERSON_COLORS`)
5. Returns **two** video arrays per sequence:
   - `gt_pelvis`: `(1, T, 3, 640, 384)` — selected body on crop image, crop K
   - `pred_pelvis`: `(1, T, 3, H_orig, W_orig)` — all bodies on original image, original K, drawn iteratively via `draw_pose_frame`

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
| People | Selected body only | All people in the scene (multi-person for `scene_2`) |
| Image | Crop (384×640) | Original uncropped image |
| Projection K | Crop K | Original K |
| Useful for | Diagnosing relative pose quality in isolation | Diagnosing full end-to-end absolute pose quality |
| Skeleton misalignment cause | Errors in predicted relative joints only | Errors in relative joints **and** pelvis localization |
