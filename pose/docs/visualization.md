# Validation Visualization Pipeline

Visualization runs every epoch after validation and logs pose-overlay videos to TensorBoard.
Implemented as BEDLAM2 mode in `Pose3dVisualizationHook` (`mmpose/engine/hooks/pose3d_visualization_hook.py`), enabled via `bedlam2_video=True` in the config.

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

## Implementation (MMEngine hook)

Implemented in `Pose3dVisualizationHook` (`mmpose/engine/hooks/pose3d_visualization_hook.py`) with `bedlam2_video=True`.

### Config

```python
custom_hooks = [
    dict(type='Pose3dVisualizationHook',
         enable=True,
         bedlam2_video=True,
         vis_interval=1),
]
```

### Key methods (all prefixed with `_bedlam2_`)

| Method | When | What |
|--------|------|------|
| `_bedlam2_select_fixed(dataset, runner)` | `before_run` | Scans data_list, loads NPZ for `rotate_flag`/`n_body`; returns `[rotate_true_idx, rotate_false_idx, multi_person_idx]` |
| `_bedlam2_random_start(dataset)` | Each epoch | Picks random `(label_path, body_idx)` key, returns first flat index |
| `_bedlam2_collect_frames(dataset, start_idx)` | Per sequence | Walks forward collecting up to `n_vis_frames` indices with same `(label_path, body_idx)` |
| `_bedlam2_recover_pelvis(depth, uv, K, h, w)` | Per frame | Unprojects `pelvis_uv` [-1,1] + `pelvis_depth` to absolute `[X, Y, Z]` using crop K |
| `_bedlam2_project_2d(joints_abs, K)` | Per frame | Projects absolute joints to pixel coords using BEDLAM2 convention |
| `_bedlam2_draw_frame(img, joints, K, color)` | Per frame | Draws body skeleton (joints 0-21, `_BODY_LINKS`) on image, returns `(3,H,W)` uint8 |
| `_bedlam2_visualize_sequence(...)` | Per sequence | Full pipeline: collect frames → forward pass → draw gt/pred videos → log to TensorBoard |

### Frame rendering

For each frame:
1. Run model forward pass: `pred_joints` `(70, 3)` root-relative, `pred_depth`, `pred_uv`
2. For `gt_pelvis`: recover GT pelvis from GT labels, add to pred joints, project using crop K, draw on crop image
3. For `pred_pelvis`: recover pred pelvis from model output using crop K, project using **original K**, draw on original image
4. Multi-person: run separate forward passes for all `body_idx` values at that frame, draw each in a different color (`_PERSON_COLORS`)

### Skeleton drawing

Draws body joints 0-21 using `_BODY_LINKS`:
```
(0,1),(0,2),(1,3),(2,4),(3,5),(4,6) — hips/legs
(0,7),(7,8),(8,9),(9,10),(10,21)    — spine/head
(8,11),(11,12),(12,13)              — left arm
(8,14),(14,15),(15,16)              — right arm
```
Joints with X <= 0.01m are skipped. Each joint gets a 4px dot, bones get 2px lines.

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
