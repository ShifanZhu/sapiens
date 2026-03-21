# Data Format & Transform Pipeline

How BEDLAM2 data is loaded, filtered, and preprocessed before reaching the model.

See also: [pipeline.md](pipeline.md) (model architecture + inference), [training_loop.md](training_loop.md) (optimizer, loss, metrics).

---

## 1. Data (`Bedlam2Dataset`)

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

The active joint subset is defined in `constants.ACTIVE_JOINT_INDICES` and applied in `LoadBedlamLabels` at load time. The model head outputs 70 joints.

### Dataset Indexing

The flat index is `(label_path, body_idx, frame_idx)`. For a multi-person sequence with 3 bodies and 50 frames, that's 150 samples. Each sample is one person in one frame.

**Filtering** — a sample is skipped (retried with a random index) if:
1. Bounding box is smaller than 32×32 px.
2. More than **70%** of the 127 raw joints have a 2D projection outside the image. This removes frames where the person is mostly off-screen.

### Splits

Sequence-level splits (not frame-level) to prevent data leakage. Seed=2026.

---

## 2. Transform Pipeline (`bedlam2_transforms.py`)

### Training: `LoadBedlamLabels → NoisyBBox → CropPerson → SubtractRoot → PackBedlamInputs`

### Validation: `LoadBedlamLabels → CropPerson → SubtractRoot → PackBedlamInputs`

### Step A: `NoisyBBox` (train only)

Simulates an imperfect person detector. Randomly jitters:
- Center position: ±10% of box size
- Scale: ±15%

Result clamped to image bounds. Skipped if jitter produces a box < 2px.

### Step B: `CropPerson`

Crops the image to the person bounding box and resizes to 640×384.

1. Expand bbox to match target aspect ratio (640:384 = 5:3)
2. Pad with zeros if expanded box extends beyond image bounds
3. Crop and resize RGB (bilinear) and depth (nearest — avoids edge bleeding)
4. **Update intrinsic K** to maintain geometric consistency:
   ```
   fx' = fx * sx         fy' = fy * sy
   cx' = (cx - x0) * sx  cy' = (cy - y0) * sy
   ```
   where `sx = out_w / crop_w`, `x0` = crop origin in original image pixels

If no `bbox` key exists, falls back to plain resize.

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

Must run **after** `CropPerson` so that K is the crop K and `img` has the crop dimensions.

**Unit design:** All three regression targets are in similar numeric ranges:
- `joints`: root-relative metres, typically ±0.5
- `pelvis_depth`: raw metres, typically 1-10
- `pelvis_uv`: normalized [-1, 1], typically ±0.3

This avoids the need for aggressive loss down-weighting. All lambdas default to 1.0 and all SmoothL1 betas are 0.05.

**Denormalization at inference:** To recover crop pixels from model output: `u_px = (u_norm + 1) / 2 * crop_w`.

### Step D: `PackBedlamInputs`

| Field | Before | After |
|-------|--------|-------|
| `rgb` | `(H,W,3) uint8` | `(3,H,W) float32`, `/255`, ImageNet mean/std normalized |
| `depth` | `(H,W) float32` metres | `(1,H,W) float32`, clipped to [0, 20m], `/20` → **[0, 1] unitless** |
| `joints` | `(70,3) float32` metres | `(70,3) float32` tensor, root-relative **metres** |
| `intrinsic` | `(3,3) float32` | `(3,3) float32` tensor |
| `pelvis_depth` | `(1,) float32` metres | `(1,) float32` tensor, **raw metres** (not normalized) |
| `pelvis_uv` | `(2,) float32` normalized | `(2,) float32` tensor, **[-1, 1]** (0 = crop center) |

Note: the depth *image* is normalized to [0,1] for the model input, but `pelvis_depth` stays in raw metres as a regression target.
