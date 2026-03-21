# BEDLAM2 RGBD 3D Pose Integration

## Contents

- [Background](#background)
- [Architecture Mapping](#architecture-mapping)
- [New Files](#new-files)
- [Modified Files](#modified-files)
- [Data Format](#data-format)
- [Training Setup](#training-setup)
- [Key Design Decisions](#key-design-decisions)

This document describes the migration of `claude_code/` — a standalone RGBD 3D pose
estimation project — into the official sapiens/mmpose/mmengine architecture.

After this integration, training runs with:

```bash
python pose/tools/train.py \
    pose/configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
    --work-dir /tmp/bedlam2_exp --amp
```

---

## Background

`claude_code/` fine-tunes a Sapiens ViT backbone with 4-channel RGB+Depth input on the
[BEDLAM2](https://bedlam.is.tue.mpg.de/) dataset to predict:

- **70 SMPL-X 3D joints** (camera-space, root-relative) in metres
- **Pelvis depth** — forward distance of the pelvis from the camera (metres)
- **Pelvis UV** — 2D pelvis position in the crop, normalised to `[-1, 1]`

The standalone code worked but lived outside the mmpose registry system, so it couldn't
use distributed training, MMEngine hooks, or config-driven hyperparameter sweeps.

---

## Architecture Mapping

```
claude_code/                                  →  pose/ (mmpose/mmengine)
──────────────────────────────────────────────────────────────────────────
data/dataset.py   BedlamFrameDataset          →  datasets/datasets/body3d/bedlam2_dataset.py
data/transforms.py (5 transforms)             →  datasets/transforms/bedlam2_transforms.py
data/splits.py    split utilities             →  tools/generate_bedlam2_splits.py
model/backbone.py SapiensBackboneRGBD         →  models/backbones/sapiens_rgbd.py
model/head.py     Pose3DHead                  →  models/heads/regression_heads/pose3d_regression_head.py
model/weights.py  load_sapiens_pretrained     →  models/utils/rgbd_weight_utils.py
train.py          standalone training loop    →  configs/sapiens_pose/bedlam2/<config>.py
                                                  +  pose/tools/train.py  (unchanged)
```

A new `RGBDPose3dEstimator` replaces `TopdownPoseEstimator` because the topdown
estimator applies a 2D affine back-transform on keypoints (undoing the crop warp),
which corrupts 3D camera-space joint coordinates.

---

## New Files

### Model

| File | Class | Purpose |
|------|-------|---------|
| `pose/mmpose/models/utils/rgbd_weight_utils.py` | `load_sapiens_pretrained_rgbd()` | Loads an RGB pretrain checkpoint into a 4-channel model: remaps key prefixes, expands patch-embed 3→4 channels (depth = mean of RGB), bicubic-interpolates `pos_embed` to the target grid size |
| `pose/mmpose/models/backbones/sapiens_rgbd.py` | `SapiensBackboneRGBD` | Wraps `mmpretrain.VisionTransformer` with `in_channels=4`, `with_cls_token=False`, `out_type='featmap'`. Calls `load_sapiens_pretrained_rgbd` in `init_weights()` when `pretrained=<path>` |
| `pose/mmpose/models/heads/regression_heads/pose3d_regression_head.py` | `Pose3dRegressionHead` | `AdaptiveAvgPool2d(1)` → flatten → `Linear+LN+GELU+Dropout` → three branches: joints `(B,70,3)`, pelvis_depth `(B,1)`, pelvis_uv `(B,2)`. `loss()` returns `(losses_dict, pred_dict)` |
| `pose/mmpose/models/data_preprocessors/rgbd_data_preprocessor.py` | `RGBDPoseDataPreprocessor` | Pass-through preprocessor — RGB and depth are already normalised inside `PackBedlamInputs` |
| `pose/mmpose/models/pose_estimators/rgbd_pose3d.py` | `RGBDPose3dEstimator` | Minimal `BasePoseEstimator` subclass. Skips the 2D affine back-transform that `TopdownPoseEstimator.add_pred_to_datasample` would apply |

### Data

| File | Class | Purpose |
|------|-------|---------|
| `pose/mmpose/datasets/datasets/body3d/bedlam2_dataset.py` | `Bedlam2Dataset` | Inherits `BaseDataset` (not `BaseCocoStyleDataset` — no COCO JSON). `load_data_list()` iterates NPZ label files and emits one dict per `(label_path, body_idx, frame_idx)` triple. Accepts `seq_paths` list or `seq_paths_file` txt path |
| `pose/mmpose/datasets/transforms/bedlam2_transforms.py` | 5 transforms | See table below |
| `pose/configs/_base_/datasets/bedlam2_smplx70.py` | — | Dataset metainfo: 70 joint names, skeleton links, flip pairs, joint weights, sigmas |

#### Transforms

| Class | Ported from | What it does |
|-------|-------------|--------------|
| `LoadBedlamLabels` | `dataset._read_frame` + `_read_depth` + `_compute_bbox` | Loads RGB (JPEG), depth (NPY mmap → NPZ LRU fallback), joints, K matrix, bbox. Returns `None` for OOB (>70% joints outside image) or tiny-bbox samples — triggers MMEngine retry |
| `NoisyBBoxTransform` | `NoisyBBox` | Training augmentation: random position and scale jitter on the bounding box |
| `CropPersonRGBD` | `CropPerson` | Crops and resizes RGB + depth to `(out_h, out_w)` using the bbox, expanding to match target aspect ratio. Pads out-of-bounds with zeros. Updates intrinsic K. Depth uses `INTER_NEAREST` to avoid edge bleeding |
| `SubtractRootJoint` | `SubtractRoot` | Subtracts pelvis from all joints (root-relative). Computes `pelvis_depth` (X coord) and `pelvis_uv` (projected, normalised to `[-1,1]`) |
| `PackBedlamInputs` | `ToTensor` + packing | Normalises RGB (ImageNet mean/std), clips depth to 20m and divides. Concatenates into `(4, H, W)` tensor. Packs GT into `PoseDataSample.gt_instances.lifting_target` and `gt_instance_labels.{pelvis_depth, pelvis_uv}` |

### Evaluation

| File | Class | Metrics |
|------|-------|---------|
| `pose/mmpose/evaluation/metrics/bedlam_metric.py` | `BedlamMPJPEMetric` | `mpjpe/all` (70 joints), `mpjpe/body` (indices 0–21), `mpjpe/hand` (indices 24–53), all in millimetres |

### Config & Tools

| File | Purpose |
|------|---------|
| `pose/configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py` | Main training config. Two-group AdamW (backbone LR = 1e-5, head LR = 1e-4), cosine LR with linear warmup, 50 epochs, batch 16, AMP |
| `pose/tools/generate_bedlam2_splits.py` | One-time CLI: reads `data/overview.txt`, filters by depth/frames availability, writes `train_seqs.txt`, `val_seqs.txt`, `test_seqs.txt` (seed=2026, sequence-level split) |

---

## Modified Files

| File | Change |
|------|--------|
| `pose/mmpose/models/backbones/__init__.py` | Added `SapiensBackboneRGBD` |
| `pose/mmpose/models/heads/regression_heads/__init__.py` | Added `Pose3dRegressionHead` |
| `pose/mmpose/models/heads/__init__.py` | Added `Pose3dRegressionHead` to re-exports |
| `pose/mmpose/models/data_preprocessors/__init__.py` | Added `RGBDPoseDataPreprocessor` |
| `pose/mmpose/models/pose_estimators/__init__.py` | Added `RGBDPose3dEstimator` |
| `pose/mmpose/datasets/transforms/__init__.py` | Added 5 new Bedlam2 transforms |
| `pose/mmpose/datasets/datasets/body3d/__init__.py` | Added `Bedlam2Dataset` |
| `pose/mmpose/evaluation/metrics/__init__.py` | Added `BedlamMPJPEMetric` |

---

## Data Format

### BEDLAM2 directory layout expected

```
<data_root>/
├── data/
│   ├── overview.txt          # sequence index with filter flags
│   ├── label/
│   │   └── <folder>/<seq>.npz   # joints_cam, intrinsic_matrix, joints_2d, n_frames
│   ├── frames/
│   │   └── <folder>/<seq>/<00000>.jpg  # pre-extracted JPEG frames
│   └── depth/
│       ├── npy/<folder>/<seq>.npy      # fast: float16 mmap (preferred)
│       └── npz/<folder>/<seq>.npz      # fallback
```

### Active joint subset (70 joints)

| Group | Active indices | Original SMPL-X indices |
|-------|---------------|------------------------|
| Body | 0–21 | 0–21 |
| Eyes | 22–23 | 23–24 (jaw=22 excluded) |
| Left hand | 24–38 | 25–39 |
| Right hand | 39–53 | 40–54 |
| Surface (toes, heels, fingertips) | 54–69 | 60–75 |

### Coordinate system (BEDLAM2 camera space)

- **X** = forward (depth), **Y** = left, **Z** = up; units = metres
- Projection: `u = fx·(−Y/X) + cx`, `v = fy·(−Z/X) + cy`

---

## Training Setup

See **[training.md](training.md)** for the complete setup guide (prerequisites, training commands, evaluation, inference demo, and hyperparameter reference).

---

## Key Design Decisions

**`RGBDPose3dEstimator` instead of `TopdownPoseEstimator`**
`TopdownPoseEstimator.add_pred_to_datasample` undoes the crop affine transform to convert
predicted keypoints back to full-image pixel coordinates. This makes sense for 2D heatmap
models but corrupts 3D camera-space joint coordinates. The custom estimator stores
predictions directly.

**Normalisation in `PackBedlamInputs`, not in the data preprocessor**
Since the model takes a 4-channel RGBD tensor, normalising inside the transform step (before
collation) keeps the preprocessor simple (`RGBDPoseDataPreprocessor` is a pass-through).

**`persistent_workers=False`**
BEDLAM2's depth NPY mmap files and label NPZ files create file descriptors that get
duplicated across persistent worker processes, quickly exhausting the FD limit. Setting
`persistent_workers=False` avoids this.

**`max_refetch=10` in dataset config**
`LoadBedlamLabels.transform()` returns `None` for out-of-bounds samples and tiny bboxes
instead of raising. MMEngine's `BaseDataset.prepare_data()` retries automatically up to
`max_refetch` times with a different random index.

**Sequence-level splits**
Splits are performed at the sequence level (not frame level) to prevent data leakage
between train and val sets. Frames from the same sequence always belong to the same split.
