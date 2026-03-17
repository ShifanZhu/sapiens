# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> This is the `pose/` module of the Sapiens project. See the root `CLAUDE.md` for project-wide context, environment setup, and BEDLAM2 coordinate conventions.

## Editing Scope

Only modify files inside `pose/`. Do NOT change code outside the `pose/` folder (the rest of the sapiens repo). The only exception is `../docs/update_log/` for session logs. If a change outside `pose/` seems necessary, always ask the user for permission first.

## Session Convention

After every session where files are modified, append an entry to `../docs/update_log/YYYY-MM-DD.md` (create the file if it doesn't exist). List each changed file and briefly describe what was changed and why. Always include a brief summary section at the top of the file each time it is modified.

## Common Commands

```bash
# Training (run from pose/)
python tools/train.py configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
  --work-dir <output_dir>    # AMP is on by default; use --no-amp to disable

# Evaluation
python tools/test.py <config> <checkpoint>

# Inference demos
bash scripts/demo/local/bedlam2.sh
bash scripts/demo/local/keypoints17.sh    # COCO 17-keypoint
bash scripts/demo/local/keypoints133.sh   # WholeBody 133-keypoint

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_models/test_heads/test_heatmap_head.py -v
```

## Architecture

### Model Pipeline
```
RGB/RGBD input → Backbone (Sapiens ViT) → [optional Neck] → Head → predictions
```

All components are registered via MMEngine's `Registry`. A new model/dataset/transform must use `@MODELS.register_module()` (or `@DATASETS`, `@TRANSFORMS`) to be addressable in configs.

### Standard 2D Topdown Pipeline
- **Backbone:** `SapiensBackbone` in `mmpose/models/backbones/` — standard ViT with 3-channel input
- **Neck:** Deconv upsample (2×) in `mmpose/models/necks/`
- **Head:** `HeatmapHead` or `DSNTHead` in `mmpose/models/heads/`
- **Estimator:** `TopdownPoseEstimator` — applies 2D affine back-transform on predictions

### BEDLAM2 RGBD 3D Pipeline (custom, accuracy pending)
- **Backbone:** `SapiensBackboneRGBD` (`mmpose/models/backbones/sapiens_rgbd.py`) — 4-channel (RGB+D) ViT
- **Head:** `Pose3DRegressionHead` (`mmpose/models/heads/regression_heads/pose3d_regression_head.py`) — 3 output branches: 70 joints (×3 m), pelvis depth (m), pelvis UV (normalized)
- **Estimator:** `RGBDPose3dEstimator` (`mmpose/models/pose_estimators/rgbd_pose3d.py`) — skips 2D affine back-transform; joints are already in camera 3D space
- **Dataset:** `Bedlam2Dataset` (`mmpose/datasets/datasets/body3d/bedlam2_dataset.py`) — indexes `(label_path, body_idx, frame_idx)` triples from NPZ files
- **Metric:** MPJPE in mm on body/hand/all joint subsets (`mmpose/evaluation/metrics/bedlam_metric.py`)

### Config System
Configs are in `configs/sapiens_pose/<task>/`. All BEDLAM2 custom modules must be listed in `custom_imports` in the config to register them before the runner builds:
```python
custom_imports = dict(
    imports=[
        'mmpose.models.pose_estimators.rgbd_pose3d',
        'mmpose.models.backbones.sapiens_rgbd',
        ...
    ],
    allow_failed_imports=False,
)
```

### Key File Locations
| Path | Purpose |
|---|---|
| `mmpose/models/backbones/sapiens_rgbd.py` | 4-channel RGBD ViT backbone |
| `mmpose/models/heads/regression_heads/pose3d_regression_head.py` | 3D joint regression head |
| `mmpose/models/pose_estimators/rgbd_pose3d.py` | RGBD estimator (no 2D back-transform) |
| `mmpose/datasets/datasets/body3d/bedlam2_dataset.py` | BEDLAM2 dataset class |
| `mmpose/datasets/datasets/body3d/constants.py` | 70 active SMPL-X joint indices |
| `mmpose/evaluation/metrics/bedlam_metric.py` | MPJPE metric |
| `tools/generate_bedlam2_splits.py` | One-time split generation |
| `demo/demo_bedlam2.py` | BEDLAM2 inference demo script |
| `scripts/demo/local/bedlam2.sh` | Shell wrapper for demo |
| `scripts/finetune/bedlam2/` | Fine-tuning scripts |
