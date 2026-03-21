# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sapiens is Meta Reality Labs' foundation model library for human-centric vision tasks (pose estimation, segmentation, depth, normal estimation). Models are pretrained on 300M in-the-wild human images at native 1024×1024 resolution with 16-pixel patches. The model family (0.3B, 0.6B, 1B, 2B ViTs) is built on the OpenMMLab stack.

There is also a custom RGBD 3D pose project in `claude_code/` — see `claude_code/CLAUDE.md` for its specific guidance.

## Environment

**Full (training):** Use the `sapiens` conda environment.
```bash
cd _install && ./conda.sh   # one-time setup
conda activate sapiens
```

**Custom RGBD project:** Use `sapiens_gpu` conda environment.
```bash
conda run -n sapiens_gpu python ...
```

**Lite (inference-only, 4× faster):**
```bash
conda create -n sapiens_lite python=3.10
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks
```

## Build / Install

Each module is an editable install. The `_install/conda.sh` script installs them in dependency order:
```bash
pip install -e engine/
pip install -e cv/
pip install -e pretrain/
pip install -e pose/
pip install -e det/
pip install -e seg/
```

## Common Commands

### Training (pose as example)
```bash
cd pose
python tools/train.py configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py \
  --work-dir <output_dir> --resume auto --amp
```

### Testing / Evaluation
```bash
cd pose
python tools/test.py <config> <checkpoint>
```

### Inference Demo (shell scripts)
```bash
pose/scripts/demo/local/keypoints17.sh    # 17-keypoint COCO
pose/scripts/demo/local/keypoints133.sh   # 133-keypoint WholeBody
pose/scripts/demo/local/keypoints308.sh   # 308-keypoint Goliath
```

### Python Inference API
```python
from mmpose.apis import init_model, inference_topdown
model = init_model(config_path, checkpoint_path, device='cuda:0')
results = inference_topdown(model, img)
```

### Feature Extraction (pretrain)
```bash
pretrain/scripts/demo/local/extract_feature.sh
```

### Run Tests
```bash
cd pose && pytest tests/
cd pretrain && pytest tests/
```

## Architecture

### Module Dependencies
```
MMEngine (training framework)
    ↓
MMCV (cv/): image ops, CNN blocks, NMS
    ↓
mmpretrain (pretrain/): ViT backbone, pretraining
mmpose (pose/):         pose estimation
mmseg (seg/):           segmentation
mmdet (det/):           person detection
```

### Configuration System
All modules use MMEngine's Python-dict config files. Configs are hierarchical:
- `configs/_base_/` — shared dataset, optimizer, runtime blocks
- Task-specific configs inherit via `_base_` lists
- Override at runtime with `--cfg-options key=value`

### Model Pipeline (Pose)
```
RGB (1024×768) → Sapiens ViT backbone → feature map (1024×48×64)
  → deconv neck (2× upsample) → HeatmapHead → keypoints (17/133/308)
```

### Registry System
Each module registers components via MMEngine's `Registry`. New models, datasets, and transforms must be registered to be addressable in configs:
```python
# Example: pose module
from mmpose.registry import MODELS
@MODELS.register_module()
class MyModel: ...
```

### Keypoint Variants
| Config dir | Joints | Use case |
|---|---|---|
| `sapiens_pose/coco/` | 17 | Standard body |
| `sapiens_pose/coco_wholebody/` | 133 | Body + face + hands |
| `sapiens_pose/goliath/` | 308 | Dense face + body |

## BEDLAM2 RGBD 3D Pose (Integrated)

A custom RGBD 3D pose task has been integrated into the main `pose/` module (commit 4bba4a7). It predicts 70 active SMPL-X joints + pelvis depth/UV from 4-channel (RGB+D) input. Two heads available: `Pose3dTransformerHead` (default, best results) and `Pose3dRegressionHead` (baseline). See `pose/docs/bedlam2/training_results.md` for A/B results.

### One-time data preprocessing (scripts still in `claude_code/`)
```bash
# Extract video frames to JPEGs
conda run -n sapiens_gpu python claude_code/scripts/extract_frames.py \
  --data-root <DATA_ROOT> --workers 8

# Convert depth NPZ → NPY mmap (3× faster loading)
conda run -n sapiens_gpu python claude_code/scripts/convert_depth_npy.py \
  --data-root <DATA_ROOT> --workers 8

# Generate train/val/test splits
python pose/tools/generate_bedlam2_splits.py \
  --data-root <DATA_ROOT> --output-dir pose/data/bedlam2_splits/
```

### Training
```bash
cd pose
python tools/train.py configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
  --work-dir <output_dir> --amp
```

### Inference demo
```bash
pose/scripts/demo/local/bedlam2.sh
```

### Architecture
- **Backbone:** `SapiensBackboneRGBD` — 4-channel ViT (`pose/mmpose/models/backbones/sapiens_rgbd.py`)
- **Head (default):** `Pose3dTransformerHead` — transformer decoder, per-joint queries + cross-attention (`pose/mmpose/models/heads/regression_heads/pose3d_transformer_head.py`)
- **Head (baseline):** `Pose3dRegressionHead` — GAP + MLP, 3 branches: joints (70×3 m), pelvis_depth (m), pelvis_uv (`pose/mmpose/models/heads/regression_heads/pose3d_regression_head.py`)
- **Estimator:** Custom `RGBDPose3dEstimator` replaces `TopdownPoseEstimator` (skips 2D back-transform) (`pose/mmpose/models/pose_estimators/rgbd_pose3d.py`)
- **Dataset:** `Bedlam2Dataset` — sequence-level splits, every 5th frame, mmap NPY depth (`pose/mmpose/datasets/datasets/body3d/bedlam2_dataset.py`)
- **Metric:** MPJPE (mm) on body/hand/all joints (`pose/mmpose/evaluation/metrics/bedlam_metric.py`)

### Coordinate convention (non-standard)
BEDLAM2 camera space: **X=forward (depth), Y=left, Z=up** — differs from OpenCV (X=right, Y=down, Z=forward).
Projection: `u = fx·(-Y/X) + cx`, `v = fy·(-Z/X) + cy`

### Joint subset
70 active joints from 127 SMPL-X: body (22) + eyes (2) + hands (30) + surface (16). Defined in `pose/mmpose/datasets/datasets/body3d/constants.py`.

### Loss
SmoothL1 (β=0.05m) on all 3 branches. Weights configurable: `--cfg-options lambda_depth=1.0 lambda_uv=1.0`

### Standalone prototype
`claude_code/` contains the original standalone prototype (uses `sapiens_gpu` env). See `claude_code/CLAUDE.md` for its specific commands.

## Key File Locations

- Checkpoint download: HuggingFace `facebook/sapiens`; set `$SAPIENS_CHECKPOINT_ROOT` to the `sapiens_host/` directory
- Task docs: `docs/tasks/` (POSE, SEG, DEPTH, NORMAL, PRETRAIN READMEs)
- Fine-tuning guides: `docs/finetune/`
- Lite inference guides: `lite/docs/`
- BEDLAM2 docs: `pose/docs/bedlam2/` (README.md → start here, training.md, integration.md, training_results.md)
- Pose design docs: `pose/docs/design/` (pipeline.md → model/inference, data_transforms.md, training_loop.md, dataload.md, visualization.md)
- PRDs & issues: `pose/docs/prd/`

## Docs Structure

### `docs/` (project-wide)
```
docs/
├── evaluate/
│   └── POSE_README.md          # Evaluation instructions for pose
├── finetune/
│   ├── DEPTH_README.md
│   ├── NORMAL_README.md
│   ├── POSE_README.md
│   └── SEG_README.md
├── tasks/
│   ├── DEPTH_README.md
│   ├── NORMAL_README.md
│   ├── POSE_README.md
│   ├── PRETRAIN_README.md
│   └── SEG_README.md
└── update_log/
    ├── README.md               # Index of session logs
    └── YYYY-MM-DD.md           # Per-session change logs
```

### `pose/docs/` (pose-module-specific)
```
pose/docs/
├── README.md                   # Navigation hub with doc-layer explanation
├── bedlam2/
│   ├── README.md               # Start here for BEDLAM2 (quick links + reading order)
│   ├── integration.md          # Architecture mapping, file locations, design decisions
│   ├── training.md             # Training, eval, and inference guide
│   └── training_results.md     # Per-epoch metrics for completed runs
├── design/
│   ├── pipeline.md             # Model architecture + inference — read first
│   ├── data_transforms.md      # Data format, coordinate system, transform chain
│   ├── training_loop.md        # Optimizer, loss, metrics, TensorBoard tags
│   ├── dataload.md             # Depth NPY conversion and loading performance
│   ├── visualization.md        # Visualization hook and demo rendering
│   ├── attention_pooling_pelvis.md  # Design: attention pooling for pelvis
│   └── mpjpe_logging_investigation.md  # Investigation: invariant training MPJPE
└── prd/
    ├── transformer_decoder_head.md     # PRD: transformer decoder head (COMPLETE)
    ├── tensorboard_restructure.md      # PRD: TensorBoard restructure (COMPLETE)
    └── issues/
        ├── 001_transformer_decoder_head_module.md  # COMPLETE
        ├── 002_training_config_smoke_test.md        # COMPLETE
        ├── 003_ab_training_evaluation.md            # IN PROGRESS
        ├── 004_restructure_tags.md                  # COMPLETE
        ├── 005_absolute_mpjpe_and_epoch_avg.md      # COMPLETE
        └── 006_bedlam2_transform_testability.md     # RFC (open)
```

## Session Convention

After every session where files are modified, write a summary to `docs/update_log/YYYY-MM-DD.md` (create the file if it doesn't exist). List each changed file and briefly describe what was changed and why.
