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

## Key File Locations

- Checkpoint download: HuggingFace `facebook/sapiens`; set `$SAPIENS_CHECKPOINT_ROOT` to the `sapiens_host/` directory
- Task docs: `docs/POSE_README.md`, `docs/PRETRAIN_README.md`, `docs/SEG_README.md`, `docs/DEPTH_README.md`, `docs/NORMAL_README.md`
- Fine-tuning guides: `docs/finetune/`
- Lite inference guides: `lite/docs/`
