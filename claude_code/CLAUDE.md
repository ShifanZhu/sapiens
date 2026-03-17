# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tune Sapiens (ViT-based) to predict 3D human joint positions (127 SMPL-X joints, camera space) from RGB + depth inputs using the BEDLAM2 dataset.

## Environment

Always use `conda run -n sapiens_gpu python ...` or activate `sapiens_gpu` first.

## Commands

```bash
# One-time: extract video frames to JPEGs (required before training)
python scripts/extract_frames.py --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest --workers 8

# One-time: convert depth NPZ → NPY mmap for 3× faster loading
python scripts/convert_depth_npy.py --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest --workers 8

# Smoke tests
python tests/test_data_pipeline.py
python tests/test_model.py

# Train
python train.py \
  --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
  --pretrain checkpoints/sapiens_0.3b_epoch_1600_clean.pth \
  --output-dir runs/exp001

# Resume training
python train.py ... --resume runs/exp001/best.pth

# Quick debug (cap batches)
python train.py ... --max-batches 5

# Profile data loading / forward / backward timing
python scripts/profile_timing.py --data-root ... --output-dir runs/profile

# Visualize GT joints + depth overlay
python scripts/visualize_depth_joints.py --data-root ... --output-dir runs/vis

# Monitor
tensorboard --logdir runs
```

## Architecture

**Full pipeline:** `(B,4,H,W)` → backbone → `(B,1024,24,40)` → head → `(B,127,3)` XYZ metres

### Data (`data/`)
- `splits.py` — sequence-level train/val/test splits parsed from `overview.txt`; filtering by `single_body_only`, `depth_required`, etc.
- `dataset.py` — `BedlamFrameDataset`: loads pre-extracted JPEGs + NPZ depth + NPZ joint labels; samples every 5th frame (30fps → 6fps); label NPZ is opened once per sequence, `joints_cam` loaded eagerly as plain ndarray, handle closed immediately (avoids FD exhaustion with many workers)
- `transforms.py` — Resize → ToTensor only (no augmentation); depth uses `INTER_NEAREST` to avoid edge bleeding; `RandomHorizontalFlip`, `ColorJitter`, `RandomResizedCropRGBD` classes exist but are **not used** in either train or val transform
- `constants.py` — 127 SMPL-X joint names, skeleton edges, flip pairs, normalization constants

**Batch dict keys:** `rgb (B,3,H,W)`, `depth (B,1,H,W)`, `joints (B,127,3)`, `intrinsic (B,3,3)`

### Model (`model/`)
- `backbone.py` — `SapiensBackboneRGBD`: mmpretrain ViT with `in_channels=4`; `out_type="featmap"`; `with_cls_token=False`
- `head.py` — `Pose3DHead`: `AdaptiveAvgPool2d(1)` → `Linear(embed→2048)` → `LayerNorm+GELU` → `Linear(2048→127×3)`
- `sapiens_pose3d.py` — `SapiensPose3D` wrapper; call `model.load_pretrained(path)` to load RGB checkpoint
- `weights.py` — Three conversions during pretrain loading: (1) add `backbone.vit.` key prefix, (2) expand patch embed `(C,3,16,16)→(C,4,16,16)` with depth channel = mean(RGB), (3) bicubic-interpolate `pos_embed` from 64×64+CLS to 24×40 grid and drop CLS token

Backbone imports from `/home/hang/repos_local/MMC/sapiens/` (paths hardcoded in `model/backbone.py`).

### Dataset stats
- 1800 sequences (1440 train / 180 val / 180 test); ~115k total frames
- `n_frames` per sequence: min=24, max=96, mean=64, median=59

### Training (`train.py`)
- **FD limit:** `resource.setrlimit(RLIMIT_NOFILE, 65536)` raised at startup to prevent `OSError: Too many open files`
- **Loss:** Smooth L1 with `beta=0.05m` (L2 below 5 cm, L1 above)
- **Optimizer:** AdamW with two param groups — backbone (`lr=1e-5`) and head (`lr=1e-4`); linear warmup → cosine decay; base LRs stored before epoch loop
- **AMP:** enabled by default (`torch.amp.GradScaler("cuda", ...)`); disable with `--no-amp`
- **Progress:** tqdm bars for train and val loops with live loss/MPJPE postfix
- **Checkpoints:** `best.pth` (lowest val MPJPE body) + `epoch_XXXX.pth` every N epochs
- **Logging:** TensorBoard scalars + validation video overlay (16 frames projected via intrinsic K); CSV `metrics.csv` with fixed column schema (`_CSV_FIELDNAMES`)
- **Metrics:** MPJPE in mm on all 127 joints, body joints (0:22), and hand joints (25:55)

## Critical Conventions

**Coordinate system (BEDLAM2 camera space):**
- X=forward (depth), Y=left, Z=up; units=metres
- Projection: `u = fx*(-Y/X)+cx`, `v = fy*(-Z/X)+cy`

**Rotation handling:**
- Raw portrait videos (`rotate_flag=True`, H=1280×W=720) are rotated CCW 90° at extraction time
- Depth NPZ is **already stored upright** — never rotate it
- Intrinsic K and joints_2d labels are always in the upright frame — never transform K

**Input resolution:** 640×384 (H×W) — portrait orientation for person crops; must be multiples of patch size 16

**Splits:** Always sequence-level (not frame-level) to prevent data leakage; seed=2026

**JPG-first:** Training requires pre-extracted frames under `data/frames/<folder>/<seq>/<idx:05d>.jpg`; run `scripts/extract_frames.py` once before any training.

**Depth NPY:** Pre-converted depth at `data/depth/npy/` (float16, 384×640) is used by default for fast mmap loading. Falls back to `data/depth/npz/` if NPY is missing. Run `scripts/convert_depth_npy.py` once after extraction.

## Relationship to Integrated Pipeline

This `claude_code/` directory is the **standalone prototype**. The model/data/training logic has been integrated into the main `pose/` module (see `pose/configs/sapiens_pose/bedlam2/`). Differences vs. prototype:
- Integrated version uses MMEngine config system and 70-joint subset (vs. 127 here)
- Integrated version predicts pelvis_depth + pelvis_uv as separate branches
- Data preprocessing scripts (`extract_frames.py`, `convert_depth_npy.py`) are still used from here even for integrated training

## Key File Paths
- Checkpoint: `checkpoints/sapiens_0.3b_epoch_1600_clean.pth`
- Data root: `/home/hang/repos_local/MMC/BEDLAM2Datatest/`
- Sapiens repo: `/home/hang/repos_local/MMC/sapiens/`
