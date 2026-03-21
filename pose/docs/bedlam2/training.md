# BEDLAM2 3D Pose: Training & Evaluation Guide

Trains a Sapiens 0.3B backbone to predict 70 root-relative SMPL-X joints + pelvis
depth + pelvis UV from RGBD crops.

---

## Environment

Use the `sapiens` conda environment for all steps.

```bash
conda activate sapiens
```

---

## Prerequisites

### 1. Data

Set the `BEDLAM2_DATA_ROOT` environment variable to your dataset path (all scripts
and the config read this automatically):

```bash
export BEDLAM2_DATA_ROOT=/path/to/bedlam2
```

Expected layout:

```
bedlam2/
  data/
    frames/     # RGB frames  (*.jpg / *.png)
    label/      # SMPL-X annotations (*.npz)
    depth/
      npy/      # Depth maps (*.npy, mmap-ready)
      npz/      # Depth maps (*.npz, fallback)
```

### 2. Pretrained checkpoint

Download the Sapiens 0.3B pretrained weights and place at:

```
pose/../pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth
```

Set in the config (`pretrained_checkpoint`) or override with `--cfg-options`.

### 3. Generate dataset splits (one-time)

```bash
cd /home/<user>/repos/sapiens/pose
python tools/generate_bedlam2_splits.py \
    --data-root ${BEDLAM2_DATA_ROOT} \
    --output-dir data/bedlam2_splits/
```

Produces `train_seqs.txt`, `val_seqs.txt`, `test_seqs.txt` (~1532 / 191 / 191 sequences).

---

## Training

### Quick start (single GPU)

```bash
cd /home/<user>/repos/sapiens/pose
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
    --work-dir Outputs/train/bedlam2/sapiens_0.3b-50e_bedlam2-640x384/node \
    --amp
```

### Shell script (recommended)

```bash
cd pose/scripts/finetune/bedlam2/sapiens_0.3b
bash node.sh
```

Edit `node.sh` to switch between modes:

| Variable | Options | Effect |
|---|---|---|
| `mode` | `multi-gpu` (default) | Distributed training via `dist_train.sh` |
| `mode` | `debug` | Single GPU, `num_workers=0`, no validation |
| `DEVICES` | `0,` / `0,1,2,3,` | GPU selection |
| `TRAIN_BATCH_SIZE_PER_GPU` | `2` (default) | Per-GPU batch size |
| `RESUME_FROM` | path to `.pth` | Resume interrupted run |

Checkpoints and logs are saved to:
```
pose/Outputs/train/bedlam2/sapiens_0.3b-50e_bedlam2-640x384/node/<timestamp>/
```

The best checkpoint is selected by `bedlam/mpjpe/body` (lower is better).

### Key hyperparameters (config)

| Parameter | Value |
|---|---|
| Input size | 640 × 384 (H × W) |
| Epochs | 50 |
| Optimizer | AdamW, base lr=1e-4, backbone lr=1e-5 |
| LR schedule | Linear warmup (500 steps) → Cosine annealing |
| Loss | SoftWeightSmoothL1Loss (β=0.05) for joints, depth, UV |
| Frame stride | 5 (sample every 5th frame) |

---

## Evaluation

Validation runs automatically after each epoch during training. To evaluate a
saved checkpoint manually:

```bash
cd /home/<user>/repos/sapiens/pose
python tools/test.py \
    configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
    Outputs/train/bedlam2/sapiens_0.3b-50e_bedlam2-640x384/node/<timestamp>/best_bedlam_mpjpe_body_epoch_*.pth
```

Reported metrics (in millimetres):

| Metric | Description |
|---|---|
| `bedlam/mpjpe/body` | MPJPE over body joints 0–21 (22 joints) |
| `bedlam/mpjpe/hand` | MPJPE over hand joints 24–53 (30 joints) |
| `bedlam/mpjpe/all` | MPJPE over all 70 active joints |

---

## Inference Demo

Runs inference on test sequences and saves side-by-side PNG visualisations:
- **Left panel** — RGB crop with 2D-projected joints (green = GT, red = predicted)
- **Right panel** — 3D scatter plot (green = GT, red = predicted)

### Shell script

```bash
cd pose/scripts/demo/local
bash bedlam2.sh
```

Update `CHECKPOINT` in `bedlam2.sh` to point to your trained model, e.g.:

```bash
CHECKPOINT=".../node/03-15-2026_18:14:31/best_bedlam_mpjpe_body_epoch_42.pth"
```

### Python directly

```bash
cd /home/<user>/repos/sapiens/pose
python demo/demo_bedlam2.py \
    configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
    /path/to/checkpoint.pth \
    --data-root ${BEDLAM2_DATA_ROOT} \
    --seq-paths-file data/bedlam2_splits/test_seqs.txt \
    --output-root Outputs/demo/bedlam2 \
    --num-samples 200 \
    --batch-size 8 \
    --device cuda:0
```

Output PNGs are saved to `--output-root/<seq_name>/<frame_idx>_body<body_idx>.png`.

---

## File Reference

| File | Purpose |
|---|---|
| `pose/configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py` | Main config (model, data, schedule) |
| `pose/tools/generate_bedlam2_splits.py` | One-time split generation |
| `pose/scripts/finetune/bedlam2/sapiens_0.3b/node.sh` | Training launch script |
| `pose/demo/demo_bedlam2.py` | Inference + visualisation script |
| `pose/scripts/demo/local/bedlam2.sh` | Demo launch script |
| `pose/data/bedlam2_splits/` | Train / val / test sequence lists |

For architecture and integration details see [bedlam2/integration.md](integration.md).
