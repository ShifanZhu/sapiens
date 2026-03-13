# Sapiens RGBD 3D Pose (BEDLAM2)

PyTorch project for **single-frame 3D human pose estimation** (SMPL-X 127 joints) from **RGB + depth**.

- Backbone: Sapiens ViT adapted to 4-channel input (RGBD)
- Head: MLP regressor to camera-space XYZ joints
- Data: BEDLAM2-style labels/depth/video structure
- Outputs: checkpoints, CSV metrics, TensorBoard logs

---

## Project Structure

- `train.py` — training loop, validation, checkpointing, TensorBoard
- `extract_frames.py` — pre-extract JPG frames from MP4 for fast loading
- `test_data_pipeline.py` — data pipeline smoke test
- `test_model.py` — model forward/load smoke test
- `demo.py` — visualization utility
- `verify_rotation.py` — geometry/rotation sanity visualization
- `data/` — dataset, transforms, split logic, constants
- `model/` — backbone/head/model wrapper + pretrained loading

---

## Data Layout (expected)

Given `--data-root <DATA_ROOT>`, code expects:

- `<DATA_ROOT>/data/overview.txt`
- `<DATA_ROOT>/data/label/<folder>/<seq>.npz`
- `<DATA_ROOT>/data/depth/npz/<folder>/<seq>.npz`
- `<DATA_ROOT>/data/mp4/<folder>_mp4/<folder>/mp4/<seq>.mp4` (needed only for extraction)
- `<DATA_ROOT>/data/frames/<folder>/<seq>/<frame_idx:05d>.jpg` (used during training)

---

## Environment

Use your existing conda env (examples below assume `sapiens_gpu`):

```bash
conda run -n sapiens_gpu python -V
```

Core runtime deps used by this repo include: `torch`, `numpy`, `opencv-python`, `tensorboard`, and Sapiens/MMPretrain dependencies (see `model/backbone.py` path assumptions).

---

## Quick Start

### 1) Extract JPG frames (one-time)

```bash
conda run -n sapiens_gpu python extract_frames.py \
  --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
  --workers 8 \
  --quality 95
```

### 2) Smoke test data pipeline

```bash
conda run -n sapiens_gpu python test_data_pipeline.py
```

### 3) Smoke test model

```bash
conda run -n sapiens_gpu python test_model.py
```

### 4) Train

```bash
conda run -n sapiens_gpu python train.py \
  --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
  --pretrain checkpoints/sapiens_0.3b_epoch_1600_clean.pth \
  --output-dir runs/exp001
```

---

## Training Notes

- Current pipeline is **JPG-first / JPG-required** for RGB loading.
- Training split filtering does **not require MP4** (`mp4_required=False`).
- If a JPG frame is missing, dataset raises a clear fail-fast error.
- Backbone + head are both trainable by default (backbone uses lower LR).

Useful args in `train.py`:

- `--batch-size`, `--num-workers`
- `--img-h`, `--img-w`
- `--lr-backbone`, `--lr-head`
- `--epochs`, `--amp/--no-amp`
- `--resume <ckpt>`

---

## Monitoring

TensorBoard logs are written to `<output-dir>/tb`.

```bash
tensorboard --logdir runs
```

Metrics CSV is saved at `<output-dir>/metrics.csv`.

---

## Troubleshooting

- **`ModuleNotFoundError: torch`**: run commands inside the correct conda env.
- **Missing JPG frame error**: re-run `extract_frames.py` for the dataset root.
- **Sapiens import/path errors**: verify local Sapiens repo paths referenced in `model/backbone.py`.
- **Slow training**: tune `--num-workers`, ensure frames are on fast storage (SSD/NVMe), and keep AMP enabled.

---

## Existing Artifacts

- Pretrained checkpoint example: `checkpoints/sapiens_0.3b_epoch_1600_clean.pth`
- Example runs: `runs/exp001`, `runs/test_tb`
