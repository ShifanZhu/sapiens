# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> This is the `pose/` module of the Sapiens project. See the root `CLAUDE.md` for project-wide context, environment setup, and BEDLAM2 coordinate conventions.

## Editing Scope

Only modify files inside `pose/`. Do NOT change code outside the `pose/` folder (the rest of the sapiens repo). The only exception is `../docs/update_log/` for session logs. If a change outside `pose/` seems necessary, always ask the user for permission first.

## Session Convention

After every session where files are modified, append an entry to `../docs/update_log/YYYY-MM-DD.md` (create the file if it doesn't exist). List each changed file and briefly describe what was changed and why. Always include a brief summary section at the top of the file each time it is modified. Also update `../docs/update_log/README.md` if necessary (e.g. new date entry or summary changed significantly).

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

# Quick smoke-test train.py — limit to 1 sequence each for train/val
# max_seqs=1 avoids indexing the full dataset (the main bottleneck).
# Fixed startup cost (~20s) is unavoidable; this minimises everything else.
python tools/train.py configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384-transformer.py \
  --work-dir runs/test_verify \
  --cfg-options train_cfg.max_epochs=1 train_cfg.val_interval=1 \
    train_dataloader.dataset.max_seqs=1 \
    val_dataloader.dataset.max_seqs=1
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

### BEDLAM2 RGBD 3D Pipeline (custom)
- **Backbone:** `SapiensBackboneRGBD` (`mmpose/models/backbones/sapiens_rgbd.py`) — 4-channel (RGB+D) ViT
- **Head (default):** `Pose3dTransformerHead` (`mmpose/models/heads/regression_heads/pose3d_transformer_head.py`) — transformer decoder with per-joint query tokens + cross-attention; switch via `head.type` in config
- **Head (baseline):** `Pose3dRegressionHead` (`mmpose/models/heads/regression_heads/pose3d_regression_head.py`) — GAP + MLP, 3 output branches: 70 joints (×3 m), pelvis depth (m), pelvis UV (normalized)
- **Estimator:** `RGBDPose3dEstimator` (`mmpose/models/pose_estimators/rgbd_pose3d.py`) — skips 2D affine back-transform; joints are already in camera 3D space
- **Dataset:** `Bedlam2Dataset` (`mmpose/datasets/datasets/body3d/bedlam2_dataset.py`) — indexes `(label_path, body_idx, frame_idx)` triples from NPZ files
- **Metric:** MPJPE in mm on body/hand/all joint subsets (`mmpose/evaluation/metrics/bedlam_metric.py`)

### MMEngine Gotchas

**`Evaluator.process()` flattens metainfo.** Before calling `metric.process()`, MMEngine's `Evaluator` calls `data_sample.to_dict()` on each `PoseDataSample`. This flattens metainfo fields (e.g. `K`, `img_shape`) to the **top level** of the dict — they are NOT nested under a `'metainfo'` key. In metric `process()`, read them as `data_sample['K']`, not `data_sample['metainfo']['K']`.

**Do not put non-loss scalars in the losses dict.** MMEngine auto-logs every key returned by `model.train_step()` (i.e. everything in the losses dict) to TensorBoard as noisy per-iteration scalars. To log epoch-level metrics cleanly, store values as head attributes (e.g. `self._train_mpjpe`) and read them from a custom hook via `runner.model.head._train_mpjpe`.

**MessageHub loss keys are prefixed with `train/`.** `RuntimeInfoHook.after_train_iter` stores every key from `model.train_step()` outputs as `f'train/{key}'` in `message_hub.log_scalars`. When purging stale keys from a resumed checkpoint, use the prefixed form (e.g. `'train/mpjpe'`, not `'mpjpe'`).

**Purge stale MessageHub keys in `before_train`, not `before_run`.** `before_run` fires before `runner.load_or_resume()`, so the checkpoint will restore stale keys after the purge. `before_train` fires after `load_or_resume()` (inside `train_loop.run()`), so the purge sticks.

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
| `mmpose/models/heads/regression_heads/pose3d_transformer_head.py` | Transformer decoder head (default) |
| `mmpose/models/heads/regression_heads/pose3d_regression_head.py` | GAP+MLP regression head (baseline) |
| `mmpose/models/pose_estimators/rgbd_pose3d.py` | RGBD estimator (no 2D back-transform) |
| `mmpose/datasets/datasets/body3d/bedlam2_dataset.py` | BEDLAM2 dataset class |
| `mmpose/datasets/datasets/body3d/constants.py` | 70 active SMPL-X joint indices |
| `mmpose/evaluation/metrics/bedlam_metric.py` | MPJPE metric |
| `tools/generate_bedlam2_splits.py` | One-time split generation |
| `demo/demo_bedlam2.py` | BEDLAM2 inference demo script |
| `scripts/demo/local/bedlam2.sh` | Shell wrapper for demo |
| `scripts/finetune/bedlam2/` | Fine-tuning scripts |

### Docs Structure (`pose/docs/`)
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

Project-wide docs (task guides, finetune, update logs) are in `../docs/` — see root `CLAUDE.md` for structure.
