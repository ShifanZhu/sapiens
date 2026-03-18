# Pose Documentation

## Design
- [pipeline.md](design/pipeline.md) — End-to-end model pipeline and training flow
- [dataload.md](design/dataload.md) — Dataset loading and preprocessing
- [visualization.md](design/visualization.md) — Visualization hooks and demo rendering

## BEDLAM2
- [integration.md](bedlam2/integration.md) — Migration from standalone prototype to mmpose
- [training.md](bedlam2/training.md) — Training and evaluation guide

## PRDs
- [transformer_decoder_head.md](prd/transformer_decoder_head.md) — Transformer decoder head for 3D pose
- [tensorboard_restructure.md](prd/tensorboard_restructure.md) — TensorBoard logging restructure

### Implementation Issues
- [001 — Transformer decoder head module](prd/issues/001_transformer_decoder_head_module.md)
- [002 — Training config smoke test](prd/issues/002_training_config_smoke_test.md)
- [003 — A/B training evaluation](prd/issues/003_ab_training_evaluation.md)
- [004 — Restructure TensorBoard tags](prd/issues/004_restructure_tags.md)
- [005 — Absolute MPJPE and epoch averaging](prd/issues/005_absolute_mpjpe_and_epoch_avg.md)
