# Pose Documentation

## Doc Layers

```
design/   — Current state. What exists, how it works. Read this to understand the system.
bedlam2/  — Operational guides. How to train, evaluate, and understand results.
prd/      — Proposals for future changes (frozen once implemented).
prd/issues/ — Discrete implementation tasks derived from PRDs.
```

**Reading order for new contributors:** `bedlam2/README.md` → `design/pipeline.md` → rest of design/ → prd/ only if curious about design rationale.

---

## Design

- [pipeline.md](design/pipeline.md) — Model architecture + inference pipeline *(start here)*
- [data_transforms.md](design/data_transforms.md) — Data format, coordinate system, transform chain
- [training_loop.md](design/training_loop.md) — Optimizer, loss, metrics, TensorBoard tags, checkpointing
- [dataload.md](design/dataload.md) — Depth NPY conversion and loading performance
- [visualization.md](design/visualization.md) — Visualization hook and demo rendering
- [attention_pooling_pelvis.md](design/attention_pooling_pelvis.md) — RFC: attention-pooled pelvis localization
- [mpjpe_logging_investigation.md](design/mpjpe_logging_investigation.md) — Investigation: invariant training MPJPE

## BEDLAM2

- [README.md](bedlam2/README.md) — Start here: navigation hub and reading order
- [training.md](bedlam2/training.md) — Training, evaluation, and inference guide
- [integration.md](bedlam2/integration.md) — Architecture mapping, file locations, design decisions
- [training_results.md](bedlam2/training_results.md) — Per-epoch metrics for completed runs

## PRDs

- [transformer_decoder_head.md](prd/transformer_decoder_head.md) — Transformer decoder head for 3D pose
- [tensorboard_restructure.md](prd/tensorboard_restructure.md) — TensorBoard logging restructure

### Implementation Issues

- [001 — Transformer decoder head module](prd/issues/001_transformer_decoder_head_module.md)
- [002 — Training config smoke test](prd/issues/002_training_config_smoke_test.md)
- [003 — A/B training evaluation](prd/issues/003_ab_training_evaluation.md)
- [004 — Restructure TensorBoard tags](prd/issues/004_restructure_tags.md)
- [005 — Absolute MPJPE and epoch averaging](prd/issues/005_absolute_mpjpe_and_epoch_avg.md)
- [006 — BEDLAM2 transform testability (RFC)](prd/issues/006_bedlam2_transform_testability.md)
