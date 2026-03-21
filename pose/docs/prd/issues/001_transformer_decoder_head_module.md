# Issue 1: Transformer decoder head module + unit tests

**Type:** AFK
**Blocked by:** None — can start immediately

## Parent PRD

`pose/docs/prd/transformer_decoder_head.md`

## What to build

Implement `Pose3dTransformerHead` as a drop-in replacement for the existing `Pose3dRegressionHead`. The new head uses a transformer decoder with per-joint query tokens and cross-attention to the backbone's spatial feature map, replacing global average pooling.

Architecture (see PRD Section "Implementation Decisions" for full details):
- Flatten backbone output `(B, embed_dim, H, W)` → spatial tokens, add fixed 2D sinusoidal positional encoding (DETR-style)
- 70 learnable joint query embeddings
- 1 transformer decoder layer: self-attention (8 heads) → cross-attention (8 heads) → FFN (GELU)
- `Linear(embed_dim, 3)` per token (shared) → joints `(B, 70, 3)`
- `Linear(embed_dim, 1)` on token 0 → pelvis_depth `(B, 1)`
- `Linear(embed_dim, 2)` on token 0 → pelvis_uv `(B, 2)`

Must expose the same `forward()`, `loss()`, and `predict()` interface as `Pose3dRegressionHead`. Register with `@MODELS.register_module()`.

Unit tests at `pose/tests/test_models/test_heads/test_pose3d_transformer_head.py`.

**Status: COMPLETE** (commit 1800b43)

## Acceptance criteria

- [x] `Pose3dTransformerHead` class implemented and registered via `@MODELS.register_module()`
- [x] `forward()` accepts `Tuple[Tensor]` (backbone features), returns dict with `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)`
- [x] `loss()` accepts features + `batch_data_samples`, returns dict of finite scalar losses (`loss_joints`, `loss_depth`, `loss_uv`, `mpjpe`)
- [x] `predict()` returns list of `InstanceData` with `keypoints (1, J, 3)`, `keypoint_scores (1, J)`, `pelvis_depth (1,)`, `pelvis_uv (1, 2)`
- [x] 2D sinusoidal positional encoding is fixed (no learnable parameters)
- [x] Pelvis branches read from decoder output token 0 (not mean-pooled)
- [x] Joint queries predict root-relative directly (no subtraction in head)
- [x] Unit tests pass for `embed_dim=1024` and `embed_dim=1280`
- [x] Unit tests verify output shapes, finite losses, and predict format

## User stories addressed

- User story 1: spatial information preserved via cross-attention
- User story 2: per-joint query tokens
- User story 3: self-attention for implicit kinematics
- User story 4: pelvis branches from pelvis query token
- User story 5: 2D sinusoidal positional encodings
- User story 6: drop-in replacement with same interface
- User story 7: registered as MMEngine module
- User story 8: unit tests
