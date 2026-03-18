# PRD: Transformer Decoder Head for BEDLAM2 3D Pose

## Problem Statement

The current BEDLAM2 RGBD 3D pose head uses global average pooling (GAP) to collapse the ViT backbone's spatial feature map into a single vector before regressing 70 joint positions. This destroys all spatial information — the network cannot know *where* in the image each joint appears. The result is a body MPJPE of 80.6mm (0.3b) and 70.5mm (2b), with hand joints performing dramatically worse (130.5mm / 113.7mm) — consistent with spatial information loss for small, localized body parts.

## Solution

Replace the GAP+MLP head with a transformer decoder head that uses per-joint learnable query tokens and cross-attention to the spatial feature map. Each joint query attends to its relevant image region, preserving the spatial structure that GAP destroys. The design is deliberately minimal (1 decoder layer, single linear output projections) to isolate the impact of cross-attention and enable clean A/B comparison against the baseline.

## User Stories

1. As a researcher, I want a pose head that preserves spatial information from the backbone feature map, so that fine-grained joints (hands, feet) can attend to their specific image regions rather than relying on a global average.
2. As a researcher, I want per-joint query tokens that learn to specialize for each anatomical joint, so that the model can develop joint-specific attention patterns.
3. As a researcher, I want self-attention across joint queries, so that the model can learn implicit kinematic relationships (e.g., elbow informs wrist prediction) without an explicit skeleton graph.
4. As a researcher, I want the pelvis depth and UV predictions to come from the pelvis query token directly, so that pelvis localization uses the most semantically relevant representation.
5. As a researcher, I want explicit 2D sinusoidal positional encodings on spatial tokens, so that cross-attention has a clean spatial signal even after flattening the feature map.
6. As a researcher, I want the new head to be a drop-in replacement with the same loss/predict interface as the current head, so that no changes are needed to the estimator, data pipeline, or loss functions.
7. As a researcher, I want the new head registered as an MMEngine module, so that I can switch between heads by changing only the config file.
8. As a researcher, I want a unit test that verifies forward pass shapes and loss computation, so that I can catch regressions without running a full training job.
9. As a researcher, I want to train from scratch (pretrained backbone, random head) with the identical recipe as the baseline, so that the A/B comparison is fair and isolates the head architecture as the only variable.
10. As a researcher, I want a clear success criterion (body MPJPE ≤ 75mm on 0.3b), so that I can objectively decide whether the added complexity is justified.

## Implementation Decisions

### Modules to build/modify

1. **New module: Transformer decoder head** — A new head class (e.g., `Pose3dTransformerHead`) registered with `@MODELS.register_module()`. Contains:
   - Fixed 2D sinusoidal positional encoding (DETR-style, zero learnable parameters)
   - 70 learnable joint query embeddings of dimension `embed_dim`
   - 1 transformer decoder layer: self-attention (8 heads) → cross-attention (8 heads) → FFN (GELU activation)
   - Joint output: single `Linear(embed_dim, 3)` applied per token (shared weights) → `(B, 70, 3)` root-relative metres
   - Pelvis depth: `Linear(embed_dim, 1)` on decoder output token 0 → `(B, 1)`
   - Pelvis UV: `Linear(embed_dim, 2)` on decoder output token 0 → `(B, 2)`
   - Same `loss()` and `predict()` interface as the existing `Pose3dRegressionHead`

2. **New config file** — A new config (or modified copy of the existing BEDLAM2 config) that points to the new head type. The only difference from the baseline config should be the `head=dict(type='Pose3dTransformerHead', ...)` block. Training recipe (optimizer, LR schedule, epochs, data pipeline) remains identical.

3. **Update `custom_imports`** — The new config must include the new head module in `custom_imports` so it registers before the runner builds.

### Architecture details

- **Input:** backbone feature map `(B, embed_dim, 40, 24)`, flattened to `(B, 960, embed_dim)` spatial tokens
- **Positional encoding:** 2D sine/cosine (DETR-style), added to spatial tokens before cross-attention. Encodes each token's `(row, col)` position in the 40×24 grid.
- **Decoder layer order:** self-attention over 70 queries → cross-attention (queries attend to spatial tokens) → FFN with residual connections and layer norm
- **Joint queries predict root-relative directly.** No subtraction inside the head — root subtraction stays in the `SubtractRootJoint` data transform.
- **Pelvis branches use query token 0** (pelvis joint), not mean-pooled output. This is more semantically coherent.
- **Single linear output projections everywhere.** The decoder's cross-attention already routes spatial information; a `Linear(1024, 3)` suffices. If insufficient, MLP depth is a straightforward follow-up.

### Parameter count (sapiens_0.3b, embed_dim=1024)

| Component | Parameters |
|-----------|-----------|
| 2D positional encoding | 0 (fixed) |
| Joint query embeddings | ~72K |
| 1 decoder layer (self-attn + cross-attn + FFN) | ~12.6M |
| Joint projection | ~3K |
| Pelvis depth + UV projections | ~3K |
| **Total** | **~13M** |

Replaces the old head (~4M params), remains small relative to the 300M backbone.

### Training plan

- Train from scratch: load pretrained Sapiens backbone, randomly initialize head
- Identical recipe: AdamW, 1e-4 head LR / 1e-5 backbone LR, weight decay 0.03, 3-epoch linear warmup (start_factor=0.333), cosine decay to 0, 50 epochs
- AMP enabled (FixedAmpOptimWrapper), grad clip max_norm=1.0
- Batch size 16, same data pipeline and augmentations

### Baseline and success criterion

| Model | Body MPJPE | Hand MPJPE | All MPJPE |
|-------|-----------|-----------|----------|
| 0.3b (GAP+MLP baseline) | 80.6 mm | 130.5 mm | 117.6 mm |
| 2b (GAP+MLP baseline) | 70.5 mm | 113.7 mm | 102.1 mm |

**Success:** body MPJPE ≤ 75mm on 0.3b (≥5mm / ~7% relative improvement).

## Testing Decisions

### What makes a good test

Tests should verify external behavior (input/output contracts), not implementation details. A head test should confirm that given a feature tensor of the expected shape, the module produces correctly shaped outputs, computes non-NaN scalar losses, and returns properly formatted predictions. It should NOT test internal layer dimensions, attention patterns, or weight values.

### Modules to test

1. **`Pose3dTransformerHead`** — unit test covering:
   - `forward()`: given `(B, embed_dim, H, W)` feature tuple, outputs dict with `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)`
   - `loss()`: given features and mock `batch_data_samples`, returns dict of finite scalar losses (`loss_joints`, `loss_depth`, `loss_uv`, `mpjpe`)
   - `predict()`: returns list of `InstanceData` with correct attribute shapes
   - Various `embed_dim` values (1024, 1280) to verify it generalizes across model sizes

### Test location

`pose/tests/test_models/test_heads/test_pose3d_transformer_head.py` — this creates the test directory structure for the first time.

### Prior art

No existing tests in `pose/tests/`. The test will use standard pytest with `torch.randn` inputs and mock MMEngine `InstanceData`/`DataSample` objects, following patterns common in OpenMMLab test suites.

## Out of Scope

- Changes to the backbone, data pipeline, estimator, loss functions, or inference demo
- Multi-layer decoder ablations (1 vs 2+ layers) — follow-up if results warrant it
- MLP output head ablations (linear vs multi-layer) — follow-up if results warrant it
- Diagnostic analysis of baseline errors (patch resolution, data distribution, occlusion hypotheses)
- Training on the 2b model — initial experiment is 0.3b only
- Hyperparameter tuning (LR, warmup, decoder width) — first run uses identical recipe

## Further Notes

- If the transformer decoder head performs worse than or equal to the baseline, the plan is to reassess rather than iterate blindly on head architecture. The real bottleneck may be elsewhere (data, resolution, depth input quality).
- The design intentionally starts minimal. Each subsequent complexity addition (more layers, larger MLPs, separate pelvis queries) can be tested independently with a clear attribution of impact.
- The existing `Pose3dRegressionHead` is preserved — configs can switch between heads by changing `type` in the head config dict.
