# Design: Attention Pooling for Pelvis Localization

## Overview

In the Sapiens Pose top-down pipeline, 3D joints are predicted in **root-relative** coordinates, and the **pelvis** (the root) is localized separately in absolute camera space. This document proposes using **Attention Pooling** across all joint tokens to predict the pelvis, rather than relying on a single dedicated pelvis token.

## The Problem: Single Point of Failure

The current `Pose3dTransformerHead` uses query token 0 (the pelvis joint) to regress `pelvis_depth` and `pelvis_uv`.

*   **Occlusion Vulnerability**: If the person's midsection is occluded (e.g., by a table, an arm, or another person), the features in the pelvis token may become noisy or "weak."
*   **Drift**: Since the absolute position of the *entire* 70-joint skeleton is anchored to this one predicted point, even a small error in the pelvis token's prediction causes the entire person to drift in 3D space.

> **Note:** The "single point of failure" framing somewhat overstates the problem. In the transformer decoder, the pelvis query token attends (via self-attention between queries, and cross-attention to all image patch features) to the entire feature map — it is not purely a local patch regression. However, the occlusion vulnerability is still real: if the pelvis region is occluded, the decoder may produce a weaker/noisier representation for that token even with global attention, because the relevant image features simply aren't present.

## The Solution: Attention Pooling

Instead of a single-token bottleneck, we aggregate information from all 70 joint tokens into a **Global Feature** before regressing the pelvis coordinates.

### 1. Mechanism: Learnable Attention
We use a small Attention Pooling module to determine which joints are most "trustworthy" or predictive of the person's location in the current frame.

1.  **Scoring**: Each of the 70 decoded tokens $T_i$ is projected to a scalar weight $s_i$ using a linear layer:
    $$s_i = \text{Linear}(T_i)$$
2.  **Normalization**: Weights are normalized across all tokens using Softmax:
    $$w_i = \frac{\exp(s_i)}{\sum_{j=1}^{70} \exp(s_j)}$$
3.  **Aggregation**: A Global Feature $G$ is computed as the weighted sum:
    $$G = \sum_{i=1}^{70} w_i \cdot T_i$$
4.  **Regression**: `pelvis_depth` and `pelvis_uv` are predicted from $G$ via linear layers.

### 2. Why this is more robust

*   **Redundancy**: If the pelvis is occluded but the head and feet are visible, the Attention Pooling module can learn to "weigh" the head and feet tokens more heavily to infer the person's global depth and position.
*   **Depth Cues**: In 2D images, the best indicators of depth ($X$) are often the feet (position on the ground plane) and the overall scale of the body. Attention pooling allows the model to leverage these global cues for the pelvis prediction.
*   **Structural Consensus**: It turns the localization problem into a "global consensus" task rather than a "local patch regression" task.

## Implementation Details

### Architecture Update
The `Pose3dTransformerHead` will be updated with the following components:

| Module | Input | Output |
|--------|-------|--------|
| `pooling_proj` | `(B, 70, C)` | `(B, 70, 1)` (Attention scores) |
| `Weighted Sum` | `(B, 70, C)` | `(B, C)` (Global Feature) |
| `depth_out` | `(B, C)` | `(B, 1)` (Pelvis depth) |
| `uv_out` | `(B, C)` | `(B, 2)` (Pelvis UV) |

### Training
The module is trained end-to-end with the existing `L_depth` and `L_uv` losses. No changes to the ground truth or data pipeline are required.

> **Risk — Gradient Entanglement:** The pelvis depth/UV loss will backprop through all 70 joint tokens via the pooling module. Those tokens are primarily supervised to regress their own root-relative positions; mixing in the pelvis localization gradient could interfere. Monitor `train/joint_loss` closely after this change. If joint accuracy degrades, apply `stop_gradient` on the joint tokens before the `pooling_proj` layer so the pelvis head sees their features but cannot modify their gradients.

### Alternative: Dedicated Global Token

A cleaner architectural alternative is to add a separate **global/CLS query token** to the decoder (e.g., as token 0, shifting existing joint queries by 1). This token is not tasked with any specific joint — it attends freely to all image features and is solely supervised by `L_depth` and `L_uv`. This avoids the gradient entanglement issue entirely. The trade-off is a small increase in decoder sequence length and the need to re-index joint outputs.

## Success Criteria
*   **Stability**: Reduced "jitter" in absolute 3D position across video sequences.
*   **Occlusion Handling**: Lower absolute MPJPE in frames where the pelvis joint is marked as "not visible" in ground truth.
*   **Accuracy**: Improvement in absolute MPJPE without degrading root-relative joint accuracy.

> **Note on evaluation:** The "occlusion handling" criterion requires splitting the eval set by pelvis visibility in ground truth. Verify that `Bedlam2Dataset` exposes per-joint visibility flags and that `BedlamMetric` can filter by them, otherwise this criterion cannot be measured.
