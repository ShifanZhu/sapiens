
"""Utilities to convert LiDAR-HMR *pose_meshgraphormer* outputs into depth token tensors.

LiDAR-HMR's ``pose_meshgraphormer`` returns ``vert_feat`` with shape ``(B, 6890, 48)``
(per-vertex features before the final mesh regression head). That tensor is a natural
replacement for Point-MAE-style depth latents when fusing with
:class:`~mmpretrain.models.backbones.vision_transformer_with_depth.VisionTransformerWithDepth`.

This module does **not** import LiDAR-HMR (heavy deps, cwd-relative paths). Run LiDAR-HMR
in its own tree or a side process, then pass tensors here or save ``.npy`` files for
``depth_embed_path``.

Example (48-D tokens for ``VisionTransformerWithDepth(depth_embed_dim=48)``):

    >>> import torch
    >>> from mmpretrain.models.utils.lidar_hmr_tokens import (
    ...     LIDAR_HMR_VERT_FEAT_DIM,
    ...     pool_vert_feat_to_depth_tokens,
    ... )
    >>> vert_feat = torch.randn(2, 6890, LIDAR_HMR_VERT_FEAT_DIM)
    >>> tokens = pool_vert_feat_to_depth_tokens(
    ...     vert_feat, num_tokens=26, out_dim=LIDAR_HMR_VERT_FEAT_DIM)
    >>> tokens.shape
    torch.Size([2, 26, 48])
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS

# Documented shape from LiDAR-HMR pose_meshgraphormer (6890 SMPL vertices, 48-D features).
LIDAR_HMR_VERT_FEAT_DIM = 48
LIDAR_HMR_VERT_COUNT = 6890


def pool_vert_feat_to_depth_tokens(
    vert_feat: torch.Tensor,
    num_tokens: int,
    out_dim: int = 1024,
    projector: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Pool per-vertex LiDAR-HMR features to a fixed token count and optional width.

    Args:
        vert_feat: Tensor of shape ``(B, V, C)`` with ``V`` vertices (6890 for LiDAR-HMR)
            and ``C == LIDAR_HMR_VERT_FEAT_DIM`` typically.
        num_tokens: Target sequence length (e.g. 26 to match prior Point-MAE depth token count).
        out_dim: Output channel dimension (matches typical ``depth_embed_dim`` for ViT fusion).
        projector: If provided, applied to pooled features (e.g. ``nn.Linear(C, out_dim)``).
            If None, a trainable linear map is created only when ``C != out_dim`` is needed;
            for a fixed deployment, build ``projector`` once and reuse.

    Returns:
        Tensor of shape ``(B, num_tokens, out_dim)`` suitable for ``depth_embeddings`` in
        :meth:`VisionTransformerWithDepth.forward`.
    """
    if vert_feat.dim() != 3:
        raise ValueError(f'vert_feat must be (B, V, C), got {vert_feat.shape}')
    b, v, c = vert_feat.shape
    # Adaptive average pool along vertex axis: (B, C, V) -> (B, C, num_tokens)
    x = vert_feat.transpose(1, 2).contiguous()
    x = torch.nn.functional.adaptive_avg_pool1d(x, num_tokens)
    x = x.transpose(1, 2).contiguous()  # (B, num_tokens, C)
    if projector is not None:
        return projector(x)
    if c != out_dim:
        raise ValueError(
            f'Channel mismatch: vert_feat has C={c}, out_dim={out_dim}. '
            'Pass a LiDARHMRVertFeatProjector (or other nn.Module) as `projector`.')
    return x


@MODELS.register_module()
class LiDARHMRVertFeatProjector(nn.Module):
    """Learnable map from pooled LiDAR-HMR vertex features to ViT ``depth_embed_dim``.

    Use after :func:`pool_vert_feat_to_depth_tokens` without the built-in linear, or on raw
    pooled features of shape ``(B, T, C)``.
    """

    def __init__(
        self,
        in_dim: int = LIDAR_HMR_VERT_FEAT_DIM,
        out_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(B, T, in_dim)``
        Returns:
            ``(B, T, out_dim)``
        """
        return self.proj(x)
