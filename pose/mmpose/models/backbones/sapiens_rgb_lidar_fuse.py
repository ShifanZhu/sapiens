# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Sapiens ViT (RGB patches) + LiDAR-HMR geometry tokens in one transformer, then
# feature map for the BEDLAM2 3D pose head. Expects ``(B, 4, H, W)`` inputs like
# ``SapiensBackboneRGBD`` but uses **only the first 3 channels** for the ViT; depth
# is already encoded via ``point_clouds`` from ``BedlamDepthToPointCloud``.

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmpretrain.models.backbones.vision_transformer_with_depth import (
    VisionTransformerWithDepth,
)

from mmpose.models.utils.rgbd_weight_utils import load_sapiens_pretrained_rgb_fusion_vit
from mmpose.registry import MODELS


@MODELS.register_module()
class SapiensBackboneRGBLidarFuse(BaseModule):
    """RGB Sapiens ViT with optional LiDAR-HMR tokens fused in-transformer.

    Args:
        arch: Sapiens ViT arch name (e.g. ``sapiens_0.3b``).
        img_size: ``(H, W)`` input size (must match crop, divisible by 16).
        drop_path_rate: Stochastic depth.
        pretrained: Path to RGB Sapiens checkpoint (3-channel patch embed).
        use_lidar_hmr_depth: If True, build ``lidar_hmr_encoder`` and fuse tokens.
        lidar_hmr_encoder: Config dict for ``LiDARHMRPoseMeshGraphormerEncoder``.
        depth_embed_dim: LiDAR-HMR channel width (48).
        default_num_depth_tokens: Fallback token count when points are missing.
        init_cfg: MMEngine init config.
    """

    def __init__(
        self,
        arch: str = 'sapiens_0.3b',
        img_size: Tuple[int, int] = (640, 384),
        drop_path_rate: float = 0.0,
        pretrained: Union[str, None] = None,
        use_lidar_hmr_depth: bool = True,
        lidar_hmr_encoder: Optional[dict] = None,
        depth_embed_dim: int = 48,
        default_num_depth_tokens: int = 26,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self._pretrained = pretrained

        if use_lidar_hmr_depth and not lidar_hmr_encoder:
            raise ValueError(
                'use_lidar_hmr_depth=True requires lidar_hmr_encoder config.')

        self.vit = VisionTransformerWithDepth(
            arch=arch,
            img_size=img_size,
            patch_size=16,
            in_channels=3,
            qkv_bias=True,
            final_norm=True,
            drop_path_rate=drop_path_rate,
            with_cls_token=False,
            out_type='featmap',
            patch_cfg=dict(padding=2),
            depth_embed_dim=depth_embed_dim,
            depth_embed_path=None,
            use_depth_projection=True,
            depth_projection_type='linear',
            default_num_depth_tokens=default_num_depth_tokens,
            use_lidar_hmr_depth=use_lidar_hmr_depth,
            lidar_hmr_encoder=lidar_hmr_encoder,
        )

    def init_weights(self) -> None:
        super().init_weights()
        if self._pretrained is not None:
            class _Wrapper(nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone

            load_sapiens_pretrained_rgb_fusion_vit(
                _Wrapper(self), self._pretrained, verbose=True)

    def forward(
        self,
        x: torch.Tensor,
        point_clouds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Args:
            x: ``(B, 4, H, W)`` RGB+depth batch (RGB normalised like BEDLAM2; depth ignored here).
            point_clouds: Optional ``(B, N, 3)`` for LiDAR-HMR (pelvis-relative).
        """
        rgb = x[:, :3].contiguous()
        feats = self.vit(rgb, point_clouds=point_clouds)
        return (feats[0],)
