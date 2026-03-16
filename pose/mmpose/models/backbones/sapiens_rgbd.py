# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Sapiens ViT backbone with 4-channel (RGB + Depth) patch embedding.

The only difference from the original Sapiens backbone is that the patch
embedding Conv2d accepts 4 input channels instead of 3.  All transformer
blocks are identical to the original.

When loading a pretrained RGB-only checkpoint, the depth channel weights
in the patch embedding are initialised as the mean of the 3 RGB channel
weights (see ``mmpose.models.utils.rgbd_weight_utils``).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from mmengine.model import BaseModule
from mmpretrain.models.backbones.vision_transformer import VisionTransformer

from mmpose.registry import MODELS
from mmpose.models.utils.rgbd_weight_utils import load_sapiens_pretrained_rgbd


@MODELS.register_module()
class SapiensBackboneRGBD(BaseModule):
    """Sapiens ViT with a 4-channel (RGB+D) patch embedding.

    Wraps ``mmpretrain.VisionTransformer`` with ``in_channels=4``,
    ``with_cls_token=False``, and ``out_type='featmap'``.

    Args:
        arch (str): One of ``sapiens_0.3b | sapiens_0.6b | sapiens_1b |
            sapiens_2b``.
        img_size (tuple[int, int]): ``(H, W)`` of the model input image in
            pixels.  Must be divisible by ``patch_size`` (16).
        drop_path_rate (float): Stochastic depth rate.
        pretrained (str, optional): Path to a Sapiens RGB pretrain checkpoint.
            If provided, weights are loaded with channel expansion and
            pos-embed interpolation during ``init_weights()``.
        init_cfg: Standard MMEngine init config (ignored when ``pretrained``
            is set; use one or the other).
    """

    def __init__(
        self,
        arch: str = 'sapiens_0.3b',
        img_size: Tuple[int, int] = (640, 384),
        drop_path_rate: float = 0.0,
        pretrained: Union[str, None] = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self._pretrained = pretrained

        self.vit = VisionTransformer(
            arch=arch,
            img_size=img_size,
            patch_size=16,
            in_channels=4,          # RGB + Depth
            qkv_bias=True,
            final_norm=True,
            drop_path_rate=drop_path_rate,
            with_cls_token=False,
            out_type='featmap',     # returns (B, C, H/16, W/16)
            patch_cfg=dict(padding=2),
        )

    def init_weights(self) -> None:
        """Load RGB pretrained weights with RGBD channel expansion."""
        super().init_weights()
        if self._pretrained is not None:
            # The weight loader expects model.backbone.vit; wrap self in a
            # thin nn.Module so state_dict/load_state_dict work correctly.
            class _Wrapper(torch.nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone
            load_sapiens_pretrained_rgbd(_Wrapper(self), self._pretrained,
                                         verbose=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward pass.

        Args:
            x: ``(B, 4, H, W)`` — concatenated RGB (normalised) + depth.

        Returns:
            Tuple with one element: feature map ``(B, C, H/16, W/16)``.
        """
        feats = self.vit(x)   # list[Tensor] when out_type='featmap'
        return (feats[0],)    # wrap in tuple for mmpose backbone convention
