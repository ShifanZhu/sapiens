"""Sapiens ViT backbone modified for 4-channel (RGB + Depth) input.

The only change vs. the original Sapiens backbone is that the patch-embedding
Conv2d accepts 4 input channels instead of 3.  All transformer blocks are
identical to the original.

When loading a pretrained RGB-only checkpoint, the depth channel weights in
the patch embedding are initialised as the mean of the 3 RGB channel weights
(see `weights.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add sapiens sub-packages to path so mmpretrain can be imported.
# claude_code/ is one level below the sapiens repo root.
_sapiens_root = Path(__file__).resolve().parent.parent.parent
for p in [str(_sapiens_root / "pretrain"), str(_sapiens_root / "engine")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mmpretrain.models.backbones.vision_transformer import VisionTransformer  # noqa: E402


# Architecture configs mirroring the original Sapiens pretrain configs
SAPIENS_ARCHS = {
    "sapiens_0.3b": dict(embed_dim=1024, num_layers=24),
    "sapiens_0.6b": dict(embed_dim=1280, num_layers=32),
    "sapiens_1b":   dict(embed_dim=1536, num_layers=40),
    "sapiens_2b":   dict(embed_dim=1920, num_layers=48),
}


class SapiensBackboneRGBD(nn.Module):
    """Sapiens ViT with a 4-channel (RGB+D) patch embedding.

    Args:
        arch:   One of ``sapiens_0.3b | sapiens_0.6b | sapiens_1b | sapiens_2b``.
        img_size: ``(H, W)`` of the model input image in pixels.
                  Must be divisible by ``patch_size`` (16).
        drop_path_rate: Stochastic depth rate for the transformer blocks.
    """

    def __init__(
        self,
        arch: str = "sapiens_0.3b",
        img_size: tuple[int, int] = (384, 640),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if arch not in SAPIENS_ARCHS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(SAPIENS_ARCHS)}")

        self.arch = arch
        self.embed_dim: int = SAPIENS_ARCHS[arch]["embed_dim"]

        self.vit = VisionTransformer(
            arch=arch,
            img_size=img_size,      # (H, W)
            patch_size=16,
            in_channels=4,          # RGB + Depth
            qkv_bias=True,
            final_norm=True,
            drop_path_rate=drop_path_rate,
            with_cls_token=False,
            out_type="featmap",     # returns (B, C, H/16, W/16)
            patch_cfg=dict(padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 4, H, W)`` — concatenated RGB + depth tensor.

        Returns:
            Feature map ``(B, embed_dim, H/16, W/16)``.
        """
        feats = self.vit(x)   # list with one element when out_type='featmap'
        return feats[0]
