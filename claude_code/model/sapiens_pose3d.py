"""SapiensPose3D: full model combining the 4-channel backbone and 3D head.

Input  : (B, 4, H, W)  — RGB (channels 0-2) + Depth (channel 3)
Output : dict with:
    joints:       (B, 127, 3)  — root-relative XYZ (metres)
    pelvis_depth: (B, 1)       — pelvis forward distance (metres)
    pelvis_uv:    (B, 2)       — pelvis (u, v) in crop pixel coordinates
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import SapiensBackboneRGBD, SAPIENS_ARCHS
from .head import Pose3DHead


class SapiensPose3D(nn.Module):
    """Sapiens-RGBD model for 3D human pose estimation.

    Args:
        arch:          Sapiens model size (``sapiens_0.3b`` etc.).
        img_size:      ``(H, W)`` input image size in pixels.
        num_joints:    Number of output joints (127 for SMPL-X).
        head_hidden:   Hidden dimension of the regression MLP.
        head_dropout:  Dropout in the regression head.
        drop_path_rate: Stochastic depth rate for ViT blocks.
    """

    def __init__(
        self,
        arch: str = "sapiens_0.3b",
        img_size: tuple[int, int] = (640, 384),
        num_joints: int = 127,
        head_hidden: int = 2048,
        head_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        embed_dim = SAPIENS_ARCHS[arch]["embed_dim"]

        self.backbone = SapiensBackboneRGBD(
            arch=arch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        self.head = Pose3DHead(
            in_channels=embed_dim,
            num_joints=num_joints,
            hidden_dim=head_hidden,
            dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: ``(B, 4, H, W)`` — concatenated [RGB | depth] tensor.

        Returns:
            Dict with ``joints`` (B, num_joints, 3), ``pelvis_depth`` (B, 1),
            ``pelvis_uv`` (B, 2).
        """
        feat = self.backbone(x)   # (B, embed_dim, H/16, W/16)
        return self.head(feat)

    def load_pretrained(self, ckpt_path: str, verbose: bool = True) -> None:
        """Load Sapiens RGB pretrained backbone weights.

        See ``model.weights.load_sapiens_pretrained`` for details.
        """
        from .weights import load_sapiens_pretrained
        load_sapiens_pretrained(self, ckpt_path, verbose=verbose)
