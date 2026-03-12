"""3D pose regression head for Sapiens-RGBD.

Takes the ViT feature map (B, C, H', W') and regresses
(B, num_joints, 3) camera-space coordinates in the
(X=forward, Y=left, Z=up) convention of BEDLAM2.

Architecture:
    Feature map (B, C, H', W')
        → AdaptiveAvgPool2d(1)          (B, C, 1, 1)
        → flatten                        (B, C)
        → Linear(C, hidden)              (B, hidden)
        → LayerNorm + GELU
        → Linear(hidden, num_joints*3)   (B, num_joints*3)
        → reshape                        (B, num_joints, 3)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Pose3DHead(nn.Module):
    """Lightweight MLP regression head for 3D joint prediction.

    Args:
        in_channels:  Embedding dimension from the backbone (e.g. 1024).
        num_joints:   Number of output joints (127 for SMPL-X).
        hidden_dim:   Width of the hidden MLP layer.
        dropout:      Dropout probability applied before the final linear.
    """

    def __init__(
        self,
        in_channels: int,
        num_joints: int = 127,
        hidden_dim: int = 2048,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_joints = num_joints

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_joints * 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: ``(B, C, H', W')`` backbone feature map.

        Returns:
            ``(B, num_joints, 3)`` camera-space joint coordinates (metres).
        """
        x = self.pool(feat).flatten(1)       # (B, C)
        x = self.mlp(x)                      # (B, num_joints*3)
        return x.view(x.size(0), self.num_joints, 3)
