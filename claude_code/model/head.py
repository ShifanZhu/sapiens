"""3D pose regression head for Sapiens-RGBD.

Takes the ViT feature map (B, C, H', W') and regresses:
  - (B, num_joints, 3) root-relative joint coordinates
  - (B, 1)             pelvis depth in metres (forward distance)
  - (B, 2)             pelvis 2D position in crop pixel coordinates

Architecture:
    Feature map (B, C, H', W')
        → AdaptiveAvgPool2d(1)          (B, C, 1, 1)
        → flatten                        (B, C)
        → Linear(C, hidden)              (B, hidden)
        → LayerNorm + GELU
        ├→ Linear(hidden, num_joints*3)  → joints_rel  (B, num_joints, 3)
        ├→ Linear(hidden, 1)             → pelvis_depth (B, 1)
        └→ Linear(hidden, 2)             → pelvis_uv    (B, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Pose3DHead(nn.Module):
    """MLP regression head for 3D joint prediction + pelvis localization.

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

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific output branches
        self.joints_out = nn.Linear(hidden_dim, num_joints * 3)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out = nn.Linear(hidden_dim, 2)

        self._init_weights()

    def _init_weights(self):
        for m in [self.trunk, self.joints_out, self.depth_out, self.uv_out]:
            for sub in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(sub, nn.Linear):
                    nn.init.trunc_normal_(sub.weight, std=0.02)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            feat: ``(B, C, H', W')`` backbone feature map.

        Returns:
            Dict with:
                ``joints``:       ``(B, num_joints, 3)`` root-relative coords (metres)
                ``pelvis_depth``: ``(B, 1)`` pelvis forward distance (metres)
                ``pelvis_uv``:    ``(B, 2)`` pelvis (u, v) in crop pixels
        """
        x = self.pool(feat).flatten(1)       # (B, C)
        h = self.trunk(x)                    # (B, hidden)

        joints = self.joints_out(h).view(h.size(0), self.num_joints, 3)
        pelvis_depth = self.depth_out(h)     # (B, 1)
        pelvis_uv = self.uv_out(h)          # (B, 2)

        return {
            "joints": joints,
            "pelvis_depth": pelvis_depth,
            "pelvis_uv": pelvis_uv,
        }
