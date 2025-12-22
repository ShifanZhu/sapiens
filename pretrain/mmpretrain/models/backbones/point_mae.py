# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
PointMAE-style backbone (encoder-only) implemented in the SAME transformer style
as VisionTransformer in Sapiens/MMPose (TransformerEncoderLayer + ModuleList +
out_indices + out_type + frozen_stages).

Key differences vs ViT:
- Input is point cloud (B, N, 3) (optionally with extra per-point features).
- PatchEmbed is replaced by a point "Group/Patch" embedder:
  FPS centers -> kNN grouping -> local MLP -> pooled patch token.
- Positional embedding is computed from 3D patch centers (pos MLP), not a 2D grid
  learnable pos_embed that needs resizing/interpolation.

Notes:
- FPS + kNN here are implemented in pure PyTorch for clarity. For real training,
  replace them with fast CUDA ops (torch_cluster, knn_cuda, pytorch3d, etc.).
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

# In your snippet, VisionTransformer uses mmpretrain.registry.MODELS.
# If you are inside MMPose/Sapiens and want consistency, keep MODELS from mmpretrain.
from mmpretrain.registry import MODELS

from ..utils import MultiheadAttention, SwiGLUFFNFused, build_norm_layer
from .base_backbone import BaseBackbone


# --------------------------
# Pure torch FPS / kNN utils
# --------------------------
@torch.no_grad()
def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz: (B, N, 3) -> idx: (B, npoint)"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)  # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)       # (B,N)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, M) or (B, M, K)
    return: (B, M, C) or (B, M, K, C)
    """
    B = points.shape[0]
    if idx.ndim == 2:
        batch_idx = torch.arange(B, device=points.device).view(B, 1)
        return points[batch_idx, idx, :]
    if idx.ndim == 3:
        batch_idx = torch.arange(B, device=points.device).view(B, 1, 1)
        return points[batch_idx, idx, :]
    raise ValueError(f"idx must be 2D or 3D, got {idx.ndim}D")


def knn_group(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz:     (B, N, 3)
    centers: (B, M, 3)
    return:  idx (B, M, k)
    """
    # (B, M, N)
    dist = torch.cdist(centers, xyz)
    idx = torch.topk(dist, k=k, dim=-1, largest=False)[1]
    return idx


# --------------------------
# ViT-style encoder layer (same as your snippet)
# --------------------------
class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='origin',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.with_cp = with_cp

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value)

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value)
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError(f"Unknown ffn_type: {ffn_type}")

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super().init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner(x):
            x = x + self.attn(self.ln1(x))
            x = self.ffn(self.ln2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner, x)
        return _inner(x)


# --------------------------
# Point Patch Embed (PointMAE-style grouping -> patch tokens)
# --------------------------
class PointPatchEmbed(BaseModule):
    """
    Create patch tokens from point clouds by:
    - FPS selecting M patch centers
    - kNN grouping K neighbors per center
    - local MLP on relative xyz (+ optional point feats)
    - pooling over K to get a patch token

    Returns:
        tokens:     (B, M, embed_dims)
        centers_xyz:(B, M, 3)
    """

    def __init__(self,
                 num_patches: int = 256,
                 knn_k: int = 32,
                 in_channels: int = 3,           # per-point feature dim if you pass feat
                 embed_dims: int = 384,
                 use_xyz_only: bool = True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_patches = int(num_patches)
        self.knn_k = int(knn_k)
        self.in_channels = int(in_channels)
        self.embed_dims = int(embed_dims)
        self.use_xyz_only = bool(use_xyz_only)

        mlp_in = 3 if self.use_xyz_only else (3 + self.in_channels)

        self.local_mlp = nn.Sequential(
            nn.Linear(mlp_in, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
        )
        self.norm = build_norm_layer(norm_cfg, embed_dims)

    def forward(self, xyz: torch.Tensor, feat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz: (B, N, 3)
        feat(optional): (B, N, C_in)
        """
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be (B,N,3), got {tuple(xyz.shape)}")

        if (not self.use_xyz_only) and feat is None:
            raise ValueError("feat must be provided when use_xyz_only=False")

        B, N, _ = xyz.shape
        M = self.num_patches
        K = self.knn_k

        # centers via FPS
        fps_idx = farthest_point_sampling(xyz, M)     # (B,M)
        centers_xyz = index_points(xyz, fps_idx)      # (B,M,3)

        # group neighbors via kNN
        nn_idx = knn_group(xyz, centers_xyz, K)       # (B,M,K)
        grouped_xyz = index_points(xyz, nn_idx)       # (B,M,K,3)
        rel_xyz = grouped_xyz - centers_xyz.unsqueeze(2)  # (B,M,K,3)

        if self.use_xyz_only:
            grouped = rel_xyz
        else:
            grouped_feat = index_points(feat, nn_idx)     # (B,M,K,C_in)
            grouped = torch.cat([rel_xyz, grouped_feat], dim=-1)

        # local MLP per point in patch
        x = self.local_mlp(grouped)                   # (B,M,K,C)
        tokens = x.mean(dim=2)                        # (B,M,C)  (pool over K)
        tokens = self.norm(tokens)                    # LN
        return tokens, centers_xyz


# --------------------------
# PointMAE Backbone (encoder-only, transformer style like VisionTransformer)
# --------------------------
@MODELS.register_module()
class PointMAEBackbone(BaseBackbone):
    """
    PointMAE-style encoder backbone, transformer-style implementation aligned with VisionTransformer:

    - patch_embed: PointPatchEmbed -> tokens (B, M, C)
    - optional cls token
    - pos embed: computed from patch centers via pos_mlp (B, M, C)
    - transformer encoder layers: ModuleList of TransformerEncoderLayer
    - out_indices: collect intermediate layers
    - out_type: similar spirit to ViT (raw / cls_token / avg_featmap)
        * raw: returns (B, L, C) where L = M + num_extra_tokens
        * cls_token: returns (B, C) (requires with_cls_token=True)
        * avg_featmap: returns (B, C) (avg over patch tokens)
        (featmap is not meaningful without a 2D grid; not supported)

    Input:
        - Tensor: (B, N, 3) (xyz only)
        - dict: {'xyz': (B,N,3), 'feat': (B,N,C_in)} if you want extra features

    Output:
        tuple(outs) consistent with other backbones.
    """

    arch_zoo = {
        # Feel free to tune these to match your PointMAE config
        **dict.fromkeys(['tiny', 't'], dict(embed_dims=192, num_layers=12, num_heads=3, feedforward_channels=192 * 4)),
        **dict.fromkeys(['small', 's'], dict(embed_dims=384, num_layers=12, num_heads=6, feedforward_channels=384 * 4)),
        **dict.fromkeys(['base', 'b'], dict(embed_dims=768, num_layers=12, num_heads=12, feedforward_channels=768 * 4)),
        **dict.fromkeys(['large', 'l'], dict(embed_dims=1024, num_layers=24, num_heads=16, feedforward_channels=1024 * 4)),
    }

    OUT_TYPES = {'raw', 'cls_token', 'avg_featmap'}  # "featmap" removed for point tokens
    num_extra_tokens = 1  # cls token if enabled

    def __init__(self,
                 arch: Union[str, dict] = 'small',
                 # point patch embedding
                 num_patches: int = 256,
                 knn_k: int = 32,
                 in_channels: int = 3,            # per-point feature dim if feat is provided
                 use_xyz_only: bool = True,
                 # transformer knobs
                 out_indices: Union[int, Sequence[int]] = -1,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 qkv_bias: bool = True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'cls_token',
                 with_cls_token: bool = True,
                 frozen_stages: int = -1,
                 layer_scale_init_value: float = 0.,
                 layer_cfgs: Union[Sequence[dict], dict] = dict(),
                 pre_norm: bool = False,
                 with_cp: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg)

        # arch
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_zoo, f'arch {arch} not in arch_zoo {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential = {'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'}
            assert isinstance(arch, dict) and essential <= set(arch), f'Custom arch must have keys {essential}'
            self.arch_settings = arch

        self.embed_dims = int(self.arch_settings['embed_dims'])
        self.num_layers = int(self.arch_settings['num_layers'])

        # out_type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported out_type {out_type}, choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # patch embedding (point grouping)
        self.patch_embed = PointPatchEmbed(
            num_patches=num_patches,
            knn_k=knn_k,
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            use_xyz_only=use_xyz_only,
            norm_cfg=norm_cfg,
        )
        self.num_patches = int(num_patches)

        # cls token (optional)
        self.with_cls_token = bool(with_cls_token)
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
            self.num_extra_tokens = 1
        else:
            self.cls_token = None
            self.num_extra_tokens = 0
            if self.out_type == 'cls_token':
                raise ValueError('with_cls_token must be True when out_type="cls_token".')

        # positional embedding from 3D centers (PointMAE style)
        # (B, M, 3) -> (B, M, C)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # out_indices
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        out_indices = list(out_indices)
        for i, idx in enumerate(out_indices):
            if idx < 0:
                out_indices[i] = self.num_layers + idx
            assert 0 <= out_indices[i] < self.num_layers, f'Invalid out_indices {idx}'
        self.out_indices = out_indices

        # layers (ViT style)
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()

        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers

        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=int(self.arch_settings['num_heads']),
                feedforward_channels=int(self.arch_settings['feedforward_channels']),
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=float(dpr[i]),
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        # pre_norm & final_norm (same style as your ViT)
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = bool(final_norm)
        if self.final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.frozen_stages = int(frozen_stages)
        if self.frozen_stages > 0:
            self._freeze_stages()

        self.feat_dim = self.embed_dims

    @property
    def norm1(self):
        return getattr(self, 'ln1', None)

    @property
    def norm2(self):
        return getattr(self, 'ln2', None)

    def init_weights(self):
        super().init_weights()
        if not (isinstance(self.init_cfg, dict) and self.init_cfg.get('type', None) == 'Pretrained'):
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=0.02)
            # pos_mlp uses default init; you can also add trunc_normal_ if desired.

    def _freeze_stages(self):
        # freeze patch embed
        self.patch_embed.eval()
        for p in self.patch_embed.parameters():
            p.requires_grad = False

        # freeze pos
        self.pos_mlp.eval()
        for p in self.pos_mlp.parameters():
            p.requires_grad = False

        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False

        # freeze dropout
        self.drop_after_pos.eval()

        # freeze pre_norm
        for p in self.pre_norm.parameters():
            p.requires_grad = False

        # freeze first N layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        # freeze final norms if fully frozen
        if self.frozen_stages == len(self.layers):
            if self.final_norm and hasattr(self, 'ln1'):
                self.ln1.eval()
                for p in self.ln1.parameters():
                    p.requires_grad = False
            if self.out_type == 'avg_featmap' and hasattr(self, 'ln2'):
                self.ln2.eval()
                for p in self.ln2.parameters():
                    p.requires_grad = False

    def _parse_input(self, x: Union[torch.Tensor, dict]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(x, torch.Tensor):
            xyz = x
            feat = None
        elif isinstance(x, dict):
            xyz = x.get('xyz', None)
            feat = x.get('feat', None)
            if xyz is None:
                raise KeyError("PointMAEBackbone expects dict input with key 'xyz'")
        else:
            raise TypeError(f'Unsupported input type: {type(x)}')

        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be (B,N,3), got {tuple(xyz.shape)}")
        return xyz, feat

    def forward(self, x: Union[torch.Tensor, dict]) -> Tuple[torch.Tensor, ...]:
        """
        Returns tuple of outputs according to out_indices and out_type.
        """
        xyz, feat = self._parse_input(x)
        B = xyz.shape[0]

        # patch embedding
        tokens, centers_xyz = self.patch_embed(xyz, feat)  # (B,M,C), (B,M,3)

        # add cls token
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)   # (B,1,C)
            tokens = torch.cat([cls_token, tokens], dim=1)  # (B,1+M,C)

            # pos: cls has 0 pos by convention (you can also learn one if you want)
            pos = self.pos_mlp(centers_xyz)                # (B,M,C)
            cls_pos = torch.zeros(B, 1, self.embed_dims, device=tokens.device, dtype=tokens.dtype)
            pos = torch.cat([cls_pos, pos], dim=1)         # (B,1+M,C)
        else:
            pos = self.pos_mlp(centers_xyz)                # (B,M,C)

        tokens = tokens + pos
        tokens = self.drop_after_pos(tokens)
        tokens = self.pre_norm(tokens)

        outs = []
        for i, layer in enumerate(self.layers):
            tokens = layer(tokens)

            if i == len(self.layers) - 1 and self.final_norm:
                tokens = self.ln1(tokens)

            if i in self.out_indices:
                outs.append(self._format_output(tokens))

        return tuple(outs)

    def _format_output(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L, C) where L = M + num_extra_tokens
        """
        if self.out_type == 'raw':
            return tokens

        if self.out_type == 'cls_token':
            return tokens[:, 0]  # (B,C)

        # avg_featmap: average over patch tokens (exclude cls token if present)
        patch_tokens = tokens[:, self.num_extra_tokens:]   # (B,M,C)
        avg = patch_tokens.mean(dim=1)                     # (B,C)
        if hasattr(self, 'ln2'):
            avg = self.ln2(avg)
        return avg

    def train(self, mode=True):
        super().train(mode)
        if mode and self.frozen_stages > 0:
            self._freeze_stages()
