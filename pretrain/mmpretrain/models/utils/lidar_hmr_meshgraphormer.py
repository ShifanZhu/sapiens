# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Wraps LiDAR-HMR's ``pose_meshgraphormer`` (PRN + mesh graphormer stack) to produce
# pooled vertex features for :class:`~mmpretrain.models.backbones.vision_transformer_with_depth.VisionTransformerWithDepth`.
#
# Requires a vendored tree at ``lidar_hmr_root`` (see ``scripts/setup_lidar_hmr.sh``),
# ``pointops`` on ``sys.path`` (PointTransformerV2 ``libs/pointops``), and running with
# CUDA for upstream point ops.

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS

from .lidar_hmr_tokens import LIDAR_HMR_VERT_FEAT_DIM


def _repo_root() -> Path:
    # pretrain/mmpretrain/models/utils/this_file.py -> repo root
    return Path(__file__).resolve().parents[4]


@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_under_repo(p: str | Path) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def _pool_vert_feat(vert_feat: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """(B, V, C) -> (B, num_tokens, C) via adaptive pooling over vertices."""
    x = vert_feat.transpose(1, 2).contiguous()
    x = F.adaptive_avg_pool1d(x, num_tokens)
    return x.transpose(1, 2).contiguous()


def _extract_tensor_state_dict(raw: object) -> dict:
    """Pull a flat parameter dict from upstream LiDAR-HMR checkpoint files."""
    if not isinstance(raw, dict):
        raise TypeError(f'Checkpoint root must be dict, got {type(raw)}')
    if 'net' in raw:
        inner = raw['net']
    elif 'state_dict' in raw:
        inner = raw['state_dict']
    elif 'model' in raw and isinstance(raw['model'], dict):
        inner = raw['model']
    else:
        inner = raw
    if not isinstance(inner, dict):
        raise TypeError('Checkpoint state payload is not a dict')
    out = {}
    for k, v in inner.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
    return out


def _strip_module_prefix(keys_vals: dict) -> dict:
    out = {}
    for k, v in keys_vals.items():
        if k.startswith('module.'):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def _remap_for_pose_meshgraphormer(flat: dict) -> dict:
    """Map full ``LiDAR_HMR`` / DDP keys to ``pose_meshgraphormer`` keys."""
    flat = _strip_module_prefix(flat)
    pmg_keys = [k for k in flat if k.startswith('pmg.')]
    if pmg_keys:
        return {k[4:]: flat[k] for k in pmg_keys}
    skip = ('smpl_', 'beta_regressor')
    return {k: v for k, v in flat.items() if not k.startswith(skip)}


def load_pose_meshgraphormer_weights(
    graphormer: nn.Module,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = 'cpu',
    verbose: bool = True,
) -> None:
    """Load upstream LiDAR-HMR weights into a ``pose_meshgraphormer`` module.

    Accepts checkpoints from the official repo: training saves ``{'net': ...}``;
    full ``LiDAR_HMR`` checkpoints prefix the mesh/graph stack with ``pmg.``.
    PRN-only files that target ``pct_pose`` load partially with ``strict=False``.
    """
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'LiDAR-HMR checkpoint not found: {ckpt_path}')

    try:
        raw = torch.load(
            str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        raw = torch.load(str(ckpt_path), map_location=map_location)
    src = _remap_for_pose_meshgraphormer(_extract_tensor_state_dict(raw))
    if not src:
        raise RuntimeError(
            f'No tensor entries found in LiDAR-HMR checkpoint: {ckpt_path}')

    model_sd = graphormer.state_dict()
    load_sd: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []

    for k, v_model in model_sd.items():
        if k not in src:
            continue
        v_ckpt = src[k]
        if v_ckpt.shape != v_model.shape:
            skipped_shape.append(
                f'{k}: model {tuple(v_model.shape)} vs ckpt {tuple(v_ckpt.shape)}')
            continue
        load_sd[k] = v_ckpt

    missing = [k for k in model_sd.keys() if k not in load_sd]
    unused = [k for k in src.keys() if k not in model_sd]

    graphormer.load_state_dict(load_sd, strict=False)

    if verbose:
        n = len(load_sd)
        print(f'[lidar_hmr_weights] Loaded {n} / {len(model_sd)} tensors from '
              f'{ckpt_path}')
        if missing:
            print(f'[lidar_hmr_weights] Missing ({len(missing)}): '
                  f'{missing[:8]}{"..." if len(missing) > 8 else ""}')
        if unused:
            print(f'[lidar_hmr_weights] Unused ckpt keys ({len(unused)}): '
                  f'{unused[:8]}{"..." if len(unused) > 8 else ""}')
        if skipped_shape:
            print(f'[lidar_hmr_weights] Shape mismatch ({len(skipped_shape)}):')
            for s in skipped_shape[:10]:
                print(f'  {s}')
            if len(skipped_shape) > 10:
                print('  ...')


@MODELS.register_module()
class LiDARHMRPoseMeshGraphormerEncoder(BaseModule):
    """Real LiDAR-HMR *pose_meshgraphormer* forward → pooled geometry tokens.

    Runs upstream ``models.pose_mesh_net.pose_meshgraphormer`` inside
    ``lidar_hmr_root`` (relative paths like ``models/graphormer`` require that cwd).

    Args:
        lidar_hmr_root: Path to LiDAR-HMR clone (e.g. ``third_party/LiDAR-HMR``).
        mesh_config: YAML under LiDAR-HMR root, passed to ``update_config`` (e.g.
            ``configs/mesh/sloper4d.yaml``).
        pointops_root: Directory containing importable ``pointops`` (PTv2
            ``libs/pointops``). If None, only existing ``sys.path`` is used.
        num_depth_tokens: Vertex pooled sequence length (e.g. 26).
        num_points: Point count resampled with 1D linear interpolate on the
            point axis (LiDAR-HMR graph uses 1024 with default configs).
        vert_feat_dim: Expected channel width from graphormer (48).
        freeze: If True, graphormer params do not train and module stays in eval.
        graphormer_device: Device string for upstream sparse buffers at init
            (e.g. ``cuda:0``). If None, uses current CUDA device or ``cpu``.
        lidar_hmr_pretrained: Path to upstream ``.pth`` (official release or
            training checkpoint). Uses ``{'net': ...}`` and strips a ``pmg.``
            prefix when loading a full ``LiDAR_HMR`` checkpoint. Optional.
        lidar_hmr_pretrained_verbose: Print load stats when pretrained is set.
        init_cfg: MMEngine init config for this wrapper.
    """

    def __init__(
        self,
        lidar_hmr_root: str = 'third_party/LiDAR-HMR',
        mesh_config: str = 'configs/mesh/sloper4d.yaml',
        pointops_root: Optional[str] = None,
        num_depth_tokens: int = 26,
        num_points: int = 1024,
        vert_feat_dim: int = LIDAR_HMR_VERT_FEAT_DIM,
        freeze: bool = False,
        graphormer_device: Optional[str] = None,
        lidar_hmr_pretrained: Optional[str] = None,
        lidar_hmr_pretrained_verbose: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.lidar_hmr_root = _resolve_under_repo(lidar_hmr_root)
        self.mesh_config = mesh_config
        self.num_depth_tokens = num_depth_tokens
        self.num_points = num_points
        self.vert_feat_dim = vert_feat_dim
        self.freeze = freeze

        if graphormer_device is None:
            if torch.cuda.is_available():
                graphormer_device = f'cuda:{torch.cuda.current_device()}'
            else:
                graphormer_device = 'cpu'
        self._graphormer_device = graphormer_device

        if pointops_root is not None:
            po = _resolve_under_repo(pointops_root)
            sys.path.insert(0, str(po))
        sys.path.insert(0, str(self.lidar_hmr_root))

        self.graphormer = self._build_graphormer()

        if lidar_hmr_pretrained:
            ckpt = _resolve_under_repo(lidar_hmr_pretrained)
            load_pose_meshgraphormer_weights(
                self.graphormer,
                ckpt,
                map_location='cpu',
                verbose=lidar_hmr_pretrained_verbose,
            )

        if self.freeze:
            self.graphormer.eval()
            for p in self.graphormer.parameters():
                p.requires_grad = False

    def _build_graphormer(self) -> nn.Module:
        with _chdir(self.lidar_hmr_root):
            from models.pmg_config import config, update_config
            from models.pose_mesh_net import pose_meshgraphormer

            update_config(self.mesh_config)
            return pose_meshgraphormer(
                pmg_cfg=config, device=self._graphormer_device)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.graphormer.eval()
        return self

    def _resample_points(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) → (B, num_points, 3)."""
        b, n, c = point_clouds.shape
        if c != 3:
            raise ValueError(f'Expected point_clouds (B, N, 3), got C={c}')
        if n == self.num_points:
            return point_clouds
        x = point_clouds.transpose(1, 2).contiguous()
        x = F.interpolate(
            x, size=self.num_points, mode='linear', align_corners=False)
        return x.transpose(1, 2).contiguous()

    def forward(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """Run LiDAR-HMR graphormer and pool ``vert_feat``.

        Args:
            point_clouds: ``(B, N, 3)`` in the same frame LiDAR-HMR expects
                (e.g. ``human_points_local``).

        Returns:
            Tensor ``(B, num_depth_tokens, vert_feat_dim)`` for ViT fusion.
        """
        if self.freeze:
            self.graphormer.eval()

        pc = self._resample_points(point_clouds)
        dev = pc.device
        # Upstream buffers may have been created on another CUDA device at init.
        g = self.graphormer.to(dev)
        out = g(pc)
        vf = out['vert_feat']
        if vf.shape[-1] != self.vert_feat_dim:
            raise ValueError(
                f'vert_feat last dim {vf.shape[-1]} != vert_feat_dim {self.vert_feat_dim}')
        return _pool_vert_feat(vf, self.num_depth_tokens)
