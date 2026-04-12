#!/usr/bin/env python3
"""Smoke test: LiDAR fusion backbone forward/backward (no full Runner).

Requires the same deps as upstream LiDAR-HMR (e.g. ``easydict``, ``torch_geometric``,
built ``pointops``). Run from repo root:

  PYTHONPATH=pose:pretrain:engine:third_party/LiDAR-HMR:third_party/PointTransformerV2/libs/pointops \\
  conda run -n sapiens python pose/tools/verify_lidar_fusion_smoke.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in ('pose', 'pretrain', 'engine'):
    sys.path.insert(0, os.path.join(_REPO, p))
sys.path.insert(0, os.path.join(_REPO, 'third_party/LiDAR-HMR'))
sys.path.insert(0, os.path.join(_REPO, 'third_party/PointTransformerV2/libs/pointops'))


def main() -> None:
    os.chdir(_REPO)
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    from mmpose.registry import MODELS
    from mmengine.registry import init_default_scope
    from mmpose.utils import register_all_modules

    register_all_modules(init_default_scope=False)
    init_default_scope('mmpose')

    backbone = dict(
        type='SapiensBackboneRGBLidarFuse',
        arch='sapiens_0.3b',
        img_size=(640, 384),
        drop_path_rate=0.1,
        pretrained=None,
        use_lidar_hmr_depth=True,
        lidar_hmr_encoder=dict(
            type='LiDARHMRPoseMeshGraphormerEncoder',
            lidar_hmr_root='third_party/LiDAR-HMR',
            mesh_config='configs/mesh/lidarh26m.yaml',
            pointops_root='third_party/PointTransformerV2/libs/pointops',
            num_depth_tokens=26,
            num_points=1024,
            freeze=False,
            lidar_hmr_pretrained='lidarh26m/lidar_hmr_mesh.pth',
        ),
        depth_embed_dim=48,
        default_num_depth_tokens=26,
    )
    net = MODELS.build(backbone).to(device)
    net.train()

    B, H, W = 2, 640, 384
    x = torch.randn(B, 4, H, W, device=device)
    pts = torch.randn(B, 1024, 3, device=device)

    feats = net(x, point_clouds=pts)
    print('Backbone output[0] shape:', tuple(feats[0].shape))

    loss = feats[0].float().mean()
    net.zero_grad(set_to_none=True)
    loss.backward()

    no_grad = [
        n for n, p in net.named_parameters()
        if p.requires_grad and p.grad is None]
    print('requires_grad but grad is None:', len(no_grad))
    for n in no_grad[:20]:
        print(' ', n)
    print('Backward OK')


if __name__ == '__main__':
    main()
