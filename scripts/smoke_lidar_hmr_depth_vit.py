#!/usr/bin/env python3
"""End-to-end smoke: LiDAR-HMR vert_feat -> depth .npy -> ViT-with-depth + one train step.

Expects:
  - Conda env **``sapiens``** (project default per CLAUDE.md): PyTorch (CUDA), mmengine,
    mmcv, editable ``pip install -e pretrain/``, plus LiDAR-HMR deps when not using
    ``--skip-lidar-hmr`` (see ``docs/update_log/2026-04-09.md`` and ``scripts/setup_lidar_hmr.sh``).
  - For the LiDAR-HMR forward: ``pointops`` on PYTHONPATH (PointTransformerV2
    ``libs/pointops`` + ``ln -s functions pointops`` if needed).
  - Run from anywhere; uses paths under the repo ``tmp_lidar_hmr_smoke/`` by default.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.registry import init_default_scope

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skip-lidar-hmr',
        action='store_true',
        help='Skip LiDAR-HMR forward if vert_feat npy already exists.',
    )
    parser.add_argument(
        '--lidar-hmr-root',
        type=Path,
        default=REPO / 'third_party' / 'LiDAR-HMR',
        help='LiDAR-HMR tree (cwd for imports).',
    )
    parser.add_argument(
        '--pointops-root',
        type=Path,
        default=REPO / 'third_party' / 'PointTransformerV2' / 'libs' / 'pointops',
        help='Directory containing the ``pointops`` package (see README in session log).',
    )
    parser.add_argument(
        '--vert-feat-dir',
        type=Path,
        default=REPO / 'tmp_lidar_hmr_smoke' / 'vert_feat',
    )
    parser.add_argument(
        '--depth-embed-dir',
        type=Path,
        default=REPO / 'tmp_lidar_hmr_smoke' / 'depth_embed',
    )
    args = parser.parse_args()

    os.environ.setdefault(
        'PYTHONPATH',
        f'{args.pointops_root}{os.pathsep}{args.lidar_hmr_root}',
    )
    if args.pointops_root not in sys.path:
        sys.path.insert(0, str(args.pointops_root))
    if args.lidar_hmr_root not in sys.path:
        sys.path.insert(0, str(args.lidar_hmr_root))

    # Avoid stem ending in ``_rgb`` — :meth:`VisionTransformerWithDepth.load_depth_embedding`
    # strips that suffix (RGB/depth naming convention) and would look up the wrong ``.npy``.
    vert_path = args.vert_feat_dir / 'lidar_smoke.npy'
    args.vert_feat_dir.mkdir(parents=True, exist_ok=True)
    args.depth_embed_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_lidar_hmr:
        print('Running LiDAR-HMR pose_meshgraphormer (random 1024 points)...')
        cwd = os.getcwd()
        try:
            os.chdir(args.lidar_hmr_root)
            from models.pmg_config import config, update_config
            from models.pose_mesh_net import pose_meshgraphormer

            update_config('configs/mesh/sloper4d.yaml')
            torch.manual_seed(0)
            m = pose_meshgraphormer(pmg_cfg=config, device='cuda').cuda()
            m.eval()
            pcd = torch.randn(1, 1024, 3, device='cuda', dtype=torch.float32) * 0.5
            with torch.no_grad():
                out = m(pcd)
            vf = out['vert_feat'].detach().cpu().numpy()
            print('vert_feat', vf.shape)
            np.save(vert_path, vf)
        finally:
            os.chdir(cwd)
    else:
        if not vert_path.is_file():
            raise FileNotFoundError(vert_path)

    print('Converting vert_feat -> depth tokens...')
    from subprocess import check_call

    check_call(
        [
            sys.executable,
            str(REPO / 'scripts' / 'convert_lidar_hmr_vert_feat.py'),
            '--input-dir',
            str(args.vert_feat_dir),
            '--output-dir',
            str(args.depth_embed_dir),
            '--num-tokens',
            '26',
            '--squeeze-batch',
        ]
    )

    init_default_scope('mmpretrain')
    from mmpretrain.registry import MODELS
    from mmpretrain.structures import DataSample

    backbone_cfg = dict(
        type='VisionTransformerWithDepth',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        out_indices=-1,
        out_type='cls_token',
        depth_embed_dim=48,
        depth_embed_path=str(args.depth_embed_dir),
        default_num_depth_tokens=26,
    )
    model_cfg = dict(
        type='ImageClassifier',
        backbone=backbone_cfg,
        neck=None,
        head=dict(
            type='VisionTransformerClsHead',
            num_classes=1000,
            in_channels=768,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
        data_preprocessor=dict(type='ClsDataPreprocessor', num_classes=1000),
    )
    model = MODELS.build(model_cfg)
    model.cuda()
    model.train()

    # Filename stem must match depth token ``*.npy`` in ``depth_embed_dir`` (see ``load_depth_embedding``).
    x = torch.randn(1, 3, 224, 224)
    samples = [
        DataSample(metainfo={'img_path': 'lidar_smoke.png'}).set_gt_label(
            torch.tensor(0, dtype=torch.long)),
    ]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    batch = model.data_preprocessor(
        {'inputs': x, 'data_samples': samples}, training=True)
    losses = model.loss(batch['inputs'], batch['data_samples'])
    losses['loss'].backward()
    opt.step()
    print('train step ok, loss=', float(losses['loss'].detach()))

    model.eval()
    with torch.no_grad():
        batch = model.data_preprocessor(
            {'inputs': x, 'data_samples': samples}, training=False)
        pred = model.predict(batch['inputs'], batch['data_samples'])
    print('predict ok, batch_size=', len(pred))


if __name__ == '__main__':
    main()
