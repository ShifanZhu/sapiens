# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# BEDLAM2 RGBD 3D pose estimation with Sapiens 0.3B backbone.
# Predicts 70 SMPL-X joints (root-relative) + pelvis depth + pelvis UV.
#
# Optional LiDAR-HMR fusion: set ``use_lidar_hmr_fusion = True`` for
# depth → point cloud → LiDAR-HMR tokens fused in the ViT with RGB patches
# (same 3D pose head). Default ``False`` = legacy 4-channel RGB+D patch embed.
# For LiDAR-HMR, set ``PYTHONPATH`` to ``third_party/PointTransformerV2/libs/pointops``
# and ``third_party/LiDAR-HMR`` (see CLAUDE.md). Pretrained mesh weights: unzip
# ``lidarh26m.zip`` at repo root → ``lidarh26m/lidar_hmr_mesh.pth`` (optional
# ``prn_pct.pth`` is for upstream two-stage training only; the mesh checkpoint
# includes full ``pmg.*`` weights).
#
# Usage:
#   # Step 0: Generate split files (one-time, from the repo root)
#   conda run -n sapiens python pose/tools/generate_bedlam2_splits.py \
#       --data-root /media/s/SF_backup/bedlam2/ \
#       --output-dir pose/data/bedlam2_splits/
#
#   # Step 1: Train
#   conda run -n sapiens python pose/tools/train.py \
#       pose/configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384.py \
#       --work-dir /tmp/bedlam2_exp

_base_ = ['../../_base_/default_runtime.py']

# True: depth → point cloud → LiDAR-HMR tokens fused in ViT (needs pointops + LiDAR-HMR).
# Enable without editing this file: ``export USE_LIDAR_HMR_FUSION=1`` before ``train.py``.
_os = __import__('os')
use_lidar_hmr_fusion = _os.environ.get(
    'USE_LIDAR_HMR_FUSION', '').strip().lower() in ('1', 'true', 'yes', 'on')

# Force-import BEDLAM2 modules so they register with MMEngine before runner build
_custom_import_list = [
    'mmpose.models.pose_estimators.rgbd_pose3d',
    'mmpose.models.backbones.sapiens_rgbd',
    'mmpose.models.heads.regression_heads.pose3d_regression_head',
    'mmpose.models.data_preprocessors.rgbd_data_preprocessor',
    'mmpose.datasets.datasets.body3d.bedlam2_dataset',
    'mmpose.datasets.transforms.bedlam2_transforms',
    'mmpose.evaluation.metrics.bedlam_metric',
    'mmpose.engine.hooks.pose3d_visualization_hook',
    'mmpose.engine.hooks.train_mpjpe_hook',
    'mmpose.engine.optim_wrappers.fixed_amp_optim_wrapper',
]
if use_lidar_hmr_fusion:
    _custom_import_list.extend([
        'mmpose.models.backbones.sapiens_rgb_lidar_fuse',
        'mmpretrain.models.backbones',
    ])
custom_imports = dict(
    imports=_custom_import_list,
    allow_failed_imports=False,
)

# ── Architecture ──────────────────────────────────────────────────────────────
model_name = 'sapiens_0.3b'
embed_dim = 1024
num_joints = 70
img_h = 640
img_w = 384

# Path to the Sapiens RGB pretrained checkpoint.
# Set via --cfg-options or update here:
pretrained_checkpoint = '../pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth'

# LiDAR-HMR geometry encoder: default is ``lidarh26m/lidar_hmr_mesh.pth`` after
# unzipping ``lidarh26m.zip`` at repo root. Used only when ``use_lidar_hmr_fusion``
# is True. Env: ``LIDAR_HMR_PRETRAINED`` — set to another ``.pth``, or ``none`` /
# ``false`` / ``0`` to skip loading (random init).
_lidar_env = __import__('os').environ.get('LIDAR_HMR_PRETRAINED', '').strip()
if _lidar_env.lower() in ('none', 'false', '0'):
    lidar_hmr_pretrained = None
elif _lidar_env:
    lidar_hmr_pretrained = _lidar_env
else:
    lidar_hmr_pretrained = 'lidarh26m/lidar_hmr_mesh.pth'

# ── Data ──────────────────────────────────────────────────────────────────────
# ``data_root`` = BEDLAM2 *release root* — the directory that **contains** ``data/``,
# **not** ``data`` itself. Example: if labels live at
#   ``/media/s/Crucial X10/BEDLAM2/data/label/...``
# then set ``BEDLAM2_DATA_ROOT=/media/s/Crucial X10/BEDLAM2`` (parent of ``data``).
# Wrong: ``.../BEDLAM2/data`` → code would look for ``.../BEDLAM2/data/data/label``.
data_root = __import__('os').path.expanduser(
    __import__('os').environ.get('BEDLAM2_DATA_ROOT', '/media/s/Crucial X10/BEDLAM2'))
splits_dir = 'data/bedlam2_splits'

# ── Training schedule ─────────────────────────────────────────────────────────
num_epochs = 50
warmup_epochs = 3

train_cfg = dict(max_epochs=num_epochs, val_interval=1)

# LiDAR-HMR graphormer has parameters (SMPL regressor, mesh layers) that don't
# directly produce gradients on every forward pass — need DDP to allow this.
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)

# ── Optimizer ─────────────────────────────────────────────────────────────────
# Two learning-rate groups: backbone at 1e-5 (lr_mult=0.1), head at 1e-4 (base lr).
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999),
                   weight_decay=0.03),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),   # backbone LR = 1e-5
        }),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# ── LR Schedule ──────────────────────────────────────────────────────────────
param_scheduler = [
    dict(type='LinearLR', begin=0, end=warmup_epochs, start_factor=0.333,
         by_epoch=True),
    dict(type='CosineAnnealingLR', begin=warmup_epochs, end=num_epochs,
         eta_min=0, by_epoch=True),
]

# Use base Visualizer (no PoseLocalVisualizer) — avoids xtcocotools import chain
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])

# ── Hooks ────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='mpjpe/body/val',
        rule='less',
        interval=5,
        max_keep_ckpts=-1,
    ),
    logger=dict(type='LoggerHook', interval=50),
)

custom_hooks = [
    dict(type='Pose3dVisualizationHook',
         enable=True,
         bedlam2_video=True,
         vis_interval=1),
    dict(type='TrainMPJPEAveragingHook'),
    dict(type='EarlyStoppingHook',
         monitor='mpjpe/body/val',
         patience=5,
         rule='less'),
]

# ── Model ────────────────────────────────────────────────────────────────────
if use_lidar_hmr_fusion:
    _backbone = dict(
        type='SapiensBackboneRGBLidarFuse',
        arch=model_name,
        img_size=(img_h, img_w),
        drop_path_rate=0.1,
        pretrained=pretrained_checkpoint,
        use_lidar_hmr_depth=True,
        lidar_hmr_encoder=dict(
            type='LiDARHMRPoseMeshGraphormerEncoder',
            lidar_hmr_root='third_party/LiDAR-HMR',
            # Must match the checkpoint (``lidar_hmr_mesh.pth`` from LiDARH26M release
            # differs from sloper4d only in ``PCD_LAST`` — use lidarh26m for those weights).
            mesh_config='configs/mesh/lidarh26m.yaml',
            pointops_root='third_party/PointTransformerV2/libs/pointops',
            num_depth_tokens=26,
            num_points=1024,
            freeze=False,
            lidar_hmr_pretrained=lidar_hmr_pretrained,
        ),
        depth_embed_dim=48,
        default_num_depth_tokens=26,
    )
else:
    _backbone = dict(
        type='SapiensBackboneRGBD',
        arch=model_name,
        img_size=(img_h, img_w),
        drop_path_rate=0.1,
        pretrained=pretrained_checkpoint,
    )

model = dict(
    type='RGBDPose3dEstimator',
    data_preprocessor=dict(type='RGBDPoseDataPreprocessor'),
    backbone=_backbone,
    head=dict(
        type='Pose3dRegressionHead',
        in_channels=embed_dim,
        num_joints=num_joints,
        hidden_dim=2048,
        dropout=0.2,
        loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                         loss_weight=1.0),
        loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                        loss_weight=1.0),
        loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                     loss_weight=1.0),
        loss_weight_depth=1.0,
        loss_weight_uv=1.0,
    ),
    test_cfg=dict(flip_test=False),
)

# ── Data Pipelines ────────────────────────────────────────────────────────────
if use_lidar_hmr_fusion:
    train_pipeline = [
        dict(type='LoadBedlamLabels', depth_required=True),
        dict(type='NoisyBBoxTransform'),
        dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
        dict(type='BedlamDepthToPointCloud', num_points=1024),
        dict(type='SubtractRootJoint'),
        dict(type='PackBedlamInputs',
             meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                        'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
    ]
    val_pipeline = [
        dict(type='LoadBedlamLabels', depth_required=True, filter_invalid=False),
        dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
        dict(type='BedlamDepthToPointCloud', num_points=1024),
        dict(type='SubtractRootJoint'),
        dict(type='PackBedlamInputs',
             meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                        'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
    ]
else:
    train_pipeline = [
        dict(type='LoadBedlamLabels', depth_required=True),
        dict(type='NoisyBBoxTransform'),
        dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
        dict(type='SubtractRootJoint'),
        dict(type='PackBedlamInputs',
             meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                        'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
    ]
    val_pipeline = [
        dict(type='LoadBedlamLabels', depth_required=True, filter_invalid=False),
        dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
        dict(type='SubtractRootJoint'),
        dict(type='PackBedlamInputs',
             meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                        'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
    ]

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,   # MUST be False: NPZ/mmap FD issues
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='Bedlam2Dataset',
        data_root=data_root,
        seq_paths_file=splits_dir + '/train_seqs.txt',
        frame_stride=1,  # frames on disk are already at 6fps (extract_frames.py downsampled)
        pipeline=train_pipeline,
        max_refetch=10,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Bedlam2Dataset',
        data_root=data_root,
        seq_paths_file=splits_dir + '/val_seqs.txt',
        frame_stride=1,  # frames on disk are already at 6fps
        pipeline=val_pipeline,
        test_mode=True,
        max_refetch=10,
    ),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Bedlam2Dataset',
        data_root=data_root,
        seq_paths_file=splits_dir + '/test_seqs.txt',
        frame_stride=1,  # frames on disk are already at 6fps
        pipeline=val_pipeline,
        test_mode=True,
        max_refetch=10,
    ),
)

# ── Evaluators ────────────────────────────────────────────────────────────────
val_evaluator = dict(type='BedlamMPJPEMetric')
test_evaluator = dict(type='BedlamMPJPEMetric')

# ── Reproducibility ───────────────────────────────────────────────────────────
randomness = dict(seed=0)
