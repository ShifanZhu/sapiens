# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# BEDLAM2 RGBD 3D pose estimation with Sapiens 0.3B backbone
# and TRANSFORMER DECODER HEAD (replacing GAP+MLP head).
#
# This config is identical to sapiens_0.3b-50e_bedlam2-640x384.py
# except for the head type. Used for A/B comparison.
#
# Usage:
#   conda run -n sapiens python pose/tools/train.py \
#       pose/configs/sapiens_pose/bedlam2/sapiens_0.3b-50e_bedlam2-640x384-transformer.py \
#       --work-dir /tmp/bedlam2_transformer_exp

_base_ = ['../../_base_/default_runtime.py']

# Force-import BEDLAM2 modules so they register with MMEngine before runner build
custom_imports = dict(
    imports=[
        'mmpose.models.pose_estimators.rgbd_pose3d',
        'mmpose.models.backbones.sapiens_rgbd',
        'mmpose.models.heads.regression_heads.pose3d_transformer_head',
        'mmpose.models.data_preprocessors.rgbd_data_preprocessor',
        'mmpose.datasets.datasets.body3d.bedlam2_dataset',
        'mmpose.datasets.transforms.bedlam2_transforms',
        'mmpose.evaluation.metrics.bedlam_metric',
        'mmpose.engine.hooks.pose3d_visualization_hook',
        'mmpose.engine.hooks.train_mpjpe_hook',
        'mmpose.engine.optim_wrappers.fixed_amp_optim_wrapper',
    ],
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

# ── Data ──────────────────────────────────────────────────────────────────────
data_root = __import__('os').path.expanduser(
    __import__('os').environ.get('BEDLAM2_DATA_ROOT', '~/repos_local/MMC/BEDLAM2Datatest'))
splits_dir = 'data/bedlam2_splits'

# ── Training schedule ─────────────────────────────────────────────────────────
num_epochs = 50
warmup_epochs = 3

train_cfg = dict(max_epochs=num_epochs, val_interval=1)

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
model = dict(
    type='RGBDPose3dEstimator',
    data_preprocessor=dict(type='RGBDPoseDataPreprocessor'),
    backbone=dict(
        type='SapiensBackboneRGBD',
        arch=model_name,
        img_size=(img_h, img_w),
        drop_path_rate=0.1,
        pretrained=pretrained_checkpoint,
    ),
    head=dict(
        type='Pose3dTransformerHead',
        in_channels=embed_dim,
        num_joints=num_joints,
        num_heads=8,
        dropout=0.1,
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
