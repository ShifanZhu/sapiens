# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# **Not BEDLAM2 / not pose:** this is a small **pretrain ImageClassifier** smoke
# config only. For BEDLAM2 RGBD 3D pose use ``pose/scripts/finetune/bedlam2/.../node.sh``
# and ``BEDLAM2_DATA_ROOT`` (dataset root = parent of ``data/``).
#
# Runnable **pretrain** job: ViT-B/16 + **live LiDAR-HMR** (``pose_meshgraphormer``)
# on CIFAR-10. Point clouds are **random placeholders** — replace the transform
# with real ``human_points_local`` for production.
#
# Example (from ``sapiens`` repo, after ``conda activate sapiens``)::
#
#   cd pretrain
#   export PYTHONPATH="${PWD}/../third_party/PointTransformerV2/libs/pointops:${PWD}/../third_party/LiDAR-HMR:${PYTHONPATH}"
#   PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
#   CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh \\
#     configs/lidar_hmr_vit/vit_b16_lidar_hmr_cifar10.py 1 \\
#     --work-dir Outputs/train/lidar_hmr_vit/cifar10/$(date +%m-%d-%Y_%H:%M:%S) \\
#     --seed 0
#
# Requires CUDA, built ``pointops``, and a LiDAR-HMR tree under ``../third_party/LiDAR-HMR``.

_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(
    imports=['mmpretrain.datasets.transforms.lidar_hmr_points'],
    allow_failed_imports=False,
)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=224),
    dict(type='RandomHumanPointsPlaceholder', num_points=1024, scale=1.0),
    dict(
        type='PackInputs',
        algorithm_keys=('human_points_local',),
    ),
]

test_pipeline = [
    dict(type='Resize', scale=224),
    dict(type='RandomHumanPointsPlaceholder', num_points=1024, scale=1.0),
    dict(
        type='PackInputs',
        algorithm_keys=('human_points_local',),
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        split='train',
        pipeline=train_pipeline,
        download=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        split='test',
        pipeline=test_pipeline,
        download=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=1, eta_min=1e-6),
]

model = dict(
    type='ImageClassifier',
    data_preprocessor=dict(
        type='ClsDataPreprocessor',
        num_classes=10,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    backbone=dict(
        type='VisionTransformerWithDepth',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        depth_embed_dim=48,
        depth_embed_path=None,
        use_depth_projection=True,
        depth_projection_type='linear',
        default_num_depth_tokens=26,
        use_lidar_hmr_depth=True,
        lidar_hmr_encoder=dict(
            type='LiDARHMRPoseMeshGraphormerEncoder',
            lidar_hmr_root='third_party/LiDAR-HMR',
            mesh_config='configs/mesh/sloper4d.yaml',
            pointops_root='third_party/PointTransformerV2/libs/pointops',
            num_depth_tokens=26,
            num_points=1024,
            freeze=False,
        ),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=10,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)
