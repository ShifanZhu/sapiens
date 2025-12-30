# Copyright (c) Meta Platforms, Inc. and affiliates.
# Navigation Command Regression Config

_base_ = ['../_base_/default_runtime.py']

##-----------------------------------------------------------------
# Model Configuration
model_name = 'sapiens_0.3b'
embed_dim = 1024
num_layers = 24

# Pretrained checkpoint from pose estimation
pretrained_checkpoint = '../pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth'

##-----------------------------------------------------------------
# Training Configuration
evaluate_every_n_epochs = 1
vis_every_iters = 100
image_size = [768, 1024]
patch_size = 16
num_epochs = 210

# Runtime
train_cfg = dict(max_epochs=num_epochs, val_interval=evaluate_every_n_epochs)

##-----------------------------------------------------------------
# Optimizer Configuration
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,  # Lower LR for regression task
        betas=(0.9, 0.999),
        weight_decay=0.1
    ),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.85,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False
    ),  # Warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=num_epochs,
        milestones=[70, 90],
        gamma=0.1,
        by_epoch=True
    )
]

# Auto-scale LR based on batch size
auto_scale_lr = dict(base_batch_size=128)

##-----------------------------------------------------------------
# Hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='loss_nav',  # Save best model by navigation loss
        rule='less',  # Lower is better
        max_keep_ckpts=3
    ),
    logger=dict(type='LoggerHook', interval=10),
)

##-----------------------------------------------------------------
# Model Definition
model = dict(
    type='TopdownPoseEstimator',  # Reuse pose estimator framework
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),

    # Backbone: Sapiens ViT
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch=model_name,
        img_size=(image_size[1], image_size[0]),
        patch_size=patch_size,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_checkpoint
        ),
    ),  # Output: (B, 1024, 48, 64) for 768×1024 input

    # Head: NavCmdHead
    head=dict(
        type='NavCmdHead',
        in_channels=embed_dim,  # 1024 from backbone
        out_channels=3,  # vx, vy, v_yaw

        # No deconv layers (not needed for regression)
        deconv_out_channels=None,
        deconv_kernel_sizes=None,

        # Convolutional refinement
        conv_out_channels=(256,),  # One 3×3 conv layer
        conv_kernel_sizes=(3,),

        # MLP architecture
        mlp_hidden_dims=(256, 128),  # 256 → 128 → 3

        # FiLM conditioning (optional - for person-following)
        ref_in_channels=None,  # Set to 512 if using reference embeddings
        film_hidden_dim=256,

        # Reference caching (optional - for video sequences)
        cache_ref_emb=False,  # Enable for video data
        ref_from_feats=False,  # Extract ref from first frame

        use_silu=True,

        # Loss function
        loss=dict(
            type='SmoothL1Loss',
            beta=1.0,
            reduction='none',
            loss_weight=1.0
        ),

        # Optional per-dimension loss weighting
        # cmd_weights=[1.0, 1.0, 2.0],

        # Optional normalization
        # cmd_mean=[0.0, 0.0, 0.0],
        # cmd_std=[1.0, 1.0, 1.0],
    ),

    # Test configuration
    test_cfg=dict(
        unnormalize=False, 
    )
)

##-----------------------------------------------------------------
# Data Pipeline

# Training pipeline
train_pipeline = [
    dict(type='LoadImage'),  # Load RGB image
    dict(type='TopdownAffine', input_size=(image_size[0], image_size[1])),  # Resize
    dict(type='PhotometricDistortion'),  # Color jittering
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(type='GaussNoise', p=0.1),
        ]
    ),
    dict(type='PackPoseInputs')  # Pack data into required format
]

# Validation pipeline (no augmentation)
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='TopdownAffine', input_size=(image_size[0], image_size[1])),
    dict(type='PackPoseInputs')
]

##-----------------------------------------------------------------
# Dataset Configuration

# Training dataset
dataset_train = dict(
    type='NavCmdDataset',
    ann_file='path/to/your/annotations/train.json',  # ← CHANGE THIS
    data_root='path/to/your/dataset',  # ← CHANGE THIS
    data_prefix=dict(img='images/'),
    pipeline=train_pipeline,
)

# Validation dataset
dataset_val = dict(
    type='NavCmdDataset',
    ann_file='path/to/your/annotations/val.json',  # ← CHANGE THIS
    data_root='path/to/your/dataset',  # ← CHANGE THIS
    data_prefix=dict(img='images/'),
    pipeline=val_pipeline,
    test_mode=True,
)

##-----------------------------------------------------------------
# Data Loaders

train_dataloader = dict(
    batch_size=32,  # Adjust based on GPU memory
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset_train,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dataset_val,
)

test_dataloader = val_dataloader

##-----------------------------------------------------------------
# Evaluator

# Custom evaluator for navigation commands
val_evaluator = dict(
    type='NavCmdMetric',
)

test_evaluator = val_evaluator
