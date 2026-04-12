
# ViT-B/16 + live LiDAR-HMR geometry tokens fused in-transformer.
#
# To turn **off** in-graph LiDAR-HMR (RGB-only or disk tokens only):
#   use_lidar_hmr_depth=False
#   lidar_hmr_encoder=None
#   depth_embed_path='/path/to/npy_dir'   # optional, for precomputed tokens
#
# Data: use PackInputs with algorithm_keys=('human_points_local',) and per-frame
# points (N, 3) in LiDAR-HMR space. Set depth_embed_dim=48 to match vert_feat.
#
# Env: vendor LiDAR-HMR under third_party/, build pointops, PYTHONPATH as in
# docs/update_log/2026-04-09.md (or pass pointops_root below).

# model settings
model = dict(
    type='ImageClassifier',
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
            # Default in BEDLAM2 config: ``lidarh26m/lidar_hmr_mesh.pth`` (from
            # ``lidarh26m.zip`` at repo root). Or set explicitly:
            # lidar_hmr_pretrained='lidarh26m/lidar_hmr_mesh.pth',
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
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ))
