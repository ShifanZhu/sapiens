#!/usr/bin/env python3
"""
Create NavCmdDataset JSON annotation files from robot data.

This script processes each sequence folder in the data directory and creates
one JSON annotation file per sequence with all the data. Optionally extracts
reference embeddings from the first frame of each sequence using a pretrained
Sapiens backbone.

Structure expected:
    data/
    ├── lgrc1218/
    │   └── converted/data_XXXXXXXX_XXXXXX/
    │       ├── D455_2/rgb/
    │       ├── go2_velocity.txt
    │       └── ...
    ├── sequence2/
    │   └── converted/data_XXXXXXXX_XXXXXX/
    │       └── ...
    └── ...

Usage:
    # Create dataset without reference embeddings
    python tools/create_nav_cmd_dataset.py \
        --data-root /path/to/data \
        --output-dir annotations

    # Create dataset with reference embeddings
    python pose/tools/create_nav_cmd_dataset.py \
        --data-root ../data \
        --output-dir ../annotations \
        --checkpoint pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_backbone(checkpoint_path: str, device: str = 'cuda'):
    """Load pretrained Sapiens backbone for reference embedding extraction.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Backbone model
    """
    from mmpretrain.models import VisionTransformer
    from mmengine.runner import load_checkpoint

    # Create backbone matching config
    model = VisionTransformer(
        arch='sapiens_0.3b',
        img_size=(1024, 768),
        patch_size=16,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
    )

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded backbone from: {checkpoint_path}")
    return model


def extract_ref_embedding(image_path: str, model: torch.nn.Module, device: str = 'cuda') -> np.ndarray:
    """Extract reference embedding from an image.

    Args:
        image_path: Path to image file
        model: Backbone model
        device: Device to run on

    Returns:
        Reference embedding as numpy array (C,)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')

    # Preprocessing (matching PoseDataPreprocessor)
    transform = transforms.Compose([
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],
            std=[58.395/255, 57.12/255, 57.375/255]
        ),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        feats = model(img_tensor)

        if isinstance(feats, tuple):
            feats = feats[0]

        # Global average pooling
        ref_emb = feats.mean(dim=[2, 3])
        ref_emb = ref_emb.squeeze(0).cpu().numpy()

    return ref_emb


def load_rgb_timestamps(timestamp_file: str) -> List[int]:
    """Load RGB image timestamps.

    Args:
        timestamp_file: Path to rgb_timestamp.txt

    Returns:
        List of timestamps (integers)
    """
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                timestamp = int(line.split('_')[0])
                timestamps.append(timestamp)
    return timestamps


def load_velocity_data(velocity_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load robot velocity commands.

    Args:
        velocity_file: Path to go2_velocity.txt

    Returns:
        timestamps: (N,) array of timestamps
        velocities: (N, 3) array of [vx, vy, v_yaw]
    """
    data = []
    with open(velocity_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 4:
                    timestamp = int(parts[0])
                    vx = float(parts[1])
                    vy = float(parts[2])
                    v_yaw = float(parts[3])
                    data.append([timestamp, vx, vy, v_yaw])

    data = np.array(data)
    timestamps = data[:, 0].astype(np.int64)
    velocities = data[:, 1:4].astype(np.float32)

    return timestamps, velocities


def find_closest_velocity(rgb_timestamp: int,
                          vel_timestamps: np.ndarray,
                          velocities: np.ndarray) -> np.ndarray:
    """Find the velocity command closest to RGB timestamp.

    Args:
        rgb_timestamp: RGB image timestamp
        vel_timestamps: Array of velocity timestamps
        velocities: Array of velocity commands (N, 3)

    Returns:
        Closest velocity command [vx, vy, v_yaw]
    """
    # Find index of closest timestamp
    idx = np.argmin(np.abs(vel_timestamps - rgb_timestamp))
    return velocities[idx]


def create_dataset_json(data_dir: str,
                       output_file: str,
                       sequence_name: str,
                       camera_name: str = 'D455_2',
                       checkpoint_path: Optional[str] = None,
                       device: str = 'cuda') -> Dict:
    
    """Create NavCmdDataset JSON annotation file for a sequence.

    Args:
        data_dir: Path to data directory (e.g., data_20251218_130141)
        output_file: Output JSON file path
        sequence_name: Name of the sequence
        camera_name: Camera folder name (default: D455_2)
        checkpoint_path: Optional path to Sapiens checkpoint for reference embedding extraction
        device: Device to use for embedding extraction (default: cuda)

    Returns:
        Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)

    # File paths
    rgb_timestamp_file = data_dir / camera_name / 'rgb_timestamp.txt'
    velocity_file = data_dir / 'go2_velocity.txt'
    rgb_dir = data_dir / camera_name / 'rgb'

    # Verify files exist
    if not rgb_timestamp_file.exists():
        raise FileNotFoundError(f"RGB timestamp file not found: {rgb_timestamp_file}")
    if not velocity_file.exists():
        raise FileNotFoundError(f"Velocity file not found: {velocity_file}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    print(f"Loading RGB timestamps from: {rgb_timestamp_file}")
    rgb_timestamps = load_rgb_timestamps(str(rgb_timestamp_file))

    print(f"Loading velocity data from: {velocity_file}")
    vel_timestamps, velocities = load_velocity_data(str(velocity_file))

    print(f"Found {len(rgb_timestamps)} RGB images")
    print(f"Found {len(vel_timestamps)} velocity commands")

    num_samples = len(rgb_timestamps)
    print(f"Processing all {num_samples} samples")

    # Load backbone if checkpoint provided
    model = None
    if checkpoint_path is not None:
        print(f"\nLoading backbone for reference embedding extraction...")
        model = load_backbone(checkpoint_path, device)

    # Create annotations
    images = []
    annotations = []

    print("\nMatching images to velocity commands...")
    for idx in range(num_samples):
        rgb_timestamp = rgb_timestamps[idx]

        # Find closest velocity command
        nav_cmd = find_closest_velocity(rgb_timestamp, vel_timestamps, velocities)

        # Image info
        image_filename = f"{camera_name}/rgb/{rgb_timestamp}_rgb.png"
        is_ref_frame = (idx == 0)  # First frame is reference

        image_info = {
            "id": idx,
            "file_name": image_filename,
            "sequence_id": sequence_name,
            "frame_idx": idx,
            "timestamp": int(rgb_timestamp),
            "is_ref_frame": is_ref_frame
        }
        images.append(image_info)

        # Annotation info
        annotation_info = {
            "image_id": idx,
            "nav_cmd": [float(nav_cmd[0]), float(nav_cmd[1]), float(nav_cmd[2])]
        }

        # Extract and store reference embedding only for reference frames
        if model is not None and is_ref_frame:
            img_path = data_dir / image_filename
            if img_path.exists():
                print(f"Extracting reference embedding from: {image_filename}")
                ref_emb = extract_ref_embedding(str(img_path), model, device)
                annotation_info["ref_emb"] = ref_emb.tolist()
                print(f"  Reference embedding shape: {ref_emb.shape}")
            else:
                print(f"  ⚠ Warning: Reference frame not found: {img_path}")

        annotations.append(annotation_info)

        # Progress
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{num_samples} images...")

    # Create final dataset dict
    dataset = {
        "info": {
            "date_created": data_dir.name,
            "sequence_name": sequence_name,
            "camera": camera_name,
            "num_samples": len(images)
        },
        "images": images,
        "annotations": annotations
    }

    # Save JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Statistics
    nav_cmds = np.array([ann['nav_cmd'] for ann in annotations])
    stats = {
        'num_samples': len(images),
        'has_ref_emb': ref_emb is not None,
        'ref_emb_dim': int(ref_emb.shape[0]) if ref_emb is not None else 0,
        'vx_mean': float(nav_cmds[:, 0].mean()),
        'vx_std': float(nav_cmds[:, 0].std()),
        'vx_min': float(nav_cmds[:, 0].min()),
        'vx_max': float(nav_cmds[:, 0].max()),
        'vy_mean': float(nav_cmds[:, 1].mean()),
        'vy_std': float(nav_cmds[:, 1].std()),
        'vy_min': float(nav_cmds[:, 1].min()),
        'vy_max': float(nav_cmds[:, 1].max()),
        'vyaw_mean': float(nav_cmds[:, 2].mean()),
        'vyaw_std': float(nav_cmds[:, 2].std()),
        'vyaw_min': float(nav_cmds[:, 2].min()),
        'vyaw_max': float(nav_cmds[:, 2].max()),
    }

    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    if stats['has_ref_emb']:
        print(f"  Reference embeddings: Yes (dim={stats['ref_emb_dim']})")
    else:
        print(f"  Reference embeddings: No")
    print(f"  vx:   mean={stats['vx_mean']:.4f}, std={stats['vx_std']:.4f}, range=[{stats['vx_min']:.4f}, {stats['vx_max']:.4f}]")
    print(f"  vy:   mean={stats['vy_mean']:.4f}, std={stats['vy_std']:.4f}, range=[{stats['vy_min']:.4f}, {stats['vy_max']:.4f}]")
    print(f"  vyaw: mean={stats['vyaw_mean']:.4f}, std={stats['vyaw_std']:.4f}, range=[{stats['vyaw_min']:.4f}, {stats['vyaw_max']:.4f}]")

    return stats


def find_data_directories(data_root: Path, camera_name: str = 'D455_2') -> List[Tuple[str, Path]]:
    """Find all valid data directories in the data root.

    Args:
        data_root: Root directory containing sequence folders
        camera_name: Camera folder name to look for

    Returns:
        List of (sequence_name, data_dir_path) tuples
    """
    data_dirs = []

    # Look for sequence folders
    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        # Look for converted/data_* subdirectories
        converted_dir = seq_dir / 'converted'
        if converted_dir.exists():
            for data_dir in sorted(converted_dir.glob('data_*')):
                if data_dir.is_dir():
                    # Check if it has the required files
                    if (data_dir / camera_name / 'rgb_timestamp.txt').exists() and \
                       (data_dir / 'go2_velocity.txt').exists():
                        sequence_name = seq_dir.name
                        data_dirs.append((sequence_name, data_dir))
                        print(f"  Found sequence: {sequence_name} -> {data_dir.name}")

    return data_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Create NavCmdDataset JSON files from robot data')

    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing sequence folders (e.g., /path/to/data)')
    parser.add_argument('--output-dir', type=str, default='annotations',
                       help='Output directory for annotation files (default: annotations)')
    parser.add_argument('--camera-name', type=str, default='D455_2',
                       help='Camera folder name (default: D455_2)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Optional: Path to Sapiens checkpoint for reference embedding extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for embedding extraction (default: cuda)')
    args = parser.parse_args()

    # Create output directory
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Creating NavCmdDataset Annotations")
    print("="*80)
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Camera: {args.camera_name}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Device: {args.device}")
        print("  → Will extract reference embeddings")
    else:
        print("  → No checkpoint provided (skipping reference embeddings)")
    print("="*80)

    # Find all data directories
    print("\nScanning for sequences...")
    data_dirs = find_data_directories(data_root, args.camera_name)

    if not data_dirs:
        print("❌ No valid data directories found!")
        print("\nExpected structure:")
        print("  data_root/")
        print("  ├── sequence1/")
        print("  │   └── converted/data_XXXXXXXX_XXXXXX/")
        print("  │       ├── D455_2/rgb_timestamp.txt")
        print("  │       └── go2_velocity.txt")
        print("  └── sequence2/")
        print("      └── ...")
        return

    print(f"\nFound {len(data_dirs)} sequence(s) to process")

    # Process each sequence
    all_stats = {}
    for seq_idx, (sequence_name, data_dir) in enumerate(data_dirs, 1):
        print(f"\n{'='*80}")
        print(f"[{seq_idx}/{len(data_dirs)}] Processing: {sequence_name}")
        print(f"{'='*80}")

        # Output file for this sequence
        output_file = output_dir / f'{sequence_name}.json'

        # Create dataset
        stats = create_dataset_json(
            data_dir=str(data_dir),
            output_file=str(output_file),
            sequence_name=sequence_name,
            camera_name=args.camera_name,
            checkpoint_path=args.checkpoint,
            device=args.device
        )

        all_stats[sequence_name] = stats

    # Summary
    print("\n" + "="*80)
    print("✅ All datasets created successfully!")
    print("="*80)
    print(f"\nGenerated files:")
    total_samples = 0
    has_embeddings = False
    for sequence_name, stats in all_stats.items():
        json_file = output_dir / f'{sequence_name}.json'
        emb_info = f" with ref_emb (dim={stats['ref_emb_dim']})" if stats['has_ref_emb'] else ""
        print(f"  {json_file} ({stats['num_samples']} samples{emb_info})")
        total_samples += stats['num_samples']
        if stats['has_ref_emb']:
            has_embeddings = True

    print(f"\nTotal samples: {total_samples}")
    if has_embeddings:
        print("Reference embeddings: ✓ Included")

if __name__ == '__main__':
    main()
