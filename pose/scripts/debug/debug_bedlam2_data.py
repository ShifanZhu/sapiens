#!/usr/bin/env python3
"""Debug script: load BEDLAM2 data through the full transform pipeline and
inspect / save the results.

Usage (from pose/):
    conda run -n sapiens python tools/debug_bedlam2_data.py \
        --data-root /path/to/BEDLAM2 \
        --splits-dir data/bedlam2_splits \
        --out-dir /tmp/bedlam2_debug \
        --n-samples 4

    # Or point at a single NPZ directly (no splits file needed):
    conda run -n sapiens python tools/debug_bedlam2_data.py \
        --data-root /path/to/BEDLAM2 \
        --seq-npz folder/seq.npz \
        --out-dir /tmp/bedlam2_debug

    # Using a splits file (normal training setup):
    conda run -n sapiens python tools/debug_bedlam2_data.py \
        --data-root ~/data/BEDLAM2 \
        --splits-dir data/bedlam2_splits \
        --n-samples 4

    # Point directly at a single NPZ (no splits file needed):
    conda run -n sapiens python tools/debug_bedlam2_data.py \
        --data-root ~/data/BEDLAM2 \
        --seq-npz folder/seq.npz \
        --n-samples 4

    # Test the training pipeline (with bbox jitter):
    conda run -n sapiens python tools/debug_bedlam2_data.py \
        --data-root ~/data/BEDLAM2 \
        --seq-npz folder/seq.npz \
        --pipeline train
Output layout:
    <out-dir>/
        sample_0000/
            rgb.jpg          -- cropped RGB (640×384)
            depth.png        -- depth map normalised to 0-255
            depth_raw.npy    -- raw float32 depth (metres) before normalisation
            overlay.jpg      -- RGB with body skeleton joints overlaid
            joints_root_rel.npy   -- (70, 3) root-relative joints in metres
            summary.txt      -- all scalar labels and tensor shapes
        ...
        batch_inputs.npy     -- stacked model inputs (N, 4, 640, 384)
"""

import argparse
import os
import sys
import textwrap

import cv2
import numpy as np
import torch

# ── Make sure mmpose is importable from pose/ ─────────────────────────────────
POSE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if POSE_ROOT not in sys.path:
    sys.path.insert(0, POSE_ROOT)

# ── Register custom BEDLAM2 modules ───────────────────────────────────────────
import mmpose.datasets.datasets.body3d.bedlam2_dataset        # noqa: F401
import mmpose.datasets.transforms.bedlam2_transforms           # noqa: F401

from mmpose.datasets.datasets.body3d.bedlam2_dataset import Bedlam2Dataset
from mmpose.datasets.transforms.bedlam2_transforms import (
    LoadBedlamLabels, NoisyBBoxTransform, CropPersonRGBD,
    SubtractRootJoint, PackBedlamInputs,
)

# Body skeleton links (joint index pairs) for overlay drawing
_SKELETON_LINKS = [
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # spine + head
    (1, 4), (4, 7), (7, 10),   # left leg
    (2, 5), (5, 8), (8, 11),   # right leg
    (9, 13), (13, 16), (16, 18), (18, 20),  # left arm
    (9, 14), (14, 17), (17, 19), (19, 21),  # right arm
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _denorm_rgb(tensor_chw: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation; return (H, W, 3) uint8 BGR for cv2."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = tensor_chw.numpy().transpose(1, 2, 0)   # (H, W, 3) float
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _draw_skeleton(bgr: np.ndarray, joints_cam: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """Project root-relative 3D joints through K and draw on image.

    joints_cam: (70, 3) root-relative metres in BEDLAM2 camera space.
    K:          (3, 3) crop intrinsic matrix.
    Note: pelvis is at origin after SubtractRootJoint, so we draw it at
    the crop centre as indicated by K's principal point.
    """
    img = bgr.copy()
    H, W = img.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts = []
    for j in range(joints_cam.shape[0]):
        X, Y, Z = joints_cam[j]
        # BEDLAM2: u = fx*(-Y/X) + cx, v = fy*(-Z/X) + cy
        # For root-relative joints, X is relative to pelvis depth.
        # Add a nominal pelvis depth (1.0) so joints don't collapse to inf.
        X_abs = X + 2.0   # shift to a reasonable depth
        if abs(X_abs) < 0.01:
            pts.append(None)
            continue
        u = int(round(fx * (-Y / X_abs) + cx))
        v = int(round(fy * (-Z / X_abs) + cy))
        pts.append((u, v) if 0 <= u < W and 0 <= v < H else None)

    # Draw body links (body joints only)
    for i, j in _SKELETON_LINKS:
        if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
            cv2.line(img, pts[i], pts[j], (0, 200, 255), 2, cv2.LINE_AA)

    # Draw joint dots
    for pt in pts:
        if pt:
            cv2.circle(img, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)

    return img


def _depth_to_png(depth_tensor_1hw: torch.Tensor) -> np.ndarray:
    """Convert normalised depth tensor (1, H, W) to uint8 grayscale."""
    d = depth_tensor_1hw[0].numpy()   # (H, W) in [0, 1]
    return (d * 255).clip(0, 255).astype(np.uint8)


def _format_tensor(t) -> str:
    if isinstance(t, torch.Tensor):
        return f"Tensor {tuple(t.shape)} dtype={t.dtype} min={t.min():.4f} max={t.max():.4f}"
    if isinstance(t, np.ndarray):
        return f"ndarray {t.shape} dtype={t.dtype} min={t.min():.4f} max={t.max():.4f}"
    return repr(t)


def save_sample(idx: int, packed: dict, out_dir: str) -> None:
    """Save all debug outputs for one packed sample."""
    sample_dir = os.path.join(out_dir, f'sample_{idx:04d}')
    os.makedirs(sample_dir, exist_ok=True)

    inputs: torch.Tensor = packed['inputs']          # (4, H, W)
    data_sample = packed['data_samples']

    rgb_tensor = inputs[:3]                          # (3, H, W)
    depth_tensor = inputs[3:4]                       # (1, H, W)

    # ── RGB ──────────────────────────────────────────────────────────────────
    bgr = _denorm_rgb(rgb_tensor)
    cv2.imwrite(os.path.join(sample_dir, 'rgb.jpg'), bgr)

    # ── Depth ─────────────────────────────────────────────────────────────────
    depth_png = _depth_to_png(depth_tensor)
    cv2.imwrite(os.path.join(sample_dir, 'depth.png'), depth_png)

    depth_raw = depth_tensor[0].numpy() * 10.0      # back to metres
    np.save(os.path.join(sample_dir, 'depth_raw.npy'), depth_raw)

    # ── Joints ────────────────────────────────────────────────────────────────
    joints = data_sample.gt_instances.lifting_target  # (1, 70, 3) Tensor
    joints_np = joints[0].numpy()                    # (70, 3)
    np.save(os.path.join(sample_dir, 'joints_root_rel.npy'), joints_np)

    # ── Overlay ───────────────────────────────────────────────────────────────
    meta = data_sample.metainfo
    K = meta.get('K')
    if K is not None:
        K = np.array(K, dtype=np.float32) if not isinstance(K, np.ndarray) else K
        overlay = _draw_skeleton(bgr, joints_np, K)
        cv2.imwrite(os.path.join(sample_dir, 'overlay.jpg'), overlay)

    # ── Summary text ──────────────────────────────────────────────────────────
    pelvis_depth = data_sample.gt_instance_labels.pelvis_depth
    pelvis_uv    = data_sample.gt_instance_labels.pelvis_uv

    lines = [
        '=== BEDLAM2 Debug Sample ===',
        '',
        '-- Model input --',
        f'  inputs (RGBD):     {_format_tensor(inputs)}',
        f'  rgb  channels:     {_format_tensor(inputs[:3])}',
        f'  depth channel:     {_format_tensor(inputs[3])}',
        '',
        '-- Ground truth --',
        f'  lifting_target:    {_format_tensor(joints)}',
        f'    pelvis (joint 0) should be ~(0,0,0): {joints_np[0]}',
        f'    left_hip (j1):   {joints_np[1]}',
        f'    right_hip (j2):  {joints_np[2]}',
        f'  pelvis_depth (m):  {_format_tensor(pelvis_depth)} → {pelvis_depth.numpy()}',
        f'  pelvis_uv (norm):  {_format_tensor(pelvis_uv)} → {pelvis_uv.numpy()}',
        '',
        '-- Metainfo --',
    ]
    for k, v in meta.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            lines.append(f'  {k}: {_format_tensor(v)}')
        else:
            lines.append(f'  {k}: {v}')

    summary_path = os.path.join(sample_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Also print to console
    print('\n'.join(lines))
    print(f'\n  → Saved to {sample_dir}\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Debug BEDLAM2 data loading and transforms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Use a splits file (normal training setup):
              python tools/debug_bedlam2_data.py \\
                  --data-root ~/data/BEDLAM2 \\
                  --splits-dir data/bedlam2_splits \\
                  --n-samples 4

              # Point at a single sequence NPZ directly:
              python tools/debug_bedlam2_data.py \\
                  --data-root ~/data/BEDLAM2 \\
                  --seq-npz folder/seq.npz \\
                  --n-samples 4

              # Train pipeline (with bbox jitter):
              python tools/debug_bedlam2_data.py ... --pipeline train
        """),
    )
    p.add_argument('--data-root', required=True,
                   help='Path to BEDLAM2 data root (contains data/label/, data/frames/, etc.)')
    p.add_argument('--splits-dir', default='data/bedlam2_splits',
                   help='Directory with train_seqs.txt / val_seqs.txt (relative to pose/)')
    p.add_argument('--seq-npz', default=None,
                   help='Single sequence NPZ path relative to data_root/data/label/. '
                        'Overrides --splits-dir.')
    p.add_argument('--out-dir', default='/tmp/bedlam2_debug',
                   help='Output directory for debug images and arrays')
    p.add_argument('--n-samples', type=int, default=4,
                   help='Number of samples to inspect')
    p.add_argument('--pipeline', choices=['train', 'val'], default='val',
                   help='"train" adds NoisyBBoxTransform; "val" does not')
    p.add_argument('--img-h', type=int, default=640)
    p.add_argument('--img-w', type=int, default=384)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                   help='Which splits file to load when using --splits-dir')
    return p.parse_args()


def build_pipeline(pipeline_mode: str, img_h: int, img_w: int):
    transforms = [
        LoadBedlamLabels(depth_required=True,
                         filter_invalid=(pipeline_mode == 'train')),
    ]
    if pipeline_mode == 'train':
        transforms.append(NoisyBBoxTransform())
    transforms += [
        CropPersonRGBD(out_h=img_h, out_w=img_w),
        SubtractRootJoint(),
        PackBedlamInputs(),
    ]
    return transforms


def apply_pipeline(sample: dict, transforms: list) -> dict:
    results = dict(sample)
    for t in transforms:
        results = t.transform(results)
        if results is None:
            return None
    return results


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Resolve sequence paths ─────────────────────────────────────────────────
    if args.seq_npz is not None:
        seq_paths = [args.seq_npz]
        print(f'Using single sequence: {args.seq_npz}')
    else:
        splits_file = os.path.join(
            POSE_ROOT, args.splits_dir, f'{args.split}_seqs.txt')
        if not os.path.exists(splits_file):
            print(f'ERROR: splits file not found: {splits_file}')
            print('Either generate splits with tools/generate_bedlam2_splits.py '
                  'or use --seq-npz to point at a single NPZ.')
            sys.exit(1)
        with open(splits_file) as f:
            seq_paths = [ln.strip() for ln in f if ln.strip()]
        print(f'Loaded {len(seq_paths)} sequences from {splits_file}')

    # ── Build dataset (index only, no pipeline — we apply transforms manually) ─
    dataset = Bedlam2Dataset(
        data_root=args.data_root,
        seq_paths=seq_paths,
        frame_stride=1,
        pipeline=[],   # no transforms — we apply them step by step below
        max_refetch=10,
    )
    print(f'Dataset has {len(dataset)} samples (before filtering).\n')

    if len(dataset) == 0:
        print('ERROR: dataset is empty — check data_root and seq_paths.')
        sys.exit(1)

    # ── Build transform chain ──────────────────────────────────────────────────
    transforms = build_pipeline(args.pipeline, args.img_h, args.img_w)
    print(f'Pipeline ({args.pipeline}):')
    for t in transforms:
        print(f'  {t.__class__.__name__}')
    print()

    # ── Run through samples ────────────────────────────────────────────────────
    saved = 0
    all_inputs = []

    for i in range(len(dataset)):
        if saved >= args.n_samples:
            break

        raw = dataset.get_data_info(i)   # raw index entry, no transforms
        packed = apply_pipeline(raw, transforms)

        if packed is None:
            print(f'  Sample {i}: filtered out (OOB/tiny bbox/far person) — skipping')
            continue

        print(f'--- Sample {i} (saved as sample_{saved:04d}) ---')
        save_sample(saved, packed, args.out_dir)
        all_inputs.append(packed['inputs'].numpy())
        saved += 1

    # ── Save stacked batch tensor ──────────────────────────────────────────────
    if all_inputs:
        batch = np.stack(all_inputs, axis=0)   # (N, 4, H, W)
        batch_path = os.path.join(args.out_dir, 'batch_inputs.npy')
        np.save(batch_path, batch)
        print(f'\nBatch tensor saved: {batch.shape} dtype={batch.dtype} → {batch_path}')

    print(f'\nDone. {saved} samples saved to {args.out_dir}')

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print('\n=== Sanity checks ===')
    if all_inputs:
        b = np.stack(all_inputs)
        rgb = b[:, :3]
        depth = b[:, 3]
        print(f'RGB   mean={rgb.mean():.3f}  std={rgb.std():.3f}  '
              f'(expect ~0 mean after ImageNet norm)')
        print(f'Depth min={depth.min():.3f}  max={depth.max():.3f}  '
              f'(expect [0, 1] after clip/normalise)')


if __name__ == '__main__':
    main()
