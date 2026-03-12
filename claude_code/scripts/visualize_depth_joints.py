"""Visualize GT joints and depth overlay on RGB frames.

For each sampled frame, saves a side-by-side image:
    [RGB + skeleton]  |  [depth colormap + skeleton]

Usage:
    conda run -n sapiens_gpu python visualize_depth_joints.py \
        --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
        --output-dir runs/vis_depth_joints \
        --n-seqs 8 --frames-per-seq 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_splits, build_val_transform, build_dataloader
from data.constants import SMPLX_SKELETON, JOINT_NAMES


def project_joints(joints: np.ndarray, K: np.ndarray):
    """Project (J,3) camera-space XYZ to pixel (u,v). Returns u,v,valid arrays."""
    X, Y, Z = joints[:, 0], joints[:, 1], joints[:, 2]
    valid = X > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(valid, K[0, 0] * (-Y / X) + K[0, 2], -1.0)
        v = np.where(valid, K[1, 1] * (-Z / X) + K[1, 2], -1.0)
    return u, v, valid


def draw_skeleton(img_bgr: np.ndarray, u, v, valid, joint_color, bone_color):
    """Draw skeleton bones and joint dots onto img_bgr in-place."""
    H, W = img_bgr.shape[:2]
    for a, b in SMPLX_SKELETON:
        if valid[a] and valid[b]:
            pa = (int(round(u[a])), int(round(v[a])))
            pb = (int(round(u[b])), int(round(v[b])))
            if (0 <= pa[0] < W and 0 <= pa[1] < H and
                    0 <= pb[0] < W and 0 <= pb[1] < H):
                cv2.line(img_bgr, pa, pb, bone_color, 2, cv2.LINE_AA)
    for j in range(len(u)):
        if valid[j]:
            pt = (int(round(u[j])), int(round(v[j])))
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                cv2.circle(img_bgr, pt, 3, joint_color, -1, cv2.LINE_AA)


def depth_to_colormap(depth_hw: np.ndarray) -> np.ndarray:
    """Convert (H,W) float32 depth to (H,W,3) uint8 BGR colormap."""
    valid = depth_hw > 0
    vis = np.zeros_like(depth_hw)
    if valid.any():
        d_min, d_max = depth_hw[valid].min(), depth_hw[valid].max()
        if d_max > d_min:
            vis[valid] = (depth_hw[valid] - d_min) / (d_max - d_min)
    vis_u8 = (vis * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_MAGMA)


RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tensor_to_uint8(rgb_chw) -> np.ndarray:
    """Denormalize (3,H,W) float32 tensor → (H,W,3) uint8 BGR."""
    rgb = rgb_chw.numpy().transpose(1, 2, 0)
    rgb = (rgb * RGB_STD + RGB_MEAN).clip(0, 1)
    return (rgb[:, :, ::-1] * 255).astype(np.uint8)


def make_panel(rgb_bgr, depth_hw, joints, K):
    """Build side-by-side panel: [RGB+skeleton | depth+skeleton]."""
    u, v, valid = project_joints(joints, K)

    left = rgb_bgr.copy()
    draw_skeleton(left, u, v, valid,
                  joint_color=(0, 255, 0), bone_color=(0, 200, 0))

    right = depth_to_colormap(depth_hw)
    draw_skeleton(right, u, v, valid,
                  joint_color=(255, 255, 0), bone_color=(200, 200, 0))

    # Label panels
    for img, label in [(left, "RGB + GT joints"), (right, "Depth + GT joints")]:
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 1, cv2.LINE_AA)

    return np.concatenate([left, right], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",       default="/home/hang/repos_local/MMC/BEDLAM2Datatest")
    p.add_argument("--output-dir",      default="runs/vis_depth_joints")
    p.add_argument("--n-seqs",          type=int, default=8,  help="Number of sequences to visualize")
    p.add_argument("--frames-per-seq",  type=int, default=4,  help="Frames per sequence")
    p.add_argument("--img-h",           type=int, default=384)
    p.add_argument("--img-w",           type=int, default=640)
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    _, val_seqs, _ = get_splits(
        overview_path=str(Path(args.data_root) / "data" / "overview.txt"),
        val_ratio=0.1, test_ratio=0.1, seed=2026,
        single_body_only=True, skip_missing_body=True,
        depth_required=True, mp4_required=False,
    )

    val_tf = build_val_transform(args.img_h, args.img_w)
    loader = build_dataloader(
        seq_paths=val_seqs, data_root=args.data_root,
        transform=val_tf, batch_size=1,
        shuffle=False, num_workers=0,
    )
    dataset = loader.dataset

    # ── Pick frames: first N frames from each of n_seqs sequences ────────────
    import random
    random.seed(args.seed)

    # Group index by sequence
    seq_to_indices: dict[str, list[int]] = {}
    for i, (label_path, _) in enumerate(dataset.index):
        seq_to_indices.setdefault(label_path, []).append(i)

    selected_seqs = random.sample(list(seq_to_indices.keys()),
                                  min(args.n_seqs, len(seq_to_indices)))

    saved = 0
    for seq_path in selected_seqs:
        indices = seq_to_indices[seq_path]
        # Evenly spaced frames across the sequence
        chosen = np.linspace(0, len(indices) - 1, args.frames_per_seq, dtype=int)
        frame_indices = [indices[c] for c in chosen]

        seq_name = Path(seq_path).stem
        folder   = Path(seq_path).parent.name

        for fi, idx in enumerate(frame_indices):
            sample = dataset[idx]

            rgb_bgr  = tensor_to_uint8(sample["rgb"])
            depth_hw = sample["depth"].squeeze(0).numpy()  # (H, W) float32 [0,1] normalized
            joints   = sample["joints"].numpy()             # (127, 3) metres
            K        = sample["intrinsic"].numpy()          # (3, 3)

            # Denormalize depth back to metres for display
            from data.constants import DEPTH_MAX_METERS
            depth_m  = depth_hw * DEPTH_MAX_METERS

            panel = make_panel(rgb_bgr, depth_m, joints, K)

            fname = out_dir / f"{folder}__{seq_name}__f{fi:02d}.jpg"
            cv2.imwrite(str(fname), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Saved {fname.name}")
            saved += 1

    print(f"\nDone. {saved} images saved to {out_dir}/")


if __name__ == "__main__":
    main()
