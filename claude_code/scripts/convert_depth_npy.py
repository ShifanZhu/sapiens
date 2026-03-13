"""Convert compressed depth NPZ files to mmappable float16 NPY at training resolution.

Reads  : data/depth/npz/<folder>/<seq>.npz   (n_frames, 720, 1280) float32
Writes : data/depth/npy/<folder>/<seq>.npy   (n_frames, 384, 640)  float16

Benefits vs NPZ:
  - Memory-mappable: OS pages in only the frames you access, zero decompression.
  - Pre-resized to training resolution: eliminates runtime cv2.resize for depth.
  - Similar disk usage: ~47 MB/seq vs ~36 MB/seq compressed (NPZ uses 10:1 zlib).

Usage:
    conda run -n sapiens_gpu python convert_depth_npy.py \
        --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
        --out-h 384 --out-w 640 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def convert_one(args_tuple):
    npz_path, npy_path, out_h, out_w = args_tuple
    npy_path = Path(npy_path)
    if npy_path.exists():
        # Validate: try to mmap the file; delete and re-convert if truncated
        try:
            np.load(str(npy_path), mmap_mode="r")
            return str(npz_path), "skip"
        except (ValueError, OSError):
            npy_path.unlink()  # truncated — delete and fall through to re-convert
    try:
        with np.load(str(npz_path)) as f:
            depth = f["depth"]  # (n_frames, H, W) float32
        n_frames, h, w = depth.shape
        if h == out_h and w == out_w:
            resized = depth.astype(np.float16)
        else:
            resized = np.stack([
                cv2.resize(depth[i], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                for i in range(n_frames)
            ]).astype(np.float16)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), resized)
        return str(npz_path), "ok"
    except Exception as e:
        return str(npz_path), f"error: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/hang/repos_local/MMC/BEDLAM2Datatest")
    p.add_argument("--out-h",    type=int, default=384)
    p.add_argument("--out-w",    type=int, default=640)
    p.add_argument("--workers",  type=int, default=8)
    args = p.parse_args()

    data_root = Path(args.data_root)
    npz_root  = data_root / "data" / "depth" / "npz"
    npy_root  = data_root / "data" / "depth" / "npy"

    npz_files = sorted(npz_root.rglob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files → converting to {args.out_h}×{args.out_w} float16 NPY")

    tasks = []
    for npz in npz_files:
        rel = npz.relative_to(npz_root)
        npy = npy_root / rel.with_suffix(".npy")
        tasks.append((str(npz), str(npy), args.out_h, args.out_w))

    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            path, status = fut.result()
            if status == "skip":
                skipped += 1
            elif status == "ok":
                done += 1
            else:
                errors += 1
                print(f"  ERROR {path}: {status}")
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] converted={done} skipped={skipped} errors={errors}")

    print(f"\nDone. {done} converted, {skipped} skipped, {errors} errors.")

    # Report disk usage
    npy_size = sum(f.stat().st_size for f in npy_root.rglob("*.npy")) / 1e9
    npz_size = sum(f.stat().st_size for f in npz_root.rglob("*.npz")) / 1e9
    print(f"NPZ size: {npz_size:.1f} GB  →  NPY size: {npy_size:.1f} GB")


if __name__ == "__main__":
    main()
