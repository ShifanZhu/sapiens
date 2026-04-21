#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Assemble preprocess output frame folders into one MP4 (e.g. *_pose_gt_bgr.jpg)."""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _frame_sort_key(path: str) -> tuple[int, str]:
    m = re.search(r"frame_(\d+)", path)
    if m:
        return (int(m.group(1)), path)
    return (0, path)


def collect_images(input_dir: str, pattern: str) -> list[str]:
    search = os.path.join(os.path.abspath(input_dir), "frame_*", pattern)
    paths = glob.glob(search)
    paths = sorted(paths, key=_frame_sort_key)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stitch frame_*/<pattern> images into a video (sorted by HD frame index)."
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Preprocess output root containing frame_0000... subfolders.",
    )
    ap.add_argument(
        "--pattern",
        default="*_pose_gt_bgr.jpg",
        help="Glob relative to each frame_* folder (default: *_pose_gt_bgr.jpg).",
    )
    ap.add_argument("--fps", type=float, default=30.0, help="Output frame rate.")
    ap.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output video path (.mp4 recommended).",
    )
    ap.add_argument(
        "--ffmpeg",
        default=shutil.which("ffmpeg") or "ffmpeg",
        help="ffmpeg binary (default: PATH).",
    )
    args = ap.parse_args()

    paths = collect_images(args.input_dir, args.pattern)
    if not paths:
        print(
            f"No images matched {args.pattern!r} under {args.input_dir!r}/frame_*/",
            file=sys.stderr,
        )
        return 1

    ffmpeg = args.ffmpeg
    if not os.path.isfile(ffmpeg) and shutil.which(ffmpeg) is None:
        print(
            "ffmpeg not found; install ffmpeg or pass --ffmpeg /path/to/ffmpeg",
            file=sys.stderr,
        )
        return 1

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pose_frames_vid_") as tmp:
        for i, src in enumerate(paths, start=1):
            dst = os.path.join(tmp, f"{i:06d}.jpg")
            os.symlink(src, dst)

        seq = os.path.join(tmp, "%06d.jpg")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-i",
            seq,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed with exit code {e.returncode}", file=sys.stderr)
            return e.returncode or 1

    print(f"Wrote {len(paths)} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
