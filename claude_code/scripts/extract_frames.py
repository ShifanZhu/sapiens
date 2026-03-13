"""Pre-extract video frames to JPEG files for fast random access during training.

Each video frame is saved as:
    data/frames/<folder>/<seq_name>/<frame_idx>.jpg

where <frame_idx> is the downsampled index (0, 1, 2, ...) corresponding to
video frame 0, 5, 10, ... (FRAME_STRIDE=5).

Usage:
    conda run -n sapiens_gpu python extract_frames.py \
        --data-root /home/hang/repos_local/MMC/BEDLAM2Datatest \
        --workers 8 \
        --quality 95
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.constants import FRAME_STRIDE
from data.splits import get_seq_paths


def extract_sequence(args):
    seq_rel, data_root, out_root, quality, stride = args
    label_path = Path(data_root) / "data" / "label" / seq_rel

    try:
        meta = np.load(str(label_path), allow_pickle=True)
        n_frames    = int(meta["n_frames"])
        folder_name = str(meta["folder_name"])
        seq_name    = str(meta["seq_name"])
        rotate_flag = bool(meta["rotate_flag"])
    except Exception as e:
        return seq_rel, 0, f"label error: {e}"

    video_path = (Path(data_root) / "data" / "mp4"
                  / f"{folder_name}_mp4" / folder_name / "mp4"
                  / f"{seq_name}.mp4")
    out_dir = out_root / folder_name / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already fully extracted
    existing = len(list(out_dir.glob("*.jpg")))
    if existing >= n_frames:
        return seq_rel, n_frames, "skip"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return seq_rel, 0, f"cannot open {video_path}"

    # Read ALL frames sequentially (no random seek) — much faster than per-frame seek.
    # H.264 sequential decode is fast; only save every stride-th frame.
    saved = 0
    video_frame_idx = 0
    frame_idx = 0
    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if video_frame_idx % stride == 0:
            out_path = out_dir / f"{frame_idx:05d}.jpg"
            if not out_path.exists():
                if rotate_flag:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
            frame_idx += 1
        video_frame_idx += 1

    cap.release()
    return seq_rel, saved, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/hang/repos_local/MMC/BEDLAM2Datatest")
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--quality",   type=int, default=95, help="JPEG quality 0-100")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_root  = data_root / "data" / "frames"
    out_root.mkdir(parents=True, exist_ok=True)

    seq_paths = get_seq_paths(
        str(data_root / "data" / "overview.txt"),
        single_body_only=True,
        skip_missing_body=True,
        depth_required=True,
        mp4_required=True,
    )
    print(f"Extracting frames for {len(seq_paths)} sequences → {out_root}")
    print(f"Using {args.workers} workers, JPEG quality={args.quality}")

    tasks = [(s, str(data_root), out_root, args.quality, FRAME_STRIDE) for s in seq_paths]

    done = skipped = errors = total_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(extract_sequence, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            seq_rel, n, status = fut.result()
            if status == "skip":
                skipped += 1
            elif status == "ok":
                done += 1
                total_frames += n
            else:
                errors += 1
                print(f"  ERROR {seq_rel}: {status}")
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] done={done} skip={skipped} err={errors} "
                      f"frames={total_frames}")

    print(f"\nDone. {done} extracted, {skipped} skipped, {errors} errors.")
    print(f"Total frames saved: {total_frames}")


if __name__ == "__main__":
    main()
