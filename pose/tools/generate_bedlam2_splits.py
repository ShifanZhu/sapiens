#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Generate train / val / test split files for BEDLAM2.

Reads ``data/overview.txt`` from the BEDLAM2 data root, filters sequences
by the given criteria, and writes three text files (one sequence path per
line) that can be passed to ``Bedlam2Dataset`` via ``seq_paths_file``.

Usage::

    python pose/tools/generate_bedlam2_splits.py \\
        --data-root /media/s/SF_backup/bedlam2/ \\
        --output-dir pose/data/bedlam2_splits/ \\
        --val-ratio 0.1 --test-ratio 0.1 --seed 2026

Prerequisites:
    - Pre-extracted JPEG frames (run ``claude_code/scripts/extract_frames.py``)
    - Pre-converted depth NPY files (run ``claude_code/scripts/convert_depth_npy.py``)
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def get_seq_paths(
    overview_path: str,
    single_body_only: bool = False,
    skip_missing_body: bool = True,
    depth_required: bool = True,
    mp4_required: bool = False,
    frames_root: str | None = None,
) -> list[str]:
    """Return filtered list of ``"folder/seq.npz"`` relative paths."""
    seq_paths = []
    with open(overview_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, conditions = line.split(':', 1)
            if single_body_only and 'not_single_body=True' in conditions:
                continue
            if skip_missing_body and 'missing_body=True' in conditions:
                continue
            if depth_required and 'no_depth=True' in conditions:
                continue
            if mp4_required and 'no_mp4=True' in conditions:
                continue
            folder, seq = name.strip().split('/')
            if frames_root is not None:
                frames_dir = Path(frames_root) / folder / seq
                if not frames_dir.is_dir():
                    continue
            seq_paths.append(f'{folder}/{seq}.npz')
    return seq_paths


def split_sequences(
    seq_paths: list[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2026,
) -> tuple[list[str], list[str], list[str]]:
    """Randomly split at sequence level to prevent data leakage."""
    rng = random.Random(seed)
    paths = list(seq_paths)
    rng.shuffle(paths)

    n = len(paths)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))

    test_paths = paths[:n_test]
    val_paths = paths[n_test: n_test + n_val]
    train_paths = paths[n_test + n_val:]
    return train_paths, val_paths, test_paths


def main():
    parser = argparse.ArgumentParser(
        description='Generate BEDLAM2 train/val/test split files')
    parser.add_argument('--data-root', required=True,
                        help='Path to BEDLAM2 data root directory')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write train/val/test .txt files')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--single-body-only', action='store_true',
                        help='Skip multi-person sequences')
    parser.add_argument('--no-depth-required', action='store_true',
                        help='Include sequences without depth')
    parser.add_argument('--mp4-required', action='store_true',
                        help='Require mp4 video to be present')
    args = parser.parse_args()

    overview_path = os.path.join(args.data_root, 'data', 'overview.txt')
    frames_root = os.path.join(args.data_root, 'data', 'frames')

    print(f'Reading overview: {overview_path}')
    seq_paths = get_seq_paths(
        overview_path,
        single_body_only=args.single_body_only,
        skip_missing_body=True,
        depth_required=not args.no_depth_required,
        mp4_required=args.mp4_required,
        frames_root=frames_root,
    )
    print(f'Found {len(seq_paths)} valid sequences after filtering')

    train, val, test = split_sequences(
        seq_paths,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f'Split: {len(train)} train / {len(val)} val / {len(test)} test')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, paths in [('train', train), ('val', val), ('test', test)]:
        out_path = out_dir / f'{name}_seqs.txt'
        with open(out_path, 'w') as f:
            f.write('\n'.join(paths) + '\n')
        print(f'Wrote {len(paths)} paths → {out_path}')


if __name__ == '__main__':
    main()
