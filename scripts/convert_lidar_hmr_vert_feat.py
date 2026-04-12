#!/usr/bin/env python3
"""Convert LiDAR-HMR vert_feat numpy dumps into ViT depth embedding files.

Expected input files:
- one .npy per sample, each containing either:
  - (V, C) typically (6890, 48), or
  - (1, V, C) / (B, V, C)

Output:
- one .npy per sample with shape (num_tokens, C), ready for
  VisionTransformerWithDepth(depth_embed_path=...).

Notes:
- This script preserves channel width C by default (typically 48).
- In your model config, set depth_embed_dim to match C (e.g. 48). The backbone's
  depth projection layer will map to RGB embed dims internally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _pool_tokens(arr: np.ndarray, num_tokens: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f'Expected (V,C) or (B,V,C), got {arr.shape}')
    x = torch.from_numpy(arr).float()          # (B, V, C)
    x = x.transpose(1, 2).contiguous()         # (B, C, V)
    x = torch.nn.functional.adaptive_avg_pool1d(x, num_tokens)
    x = x.transpose(1, 2).contiguous()         # (B, T, C)
    return x.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert LiDAR-HMR vert_feat .npy files to depth token .npy files.')
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing LiDAR-HMR vert_feat .npy files.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to write depth token .npy files.')
    parser.add_argument(
        '--num-tokens',
        type=int,
        default=26,
        help='Number of output depth tokens per sample.')
    parser.add_argument(
        '--glob',
        default='*.npy',
        help='Glob pattern for input files (default: *.npy).')
    parser.add_argument(
        '--squeeze-batch',
        action='store_true',
        help='If set, save first batch item as (T, C) instead of (B, T, C).')
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f'No files matched {args.glob} in {in_dir}')

    converted = 0
    for file in files:
        arr = np.load(file)
        pooled = _pool_tokens(arr, args.num_tokens)
        if args.squeeze_batch:
            if pooled.shape[0] != 1:
                raise ValueError(
                    f'--squeeze-batch requires single-batch input, got {pooled.shape} for {file}')
            pooled = pooled[0]  # (T, C)
        out_path = out_dir / file.name
        np.save(out_path, pooled)
        converted += 1

    print(f'Converted {converted} files to {out_dir}')
    print('Set backbone depth_embed_dim to output channel count (typically 48).')


if __name__ == '__main__':
    main()
