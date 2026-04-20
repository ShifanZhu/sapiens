# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Read Kinect v2 depth frames from Panoptic ``depthdata.dat``.

Format matches CMU PanopticStudio ``readDepthIndex_1basedIdx.m``:
  - Each frame is ``512 * 424`` uint16 values (depth in millimetres; 0 = invalid).
  - Frames are stored contiguously; frame ``idx`` (1-based) starts at byte offset
    ``2 * 512 * 424 * (idx - 1)``.
  - After reshape/transpose/flip, the image is ``(424, 512)`` with the same
    layout as the MATLAB toolbox (height x width).
"""

from __future__ import annotations

import os
from typing import BinaryIO

import numpy as np

# From panoptic-toolbox matlab/kinoptic-tools/readDepthIndex_1basedIdx.m
_DEPTH_W = 512
_DEPTH_H = 424
_FRAME_BYTES = 2 * _DEPTH_W * _DEPTH_H


def depth_frame_count(file_size_bytes: int) -> int:
    if file_size_bytes % _FRAME_BYTES != 0:
        raise ValueError(
            f'depthdata.dat size {file_size_bytes} is not a multiple of '
            f'frame size {_FRAME_BYTES}')
    return file_size_bytes // _FRAME_BYTES


def read_depth_frame_1based(path: str, idx_1based: int) -> np.ndarray:
    """Load one depth frame (same semantics as MATLAB, **1-based** index).

    Args:
        path: Path to ``depthdata.dat``.
        idx_1based: Frame index starting at 1 (first frame = 1).

    Returns:
        ``float32`` array of shape ``(424, 512)``. Invalid pixels are ``0``.
    """
    if idx_1based < 1:
        raise ValueError(f'idx_1based must be >= 1, got {idx_1based}')
    offset = _FRAME_BYTES * (idx_1based - 1)
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = np.fromfile(f, dtype=np.uint16, count=_DEPTH_W * _DEPTH_H)
    if raw.size != _DEPTH_W * _DEPTH_H:
        raise EOFError(
            f'Could not read full frame at 1-based index {idx_1based} from {path}')
    # MATLAB: reshape(data1, 512, 424)'  then fliplr
    im = raw.reshape((_DEPTH_W, _DEPTH_H), order='F').astype(np.float32).T
    im = im[:, ::-1].copy()
    return im


def read_depth_frame_1based_fp(
    fp: BinaryIO,
    idx_1based: int,
    *,
    file_size: int | None = None,
) -> np.ndarray:
    """Same as :func:`read_depth_frame_1based` but from an open file object."""
    if idx_1based < 1:
        raise ValueError(f'idx_1based must be >= 1, got {idx_1based}')
    if file_size is not None:
        nframes = depth_frame_count(file_size)
        if idx_1based > nframes:
            raise ValueError(
                f'1-based frame index {idx_1based} out of range '
                f'(file has {nframes} frames)')
    offset = _FRAME_BYTES * (idx_1based - 1)
    fp.seek(offset)
    raw = np.fromfile(fp, dtype=np.uint16, count=_DEPTH_W * _DEPTH_H)
    if raw.size != _DEPTH_W * _DEPTH_H:
        raise EOFError(f'Could not read full frame at 1-based index {idx_1based}')
    im = raw.reshape((_DEPTH_W, _DEPTH_H), order='F').astype(np.float32).T
    im = im[:, ::-1].copy()
    return im


def get_depthdata_path(sequence_dir: str, kinect_node: int) -> str:
    """Return path to ``kinect_shared_depth/KINECTNODE{node}/depthdata.dat``."""
    if kinect_node < 1 or kinect_node > 10:
        raise ValueError(f'kinect_node must be 1..10, got {kinect_node}')
    return os.path.join(
        sequence_dir, 'kinect_shared_depth', f'KINECTNODE{kinect_node}',
        'depthdata.dat')
