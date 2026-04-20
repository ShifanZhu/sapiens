# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Load Panoptic sync tables and pick Kinect color/depth frame indices.

Follows ``demo_kinoptic_gen_ptcloud.m`` (CMU panoptic-toolbox):
  - ``selUnivTime = psync['hd']['univ_time'][hd_index]`` (MATLAB 1-based)
  - Color: ``argmin |selUnivTime - (color_univ - 6.25)|``
  - Depth: ``argmin |selUnivTime - depth_univ|``
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Tuple

import numpy as np


def _find_json(sequence_dir: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(sequence_dir, pattern))
    if not hits:
        raise FileNotFoundError(
            f'No file matching {pattern!r} under {sequence_dir}')
    hits.sort()
    return hits[0]


def load_ksync(sequence_dir: str) -> dict:
    return json.load(open(_find_json(sequence_dir, 'ksynctables*.json')))


def load_psync(sequence_dir: str) -> dict:
    return json.load(open(_find_json(sequence_dir, 'synctables*.json')))


def kinect_node_name(kinect_node: int) -> str:
    if kinect_node < 1 or kinect_node > 10:
        raise ValueError(f'kinect_node must be 1..10, got {kinect_node}')
    return f'KINECTNODE{kinect_node}'


def _get_univ_time_vec(ksync: dict, node: str, kind: str) -> np.ndarray:
    block = ksync['kinect'][kind][node]
    if isinstance(block, dict) and 'univ_time' in block:
        return np.asarray(block['univ_time'], dtype=np.float64)
    raise ValueError(f'Unsupported kinect.{kind}[{node}] schema: {type(block)}')


def select_kinect_frames(
    ksync: dict,
    sel_univ_time: float,
    kinect_node: int,
    *,
    color_time_offset_ms: float = 6.25,
    max_color_dt_ms: float = 30.0,
    max_depth_dt_ms: float = 17.0,
    max_color_depth_skew_ms: float = 6.5,
) -> Tuple[int, int, Dict[str, float]]:
    """Return **1-based** color and depth frame indices for one HD time."""
    node = kinect_node_name(kinect_node)
    ct = _get_univ_time_vec(ksync, node, 'color')
    dt = _get_univ_time_vec(ksync, node, 'depth')

    drel = np.abs(sel_univ_time - dt)
    crel = np.abs(sel_univ_time - (ct - color_time_offset_ms))
    dindex = int(np.argmin(drel)) + 1
    cindex = int(np.argmin(crel)) + 1

    dbg = {
        'color_dt_ms': float(crel[cindex - 1]),
        'depth_dt_ms': float(drel[dindex - 1]),
        'color_depth_skew_ms': float(
            abs(dt[dindex - 1] - ct[cindex - 1])),
    }
    if dbg['color_dt_ms'] > max_color_dt_ms or dbg['depth_dt_ms'] > max_depth_dt_ms:
        raise RuntimeError(
            f'sync match too loose: {dbg} (thresholds '
            f'color<={max_color_dt_ms}, depth<={max_depth_dt_ms})')
    if dbg['color_depth_skew_ms'] > max_color_depth_skew_ms:
        raise RuntimeError(
            f'color/depth skew too large: {dbg["color_depth_skew_ms"]} ms')
    return cindex, dindex, dbg


def hd_psync_index_for_output_frame(hd_output_frame: int) -> int:
    """0-based index into ``psync['hd']['univ_time']`` for toolbox file label.

    MATLAB adds +2 to user list before indexing HD time; output filename uses
    the original number. So for label ``hd_output_frame`` (e.g. 500),
    ``hd_index_matlab = hd_output_frame + 2`` and Python 0-based index is
    ``hd_output_frame + 1``.
    """
    return hd_output_frame + 1
