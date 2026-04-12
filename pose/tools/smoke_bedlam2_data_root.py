#!/usr/bin/env python3
"""Quick check that BEDLAM2 data on disk loads through Bedlam2Dataset + val pipeline.

Expects ``DATA_ROOT`` to be the directory that *contains* ``data/label``, ``data/frames``,
``data/depth`` (i.e. the BEDLAM2 bundle root), **not** the inner ``.../BEDLAM2/data`` folder.

Example::

    conda activate sapiens
    python pose/tools/smoke_bedlam2_data_root.py \\
        --data-root "/media/s/Crucial X10/BEDLAM2" \\
        --seq-paths-file pose/data/bedlam2_splits/test_seqs.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--data-root',
        default=os.environ.get('BEDLAM2_DATA_ROOT', ''),
        help='BEDLAM2 bundle root (contains data/label, ...). '
             'Default: env BEDLAM2_DATA_ROOT',
    )
    p.add_argument(
        '--seq-paths-file',
        default='data/bedlam2_splits/test_seqs.txt',
        help='Relative to cwd or absolute; lists folder/seq.npz paths',
    )
    p.add_argument('--max-seqs', type=int, default=1,
                   help='Only index this many sequences (fast smoke test)')
    args = p.parse_args()
    if not args.data_root:
        print('ERROR: pass --data-root or set BEDLAM2_DATA_ROOT', file=sys.stderr)
        return 1

    pose_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(pose_root))
    os.chdir(pose_root)

    from mmengine.registry import init_default_scope

    init_default_scope('mmpose')
    import mmpose.datasets.transforms.bedlam2_transforms  # noqa: F401
    from mmpose.datasets.datasets.body3d.bedlam2_dataset import Bedlam2Dataset

    img_h, img_w = 640, 384
    val_pipeline = [
        dict(type='LoadBedlamLabels', depth_required=True, filter_invalid=False),
        dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
        dict(type='SubtractRootJoint'),
        dict(type='PackBedlamInputs',
             meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                        'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
    ]

    ds = Bedlam2Dataset(
        data_root=args.data_root,
        seq_paths_file=args.seq_paths_file,
        frame_stride=1,
        pipeline=val_pipeline,
        test_mode=True,
        max_refetch=10,
        max_seqs=args.max_seqs,
    )
    print(f'Indexed samples: {len(ds)}')
    out = ds.prepare_data(0)
    assert isinstance(out, dict) and 'inputs' in out
    print('prepare_data(0): OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
