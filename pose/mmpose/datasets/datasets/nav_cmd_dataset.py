# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import exists, load
from mmengine.structures import InstanceData

from mmpose.registry import DATASETS
from mmpose.datasets.base_dataset import BaseDataset


@DATASETS.register_module()
class NavCmdDataset(BaseDataset):
    """Dataset for navigation command regression.

    This dataset loads RGB images and associated navigation commands (vx, vy, v_yaw)
    for training robot navigation models.

    Args:
        ann_file (str): Path to annotation file (JSON or TXT format)
        data_root (str): Root directory for data files
        data_prefix (dict): Prefix for different data types (default: {'img': ''})
        pipeline (list): Data processing pipeline
        test_mode (bool): Whether in test mode

    Annotation File Formats:
        
        JSON format:
        {
            "images": [
                {
                    "id": 0,
                    "file_name": "seq_001/frame_000000.jpg",
                    "sequence_id": "seq_001",
                    "frame_idx": 0,
                    "is_ref_frame": true
                }
            ],
            "annotations": [
                {
                    "image_id": 0,
                    "nav_cmd": [0.5, 0.2, 0.1],
                    "ref_emb": [...]  # Optional
                }
            ]
        }

        TXT/CSV format:
        image_path,sequence_id,frame_idx,vx,vy,v_yaw
        seq_001/frame_000000.jpg,seq_001,0,0.5,0.2,0.1
    """

    # Dataset metadata
    METAINFO = dict(
        dataset_name='nav_cmd_dataset',
        paper_info=dict( # TO BE FILLED
            author='Your Name',
            title='Navigation Command Dataset',
            year='2026',
        )
    )

    def __init__(self,
                 ann_file: str = '',
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs
        )

    def load_data_list(self) -> List[dict]:
        """Load annotations from file.

        Returns:
            List[dict]: List of data info dictionaries
        """
        
        if not exists(self.ann_file):
            raise FileNotFoundError(f'Annotation file not found: {self.ann_file}')

        # Determine file format
        if self.ann_file.endswith('.json'):
            return self._load_json_annotations()
        elif self.ann_file.endswith(('.txt', '.csv')):
            return self._load_txt_annotations()
        else:
            raise ValueError(f'Unsupported annotation format: {self.ann_file}')

    def _load_json_annotations(self) -> List[dict]:
        """Load annotations from JSON file."""
        with open(self.ann_file, 'r') as f:
            data = json.load(f)

        images = {img['id']: img for img in data['images']}
        annotations = {ann['image_id']: ann for ann in data['annotations']}

        # Build a mapping from sequence_id to reference embedding
        # Reference embeddings are stored only in reference frame annotations
        seq_ref_emb = {}
        for img_id, img_info in images.items():
            if img_info.get('is_ref_frame', False) and img_id in annotations:
                seq_id = img_info.get('sequence_id')
                ann = annotations[img_id]
                if seq_id and 'ref_emb' in ann:
                    seq_ref_emb[seq_id] = np.array(ann['ref_emb'], dtype=np.float32)

        data_list = []
        for img_id, img_info in images.items():
            if img_id not in annotations:
                continue

            ann = annotations[img_id]
            seq_id = img_info.get('sequence_id', None)

            # Look up reference embedding for this sequence
            # If FiLM conditioning is enabled, all frames need access to ref_emb
            ref_emb = seq_ref_emb.get(seq_id) if seq_id else None

            # Build data info dict
            data_info = dict(
                # Image information
                img_path=os.path.join(self.data_prefix['img'], img_info['file_name']),
                img_id=img_id,

                # Sequence/video information (for caching)
                sequence_id=seq_id,
                frame_idx=img_info.get('frame_idx', None),
                is_ref_frame=img_info.get('is_ref_frame', False),

                # Navigation command (ground truth)
                nav_cmd=np.array(ann['nav_cmd'], dtype=np.float32),

                # Reference embedding (looked up from reference frame)
                ref_emb=ref_emb,
            )

            data_list.append(data_info)

        return data_list

    def _load_txt_annotations(self) -> List[dict]:
        """Load annotations from TXT/CSV file."""
        data_list = []

        with open(self.ann_file, 'r') as f:
            lines = f.readlines()

        # Skip header if present
        if 'image_path' in lines[0] or 'img_path' in lines[0]:
            lines = lines[1:]

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Parse CSV line: image_path,sequence_id,frame_idx,vx,vy,v_yaw
            parts = line.split(',')

            if len(parts) < 6:
                print(f"Warning: Skipping malformed line {idx}: {line}")
                continue

            img_path, seq_id, frame_idx, vx, vy, v_yaw = parts[:6]

            # Build data info dict
            data_info = dict(
                # Image information
                img_path=os.path.join(self.data_prefix['img'], img_path.strip()),
                img_id=idx,

                # Sequence information
                sequence_id=seq_id.strip(),
                frame_idx=int(frame_idx.strip()),
                is_ref_frame=(int(frame_idx.strip()) == 0),  # First frame is reference

                # Navigation command
                nav_cmd=np.array([float(vx), float(vy), float(v_yaw)], dtype=np.float32),

                # No ref_emb in simple format
                ref_emb=None,
            )

            data_list.append(data_info)

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw data info to format required by pipeline.

        This method converts the loaded data info into MMPose's standard format
        with gt_instance_labels and metainfo.

        Args:
            raw_data_info: Raw data info dict from load_data_list()

        Returns:
            dict: Parsed data info with standard MMPose format
        """
        data_info = dict(
            # Image path
            img_path=raw_data_info['img_path'],
            img_id=raw_data_info['img_id'],
        )

        # ===== Create gt_instance_labels (InstanceData) =====
        # This is where we store per-instance annotations
        gt_instance_labels = InstanceData()
        gt_instance_labels.nav_cmd = raw_data_info['nav_cmd'] 

        # Add reference embedding if present
        if raw_data_info['ref_emb'] is not None:
            gt_instance_labels.ref_emb = raw_data_info['ref_emb']

        data_info['gt_instance_labels'] = gt_instance_labels

        # ===== Create metainfo (dict) =====
        # Store metadata and auxiliary information
        metainfo = dict(
            img_id=raw_data_info['img_id'],
        )

        # Add sequence/caching information if present
        if raw_data_info['sequence_id'] is not None:
            metainfo['sequence_id'] = raw_data_info['sequence_id']
        if raw_data_info['frame_idx'] is not None:
            metainfo['frame_idx'] = raw_data_info['frame_idx']
        if 'is_ref_frame' in raw_data_info:
            metainfo['is_ref_frame'] = raw_data_info['is_ref_frame']

        data_info['metainfo'] = metainfo

        return data_info
