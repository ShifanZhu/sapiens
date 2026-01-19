
import os.path as osp
from typing import Optional

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class NTURGBDDataset(BaseCocoStyleDataset):
    """NTU RGB+D dataset loader for RGB + depth images."""
    
    METAINFO: dict = dict(
        dataset_name='ntu_rgbd',
        paper_info=dict(
            author='Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang',
            title='NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis',
            container='CVPR',
            year='2016',
            homepage='https://rose1.ntu.edu.sg/datasets/actionrecognition.asp',
        ),
        keypoint_info={},  # Will be set from annotation file
        skeleton_info={},  # Will be set from annotation file
    )
    
    def __init__(self,
                 ann_file: str = '',
                 data_prefix: dict = dict(img='', depth=''),
                 depth_path_suffix: Optional[str] = None,
                 depth_path_replace: Optional[dict] = None,
                 **kwargs):
        self.depth_path_suffix = depth_path_suffix
        self.depth_path_replace = depth_path_replace
        
        if 'depth' not in data_prefix:
            data_prefix['depth'] = ''
        
        super().__init__(
            ann_file=ann_file,
            data_prefix=data_prefix,
            **kwargs)
    
    def _load_annotations(self):
        """Load annotations and add depth paths."""
        instance_list, image_list = super()._load_annotations()
        
        for img in image_list:
            depth_path = self._get_depth_path(img)
            if depth_path:
                img['depth_path'] = depth_path
        
        return instance_list, image_list
    
    def _get_depth_path(self, img_info: dict) -> Optional[str]:
        """Get depth path from RGB path."""
        # check if already in annotation
        if 'depth_path' in img_info:
            depth_path = img_info['depth_path']
            if self.data_root:
                depth_path = osp.join(self.data_root, depth_path)
            return depth_path
        
        # get RGB path
        if 'img_path' in img_info:
            rgb_path = img_info['img_path']
        elif 'file_name' in img_info:
            rgb_path = osp.join(self.data_prefix.get('img', ''), img_info['file_name'])
        else:
            return None
        
        # construct depth path
        if self.depth_path_replace:
            depth_path = rgb_path
            for old, new in self.depth_path_replace.items():
                depth_path = depth_path.replace(old, new)
        elif self.depth_path_suffix:
            base, ext = osp.splitext(rgb_path)
            depth_path = base + self.depth_path_suffix + ext
        else:
            if 'depth' in self.data_prefix and self.data_prefix['depth']:
                img_prefix = self.data_prefix.get('img', '')
                if img_prefix and rgb_path.startswith(img_prefix):
                    rel_path = osp.relpath(rgb_path, img_prefix)
                    depth_path = osp.join(self.data_prefix['depth'], rel_path)
                else:
                    filename = osp.basename(rgb_path)
                    depth_path = osp.join(self.data_prefix['depth'], filename)
            else:
                return None
        
        return depth_path
    
    def parse_data_info(self, raw_data_info: dict):
        """Parse annotation and add depth_path."""
        data_info = super().parse_data_info(raw_data_info)
        
        if data_info is None:
            return None
        
        img_info = raw_data_info.get('raw_img_info', {})
        depth_path = self._get_depth_path(img_info)
        if depth_path:
            data_info['depth_path'] = depth_path
        
        return data_info



