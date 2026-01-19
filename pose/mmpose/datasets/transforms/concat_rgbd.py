
from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConcatRGBD(BaseTransform):
    """Concatenate RGB and depth into 4-channel tensor."""
    
    def __init__(self,
                 depth_normalize: bool = True,
                 depth_min: Optional[float] = None,
                 depth_max: Optional[float] = None):
        super().__init__()
        self.depth_normalize = depth_normalize
        self.depth_min = depth_min
        self.depth_max = depth_max
    
    def transform(self, results: dict) -> dict:
        """Combine RGB and depth into 4-channel image."""
        img = results['img']
        depth = results['depth']
        
        # check shapes
        assert img.ndim == 3 and img.shape[2] == 3, \
            f"Expected RGB image with shape (H, W, 3), got {img.shape}"
        
        if depth.ndim == 3:
            assert depth.shape[2] == 1, \
                f"Expected depth with shape (H, W, 1) or (H, W), got {depth.shape}"
            depth = depth[..., 0]
        
        assert depth.ndim == 2, \
            f"Expected depth with shape (H, W), got {depth.shape}"
        
        assert img.shape[:2] == depth.shape[:2], \
            f"RGB and depth must have same spatial dimensions. " \
            f"RGB: {img.shape[:2]}, Depth: {depth.shape}"
        
        # normalize depth
        if self.depth_normalize:
            depth_min = self.depth_min if self.depth_min is not None else depth.min()
            depth_max = self.depth_max if self.depth_max is not None else depth.max()
            
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)
        
        # add channel dim and concat
        depth = depth[..., np.newaxis]
        img_4ch = np.concatenate([img, depth], axis=2)
        
        results['img'] = img_4ch
        if 'depth' in results:
            del results['depth']
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(depth_normalize={self.depth_normalize}'
        if self.depth_min is not None:
            repr_str += f', depth_min={self.depth_min}'
        if self.depth_max is not None:
            repr_str += f', depth_max={self.depth_max}'
        repr_str += ')'
        return repr_str

