# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadRGBD(LoadImageFromFile):
    """Load RGB + Depth from file or from np.ndarray already in results.

    Required Keys:
        - img_path (RGB path) OR img (optional)
        - depth_path (Depth path) OR depth (optional)

    Modified Keys:
        - img
        - depth
        - img_shape
        - ori_shape
        - img_path (optional)
        - depth_path (optional)

    Notes:
        - RGB is loaded via MMCV's LoadImageFromFile (same as MMPose LoadImage).
        - Depth is loaded via MMCV's LoadImageFromFile but with color_type='unchanged'
          so uint16 PNG depth will be preserved.
        - If `to_float32=True`, both img and depth are converted to float32.
        - If `depth_to_meters=True`, depth is divided by `depth_scale`
          (typical: uint16 depth in millimeters -> meters via 1000.0).

    Args:
        to_float32 (bool): Convert loaded arrays to float32. Default False.
        color_type (str): RGB color_type for mmcv.imfrombytes. Default 'color'.
        imdecode_backend (str): Backend for decoding. Default 'cv2'.
        backend_args (dict, optional): Backend args. Default None.
        ignore_empty (bool): Whether to allow missing file paths. Default False.
        depth_color_type (str): Depth color_type. Default 'unchanged'.
        depth_to_meters (bool): Convert depth to meters by dividing by depth_scale.
            Default True.
        depth_scale (float): Depth scale factor. Default 1000.0.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 backend_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 depth_color_type: str = 'unchanged',
                 depth_to_meters: bool = True,
                 depth_scale: float = 1000.0):
        super().__init__(to_float32=to_float32,
                         color_type=color_type,
                         imdecode_backend=imdecode_backend,
                         backend_args=backend_args,
                         ignore_empty=ignore_empty)
        self.depth_color_type = depth_color_type
        self.depth_to_meters = depth_to_meters
        self.depth_scale = depth_scale

        # A separate loader instance for depth so we can set color_type='unchanged'
        self._depth_loader = LoadImageFromFile(
            to_float32=to_float32,
            color_type=depth_color_type,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
            ignore_empty=ignore_empty)

    def _ensure_hw(self, arr: np.ndarray) -> np.ndarray:
        """Ensure depth is HxW (drop channel dim if HxWx1)."""
        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr[..., 0]
        return arr

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadRGBD`."""
        # ---- RGB ----
        if 'img' not in results:
            # Uses self's LoadImageFromFile.transform, reading from results['img_path']
            results = super().transform(results)
        else:
            img = results['img']
            assert isinstance(img, np.ndarray)
            if self.to_float32:
                img = img.astype(np.float32)
            if 'img_path' not in results:
                results['img_path'] = None
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
            results['img'] = img  # ensure possibly converted dtype is stored

        # ---- Depth ----
        if 'depth' not in results:
            # Need depth_path to load
            if 'depth_path' not in results:
                raise KeyError(
                    "LoadRGBD requires `depth_path` when `depth` is not provided."
                )

            # Load depth using separate depth loader
            depth_results = {'img_path': results['depth_path']}
            depth_results = self._depth_loader.transform(depth_results)
            depth = depth_results['img']
        else:
            depth = results['depth']
            assert isinstance(depth, np.ndarray)
            if self.to_float32:
                depth = depth.astype(np.float32)
            if 'depth_path' not in results:
                results['depth_path'] = None

        depth = self._ensure_hw(depth)

        # Optionally convert depth to meters
        # (commonly depth is uint16 in millimeters)
        if self.depth_to_meters:
            # Ensure float for scaling
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            depth = depth / float(self.depth_scale)

        results['depth'] = depth

        # Ensure shapes exist (RGB branch sets them; but keep safe if rgb was preloaded weirdly)
        if 'img' in results and ('img_shape' not in results or 'ori_shape' not in results):
            img = results['img']
            if isinstance(img, np.ndarray):
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]

        return results
