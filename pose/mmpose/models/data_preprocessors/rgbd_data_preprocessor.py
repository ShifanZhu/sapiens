# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data pre-processor for RGBD pose estimation.

Since RGB normalization and depth normalization are applied inside the
``PackBedlamInputs`` transform (the pipeline step that produces the
``inputs`` tensor), this preprocessor is a thin pass-through that only
handles stacking samples into a batch tensor.  A 4-channel input tensor
``(B, 4, H, W)`` is returned unchanged.
"""

from mmengine.model import BaseDataPreprocessor

from mmpose.registry import MODELS


@MODELS.register_module()
class RGBDPoseDataPreprocessor(BaseDataPreprocessor):
    """Pass-through data preprocessor for RGBD inputs.

    The 4-channel ``inputs`` tensor produced by ``PackBedlamInputs``
    (RGB normalised + depth normalised) is passed directly to the model
    without any further transformation.

    Note:
        Normalisation is already applied inside the data pipeline, so no
        per-channel mean/std subtraction is performed here.
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Stack inputs and move to the correct device.

        Args:
            data (dict): Data from the dataloader with keys ``inputs`` and
                ``data_samples``.
            training (bool): Whether in training mode.

        Returns:
            dict: Same structure with tensors on the model device.
        """
        data = self.cast_data(data)
        return data
