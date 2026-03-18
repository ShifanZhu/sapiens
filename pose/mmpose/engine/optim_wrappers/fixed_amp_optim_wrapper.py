# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Fixed AMP optimizer wrapper.

The upstream Sapiens fork of MMEngine's ``AmpOptimWrapper`` zeros out
inf/nan gradients *before* ``loss_scaler.unscale_()``.  This prevents
the GradScaler from detecting overflow, so it never reduces the loss
scale.  The result is that once overflow starts, every subsequent
iteration has zero gradients and the model stops learning.

This subclass overrides ``step()`` to remove that hack and restore the
standard PyTorch AMP flow: unscale → clip → step (skipped on overflow)
→ update (reduces scale on overflow).
"""

from mmengine.optim import AmpOptimWrapper

from mmpose.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class FixedAmpOptimWrapper(AmpOptimWrapper):
    """AmpOptimWrapper with correct overflow handling."""

    def step(self, **kwargs):
        if self.clip_grad_kwargs:
            self.loss_scaler.unscale_(self.optimizer)
            self._clip_grad()
        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)
