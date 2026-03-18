"""Hook to log epoch-averaged training MPJPE to TensorBoard.

Accumulates per-batch ``mpjpe`` and ``mpjpe_abs`` values from the head's
``loss()`` output, computes epoch averages, and writes them as
``mpjpe/rel/train`` and ``mpjpe/abs/train`` scalars.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from mmengine.hooks import Hook

from mmpose.registry import HOOKS


@HOOKS.register_module()
class TrainMPJPEAveragingHook(Hook):
    """Epoch-averaged training MPJPE logger.

    Reads ``outputs['mpjpe']`` and ``outputs['mpjpe_abs']`` from each
    training iteration and logs their epoch averages to TensorBoard.
    """

    def __init__(self) -> None:
        self._mpjpe_buffer: list[float] = []
        self._mpjpe_abs_buffer: list[float] = []

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[dict],
        outputs: dict,
    ) -> None:
        if 'mpjpe' in outputs:
            val = outputs['mpjpe']
            self._mpjpe_buffer.append(
                val.item() if isinstance(val, torch.Tensor) else float(val))
        if 'mpjpe_abs' in outputs:
            val = outputs['mpjpe_abs']
            self._mpjpe_abs_buffer.append(
                val.item() if isinstance(val, torch.Tensor) else float(val))

    def after_train_epoch(self, runner) -> None:
        if not self._mpjpe_buffer:
            return

        avg_rel = sum(self._mpjpe_buffer) / len(self._mpjpe_buffer)
        avg_abs = (sum(self._mpjpe_abs_buffer) / len(self._mpjpe_abs_buffer)
                   if self._mpjpe_abs_buffer else 0.0)

        # Write to TensorBoard
        tb_writer = self._get_tb_writer(runner)
        if tb_writer is not None:
            tb_writer.add_scalar('mpjpe/rel/train', avg_rel, runner.epoch)
            tb_writer.add_scalar('mpjpe/abs/train', avg_abs, runner.epoch)

        # Reset
        self._mpjpe_buffer.clear()
        self._mpjpe_abs_buffer.clear()

    @staticmethod
    def _get_tb_writer(runner):
        """Return TensorBoard SummaryWriter or None."""
        try:
            tb = runner.visualizer._vis_backends.get(
                'TensorboardVisBackend')
            if tb is None:
                return None
            if not tb._env_initialized:
                tb._init_env()
            return tb._tensorboard
        except Exception:
            return None
