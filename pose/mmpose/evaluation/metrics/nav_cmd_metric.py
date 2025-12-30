# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS


@METRICS.register_module()
class NavCmdMetric(BaseMetric):
    """Navigation Command Regression Metric.

    Computes regression metrics for navigation command prediction (vx, vy, v_yaw).

    Metrics computed: MSE, MAE and RMSE

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    default_prefix: Optional[str] = 'nav_cmd'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
                Each dict contains:
                - 'gt_instances': Ground truth InstanceData with 'nav_cmd' field
                - 'pred_instances': Predicted InstanceData with 'nav_cmd' field
        """
        for data_sample in data_samples:
            # Extract ground truth navigation commands
            # Shape: (num_instances, 3) where 3 = [vx, vy, v_yaw]
            gt = data_sample['gt_instances']
            pred = data_sample['pred_instances']

            # Get navigation commands as numpy arrays
            if 'nav_cmd' in gt:
                gt_nav_cmd = np.array(gt['nav_cmd'])  # (N, 3)
                pred_nav_cmd = np.array(pred['nav_cmd'])  # (N, 3)

                # Store results for later metric computation
                result = {
                    'gt_nav_cmd': gt_nav_cmd,
                    'pred_nav_cmd': pred_nav_cmd,
                }
                self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch. Each element
                contains:
                - 'gt_nav_cmd': Ground truth navigation commands (N, 3)
                - 'pred_nav_cmd': Predicted navigation commands (N, 3)

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f'Evaluating {self.__class__.__name__}...')

        # Aggregate all predictions and ground truths
        gt_nav_cmds = []
        pred_nav_cmds = []

        for result in results:
            gt_nav_cmds.append(result['gt_nav_cmd'])
            pred_nav_cmds.append(result['pred_nav_cmd'])

        gt_nav_cmds = np.concatenate(gt_nav_cmds, axis=0)
        pred_nav_cmds = np.concatenate(pred_nav_cmds, axis=0)

        # Compute errors
        errors = pred_nav_cmds - gt_nav_cmds 
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2

        # Overall metrics
        mse = np.mean(squared_errors)
        mae = np.mean(abs_errors)
        rmse = np.sqrt(mse)

        # Per-dimension metrics
        # Dimension 0: vx (forward/backward velocity)
        # Dimension 1: vy (left/right velocity)
        # Dimension 2: v_yaw (angular velocity)
        mse_vx = np.mean(squared_errors[:, 0])
        mse_vy = np.mean(squared_errors[:, 1])
        mse_vyaw = np.mean(squared_errors[:, 2])

        mae_vx = np.mean(abs_errors[:, 0])
        mae_vy = np.mean(abs_errors[:, 1])
        mae_vyaw = np.mean(abs_errors[:, 2])

        rmse_vx = np.sqrt(mse_vx)
        rmse_vy = np.sqrt(mse_vy)
        rmse_vyaw = np.sqrt(mse_vyaw)

        # Compile metrics dictionary
        metrics = {
            # Overall metrics
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),

            # Per-dimension MSE
            'MSE/vx': float(mse_vx),
            'MSE/vy': float(mse_vy),
            'MSE/v_yaw': float(mse_vyaw),

            # Per-dimension MAE
            'MAE/vx': float(mae_vx),
            'MAE/vy': float(mae_vy),
            'MAE/v_yaw': float(mae_vyaw),

            # Per-dimension RMSE
            'RMSE/vx': float(rmse_vx),
            'RMSE/vy': float(rmse_vy),
            'RMSE/v_yaw': float(rmse_vyaw),
        }

        return metrics
