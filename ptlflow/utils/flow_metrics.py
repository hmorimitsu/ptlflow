"""Handle and compute metrics for optical flow and related estimations.

This handler is designed according to the torchmetrics specifications. Besides accuracy metrics for optical flow, it can also
compute basic metrics for occlusion, motion boundary and flow confidence estimations.
"""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from typing import Dict

from torchmetrics import Metric
import torch


class FlowMetrics(Metric):
    """Handler for optical flow and related metrics.

    Attributes
    ----------
    average_mode : str, default 'epoch_mean'
        How the final metric is averaged. It can be either 'epoch_mean' or 'ema' (exponential moving average).
    ema_decay : float, default 0.99
        The decay to be applied if average_mode is 'ema'.
    prefix : str, optional
        A prefix string that will be attached to the metric names.
    """

    def __init__(
        self,
        dist_sync_on_step: bool = False,
        prefix: str = '',
        average_mode: str = 'epoch_mean',
        ema_decay: float = 0.99
    ) -> None:
        """Initialize FlowMetrics.

        Parameters
        ----------
        dist_sync_on_step : bool, default False
            Used by torchmetrics to sync metrics between multiple processes.
        prefix : str, optional
            A prefix string that will be attached to the metric names.
        average_mode : str, default 'epoch_mean'
            How the final metric is averaged. It can be either 'epoch_mean' or 'ema' (exponential moving average).
        ema_decay : float, default 0.99
            The decay to be applied if average_mode is 'ema'.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        assert average_mode in ['epoch_mean', 'ema']

        self.average_mode = average_mode
        self.prefix = prefix
        self.ema_decay = ema_decay
        self.ema_max_count = min(100, int(1.0/(1.0-ema_decay)))

        self.add_state("epe", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("epe_non_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("epe_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.add_state("px1", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("px1_non_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("px1_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.add_state("px3", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("px3_non_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("px3_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.add_state("px5", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("px5_non_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("px5_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.add_state("outlier", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("outlier_non_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("outlier_occ", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.add_state("occ_wf1", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("mb_wf1", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("conf_l1", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.add_state("sample_count", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("step_count", default=torch.tensor(0).float(), dist_reduce_fx="sum")

        self.include_occlusion = False

        self.used_keys = []

    def update(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> None:
        """Compute and update one step of the metrics.

        Parameters
        ----------
        preds : dict[str, torch.Tensor]
            The predictions of the optical flow model.
        targets : dict[str, torch.Tensor]
            The groundtruth of the predictions.
        """
        if self.average_mode == 'epoch_mean':
            prev_weight = 1.0
            next_weight = 1.0
        else:
            prev_weight = self.ema_decay
            next_weight = 1.0 - self.ema_decay

        batch_size = self._get_batch_size(targets['flows'])
        flow_pred = self._fix_shape(preds['flows'], batch_size)
        flow_target = self._fix_shape(targets['flows'], batch_size)

        valid_target = targets.get('valids')
        if valid_target is not None:
            valid_target = self._fix_shape(valid_target, batch_size)
        else:
            valid_target = torch.ones_like(flow_target[:, :1])
        valid_target = valid_target[:, 0]

        occlusion_target = targets.get('occs')
        if occlusion_target is not None:
            occlusion_target = self._fix_shape(occlusion_target, batch_size)

        epe = torch.norm(flow_pred - flow_target, p=2, dim=1)
        px1_mask = (epe < 1).float()
        px3_mask = (epe < 3).float()
        px5_mask = (epe < 5).float()
        target_norm = torch.norm(flow_target, p=2, dim=1)
        outlier_mask = ((epe > 8) & (epe > (0.1*target_norm))).float()
        self.used_keys = [('epe', 'epe', 'valid_target'),
                          ('px1', 'px1_mask', 'valid_target'),
                          ('px3', 'px3_mask', 'valid_target'),
                          ('px5', 'px5_mask', 'valid_target'),
                          ('outlier', 'outlier_mask', 'valid_target')]

        if occlusion_target is not None:
            valid_occ = occlusion_target[:, 0] * valid_target
            valid_non_occ = (1 - occlusion_target[:, 0]) * valid_target
            self.used_keys.extend([('epe_occ', 'epe', 'valid_occ'),
                                   ('epe_non_occ', 'epe', 'valid_non_occ'),
                                   ('px1_occ', 'px1_mask', 'valid_occ'),
                                   ('px1_non_occ', 'px1_mask', 'valid_non_occ'),
                                   ('px3_occ', 'px3_mask', 'valid_occ'),
                                   ('px3_non_occ', 'px3_mask', 'valid_non_occ'),
                                   ('px5_occ', 'px5_mask', 'valid_occ'),
                                   ('px5_non_occ', 'px5_mask', 'valid_non_occ'),
                                   ('outlier_occ', 'outlier_mask', 'valid_occ'),
                                   ('outlier_non_occ', 'outlier_mask', 'valid_non_occ')])
            self.include_occlusion = True

            if preds.get('occs') is not None:
                occlusion_pred = self._fix_shape(preds['occs'], batch_size)
                occ_wf1 = self._weighted_f1_score(occlusion_pred, occlusion_target)
                self.used_keys.extend([('occ_wf1', 'occ_wf1', 'valid_target')])

        if preds.get('mbs') is not None and targets.get('mbs') is not None:
            mb_pred = self._fix_shape(preds['mbs'], batch_size)
            mb_target = self._fix_shape(targets['mbs'], batch_size)
            mb_wf1 = self._weighted_f1_score(mb_pred, mb_target)
            self.used_keys.extend([('mb_wf1', 'mb_wf1', 'valid_target')])

        if preds.get('confs') is not None:
            conf_target = torch.exp(-torch.pow(flow_target - flow_pred, 2).sum(dim=1, keepdim=True))
            conf_pred = self._fix_shape(preds['confs'], batch_size)
            conf_l1 = torch.abs(conf_target - conf_pred)
            self.used_keys.extend([('conf_l1', 'conf_l1', 'valid_target')])

        for v1, v2, v3 in self.used_keys:
            setattr(self, v1, prev_weight*getattr(self, v1) + next_weight*self._compute_total(locals()[v2], locals()[v3]))

        self.sample_count += batch_size
        self.step_count += 1

    def calculate_metrics(self) -> Dict[str, torch.Tensor]:
        """Compute and return the average of all metrics.

        On Pytorch-Lightning < 1.2, compute() automatically calls reset(). Sometimes this is not desirable, so the metrics
        are calculated here in this other function, which can be called externally.

        Returns
        -------
        Dict[str, torch.Tensor]
            The average of the metrics.
        """
        if self.average_mode == 'epoch_mean':
            divider = self.sample_count
        else:
            divider = 1.0
            if self.step_count < self.ema_max_count:
                divider -= self.ema_decay ** self.step_count

        metrics = {}
        for k in self.used_keys:
            metrics[self.prefix + k[0]] = getattr(self, k[0]) / divider

        return metrics

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute and return the average of all metrics.

        Called internally by torchmetrics.

        Returns
        -------
        Dict[str, torch.Tensor]
            The average of the metrics.
        """
        return self.calculate_metrics()

    def _compute_total(
        self,
        tensor: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        tensor = tensor * valid_mask
        tensor = tensor.view(tensor.shape[0], -1)
        valid_sum = valid_mask.view(valid_mask.shape[0], -1).sum(dim=1)
        valid_sum = torch.clamp(valid_sum, 1)
        tensor = tensor.sum(dim=1) / valid_sum
        if self.average_mode == 'epoch_mean':
            tensor = tensor.sum()
        else:
            tensor = tensor.mean()
        return tensor

    def _weighted_f1_score(self,
                           pred: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        f1_pos = self._f1_score(pred, target)
        f1_neg = self._f1_score(1-pred, 1-target)

        target_pos = (target > 0.5).float()
        target_pos = target_pos.view(target_pos.shape[0], target_pos.shape[1], -1)
        n_pos = target_pos.sum(dim=2)[:, :, None, None]
        w_pos = n_pos / target_pos.shape[2]

        target_neg = (target <= 0.5).float()
        target_neg = target_neg.view(target_neg.shape[0], target_neg.shape[1], -1)
        n_neg = target_neg.sum(dim=2)[:, :, None, None]
        w_neg = n_neg / target_neg.shape[2]

        f1_weighted = w_pos*f1_pos + w_neg*f1_neg

        return f1_weighted

    def _f1_score(self,
                  pred: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        pred_bin = (pred > 0.5).float()
        target_bin = (target > 0.5).float()

        tp = (pred_bin * target_bin)
        fp = ((1-pred_bin) * target_bin)
        fn = (pred_bin * (1-target_bin))

        eps = torch.finfo(pred.dtype).eps
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2*precision*recall / (precision + recall + eps)

        return f1

    def _fix_shape(self,
                   tensor: torch.Tensor,
                   batch_size: int) -> torch.Tensor:
        if len(tensor.shape) == 2:
            tensor = tensor[None, None]
        elif len(tensor.shape) == 3:
            if tensor.shape[0] == batch_size:
                tensor = tensor[:, None]
            else:
                tensor = tensor[None]
        elif len(tensor.shape) == 5:
            tensor = tensor.view(tensor.shape[0]*tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4])
        return tensor

    def _get_batch_size(self,
                        flow_tensor: torch.Tensor) -> int:
        if len(flow_tensor.shape) < 4:
            return 1
        else:
            return flow_tensor.shape[0]
