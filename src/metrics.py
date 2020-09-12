from typing import Any

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.metric import TensorMetric, NumpyMetric


class GAPMetric(NumpyMetric):
    """
        A callback metric to calculate GAP@1 (Global Average Precision at k=1), also known as micro Average Precision (μAP).
        The function takes the predictions, confidence scores and labels vectors as input and
        outputs single value in the end of epoch.
        Args:
            prefix: str
                metric name.
            preds: torch.Tensor
                predictions as integer value.
            confs: torch.Tensor
                confidence score for predictions.
            labels: torch.Tensor
                true labels (integer values).
        Returns:
            GAP score: float
        Todo:
            Performance improvements.
            Currently intermediate data is stored as list of tensors on the same as input tensors (e.g. 'cuda:0').
            Possible options:
            - list of tensors and concat on demand (metric computation)
            - torch.cat(all_metric_values, new_metric_values) on each update
            - convert to numpy and concat
        """
    def __init__(self,
                 reduce_group: Any = None,
                 reduce_op: Any = torch.cat,
                 device: bool = None,
                 **kwargs):
        super().__init__(name="GAP",
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.preds = []
        self.confs = []
        self.labels = []

    def reset_stats(self):
        """Initialize empty tensors for each epoch run."""
        self.preds = []
        self.confs = []
        self.labels = []

    @torch.no_grad()
    def forward(self, output_batch, labels_batch):
        output_sm = F.softmax(output_batch, dim=1)
        confs_batch, preds_batch = torch.max(output_sm, dim=1)

        self.preds.append(preds_batch.cpu().numpy())
        self.confs.append(confs_batch.cpu().numpy())
        self.labels.append(labels_batch.cpu().numpy())

        return

    def compute_final(self):
        return self._compute_metric()

    @torch.no_grad()
    def _compute_metric(self):
        preds = np.concatenate(self.preds)
        confs = np.concatenate(self.confs)
        labels = np.concatenate(self.labels)

        res = pd.DataFrame({'pred': preds,
                            'conf': confs,
                            'label': labels})

        res = res.sort_values('conf', ascending=False, na_position='last')

        res['correct'] = (res.pred == res.label).astype(int)
        res['pred_k'] = res.correct.cumsum() / (np.arange(len(res)) + 1)
        res['term'] = res.pred_k * res.correct
        gap_score = res.term.sum()/res.label.count()
        return gap_score
