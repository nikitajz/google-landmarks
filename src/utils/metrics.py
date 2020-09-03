from typing import TYPE_CHECKING, Union, List, Dict
from catalyst.dl import Callback, CallbackOrder, CallbackNode
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class GAPMetricCallback(Callback):
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
                 prefix: str = "gap",
                 input_key: Union[str, List[str], Dict[str, str]] = "targets",
                 output_key: Union[str, List[str], Dict[str, str]] = "logits",
                 device: bool = None,
                 **kwargs):
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.preds = []
        self.confs = []
        self.labels = []
        self.device = device

    def on_epoch_start(self, runner: "IRunner"):
        """Initialize empty tensors for each epoch run."""
        self.preds = []
        self.confs = []
        self.labels = []

    @torch.no_grad()
    def on_batch_end(self, runner: "IRunner"):
        output_sm = F.softmax(runner.output[self.output_key], dim=1)
        confs_batch, preds_batch = torch.max(output_sm, dim=1)
        labels_batch = runner.input[self.input_key]

        self.preds.append(preds_batch)
        self.confs.append(confs_batch)
        self.labels.append(labels_batch)

        runner.batch_metrics[self.prefix] = self._compute_metric()

    def on_loader_end(self, runner: "IRunner"):
        runner.loader_metrics[self.prefix] = self._compute_metric()

    @torch.no_grad()
    def _compute_metric(self):
        preds = torch.cat(self.preds)
        confs = torch.cat(self.confs)
        labels = torch.cat(self.labels)

        res = pd.DataFrame({'pred': preds.cpu().numpy(),
                            'conf': confs.cpu().numpy(),
                            'label': labels.cpu().numpy()})

        res = res.sort_values('conf', ascending=False, na_position='last')

        res['correct'] = (res.pred == res.label).astype(int)
        res['pred_k'] = res.correct.cumsum() / (np.arange(len(res)) + 1)
        res['term'] = res.pred_k * res.correct
        gap_score = res.term.sum()/res.label.count()
        return gap_score
