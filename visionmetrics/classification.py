import torch
from torch import Tensor
import typing
from typing import Any

from torchmetrics import Metric
from vision_evaluation.prediction_filters import TopKPredictionFilter

# NOTE: This is an example of extending torchmetrics.Metric
class TopKAccuracy(Metric):
    def __init__(self, k: int) -> None:
        super().__init__()
        assert k > 0
        self.prediction_filter = TopKPredictionFilter(k)

        self.add_state("topk_correct_num", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_num", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: typing.Union[list, Tensor], targets: typing.Union[list, Tensor]) -> None:

        assert len(predictions) == len(targets)
        assert len(targets.shape) == 1

        n_sample = len(predictions)

        top_k_predictions = self.prediction_filter.filter(predictions, 'indices')
        self.topk_correct_num += len([1 for sample_idx in range(n_sample) if targets[sample_idx] in top_k_predictions[sample_idx]])
        self.total_num += n_sample

    def compute(self) -> Any:
        return {f'accuracy_{self.prediction_filter.identifier}': float(self.topk_correct_num) / self.total_num if self.total_num else 0.0}
