import torch
from torch import Tensor
import typing
from typing import Any

from torchmetrics import Metric


class DoNothingMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", torch.tensor(1.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(1.0), dist_reduce_fx="sum")

    def update(self, preds: typing.Union[list, Tensor], target: typing.Union[list, Tensor]) -> None:
        self.correct += torch.sum(preds == target)
        self.total = torch.numel(target)

    def compute(self) -> Any:
        return "DoNothingMetric"
