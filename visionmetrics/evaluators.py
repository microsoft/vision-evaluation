import typing
import numpy
from .metrics import Metrics
from .states import StateContext


class Evaluator:
    def __init__(self, context: StateContext, metrics: typing.Dict[str, Metrics]):
        self._context = context
        self._metrics = metrics

    def add_predictions(self, predictions, targets):
        self.validate_predictions(predictions, targets)
        for s in self._context.states:
            s.add_predictions(predictions, targets)

        for m in self._metrics:
            m.add_predictions(predictions, targets)

    def get_report(self):
        report = {}
        for m in self._metrics:
            assert m.key_name not in report
            report[m.key_name] = m.compute()
        return report

    def reset(self):
        for s in self._context.states:
            s.reset()
        for m in self._metrics:
            m.reset()

    def validate_predictions(self, predictions, targets):
        raise NotImplementedError


class ClassificationMulticlassEvaluator(Evaluator):
    def validate_predictions(self, predictions, targets):
        if not isinstance(predictions, numpy.array) or not isinstance(targets, numpy.array):
            raise TypeError(f"Predictions and targets must be numpy.array. {type(predictions)=}, {type(targets)=}")
        if len(predictions.shape) != 2:
            raise ValueError(f"Unexpected shape {predictions.shape=}")
        if len(targets.shape) != 1:
            raise ValueError(f"Unexpected shape {targets.shape=}")
        if len(predictions) != len(targets):
            raise ValueError(f"The size of predictions and targets doesn't match. {len(predictions)=}, {len(targets)=}")


class ClassificationMultilabelEvaluator(Evaluator):
    def validate_predictions(self, predictions, targets):
        if not isinstance(predictions, numpy.array) or not isinstance(targets, numpy.array):
            raise TypeError(f"Predictions and targets must be numpy.array. {type(predictions)=}, {type(targets)=}")
        if len(predictions.shape) != 2:
            raise ValueError(f"Unexpected shape {predictions.shape=}")
        if len(targets.shape) != 2:
            raise ValueError(f"Unexpected shape {targets.shape=}")
        if predictions.shape != targets.shape:
            raise ValueError(f"The size of predictions and targets doesn't match. {predictions.shape=}, {targets.shape=}")


class ObjectDetectionEvaluator(Evaluator):
    def validate_predictions(self, predictions, targets):
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise TypeError(f"Predictions and targets must be a list. {type(predictions)=}, {type(targets)=}")
        if not all(p.shape[1] == 6 for p in predictions):
            raise ValueError("Prediction has an unexpected shape.")
        if not all(t.shape[1] == 5 for t in targets):
            raise ValueError("Targets has an unexpected shape.")
        if len(predictions) != len(targets):
            raise ValueError(f"The size of predictions and targets doesn't match. {len(predictions)=}, {len(targets)=}")
