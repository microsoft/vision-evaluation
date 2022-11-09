import numpy as np
from .metrics import MetricsRegistry, Metrics
from .prediction_filters import TopKPredictionFilter, ThresholdPredictionFilter


@MetricsRegistry.register('topk_accuracy', 'classification_multiclass')
class TopKAccuracyMetrics(Metrics):
    """Top k accuracy evaluator for multiclass classification"""

    def __init__(self, context, k: int):
        super().__init__(context)
        assert k > 0
        self._k = k
        self.prediction_filter = TopKPredictionFilter(k)

    @property
    def key_name(self):
        return f'top{self._k}_accuracy'

    def reset(self):
        super().reset()
        self.total_num = 0
        self.topk_correct_num = 0

    def add_predictions(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the model output numpy array. Shape (N, num_class)
            targets: the golden truths. Shape (N,)
        """
        assert len(predictions) == len(targets)
        assert len(targets.shape) == 1

        n_sample = len(predictions)

        top_k_predictions = self.prediction_filter.filter(predictions, 'indices')
        self.topk_correct_num += len([1 for sample_idx in range(n_sample) if targets[sample_idx] in top_k_predictions[sample_idx]])
        self.total_num += n_sample

    def compute(self):
        return float(self.topk_correct_num) / self.total_num if self.total_num else 0.0


def _targets_to_mat(targets, n_class):
    if len(targets.shape) == 1:
        target_mat = np.zeros((len(targets), n_class), dtype=int)
        for i, t in enumerate(targets):
            target_mat[i, t] = 1
    else:
        target_mat = targets

    return target_mat


@MetricsRegistry.register('threshold_accuracy', ('classification_multiclass', 'classification_multilabel'))
class ThresholdAccuracyMetrics(Metrics):
    """Threshold-based accuracy evaluator for multilabel classification, calculated in a sample-based flavor
    Note that
        1. this could be used for multi-class classification, but does not make much sense
        2. sklearn.metrics.accuracy_score actually is computing exact match ratio for multi-label classification, which is too harsh
    """

    def __init__(self, context, threshold):
        super().__init__(context)
        self.prediction_filter = ThresholdPredictionFilter(threshold)

    @property
    def key_name(self):
        return f'accuracy_{self.prediction_filter.identifier}'

    def reset(self):
        super().reset()
        self.num_sample = 0
        self.sample_accuracy_sum = 0

    def add_predictions(self, predictions, targets):
        """ Add a batch of predictions for evaluation.
        Args:
            predictions: the model output array. Shape (N, num_class)
            targets: the ground truths. Shape (N, num_class) for multi-label (or (N,) for multi-class)
        """

        assert len(predictions) == len(targets)

        num_samples = len(predictions)
        target_mat = _targets_to_mat(targets, predictions.shape[1])

        prediction_over_threshold = self.prediction_filter.filter(predictions, 'vec')
        n_correct_predictions = np.multiply(prediction_over_threshold, target_mat).sum(1)  # shape (N,)
        n_total = (np.add(prediction_over_threshold, target_mat) >= 1).sum(1)  # shape (N,)
        n_total[n_total == 0] = 1  # To avoid zero-division. If n_total==0, num should be zero as well.
        self.sample_accuracy_sum += float((n_correct_predictions / n_total).sum())
        self.num_sample += num_samples

    def compute(self):
        return self.sample_accuracy_sum / self.num_sample if self.num_sample else 0.0
