import numpy as np
from vision_evaluation.evaluators import Evaluator, _targets_to_mat
from vision_evaluation.prediction_filters import PredictionFilter


class SamplePrecisionEvaluator(Evaluator):
    """
    Precision evaluator for both multi-class and multi-label classification, with average='samples'

    This evaluator demands less memory, by no storing the original predictions and targets.
    """

    def __init__(self, prediction_filter: PredictionFilter):
        super().__init__()
        self.prediction_filter = prediction_filter

    def add_predictions(self, predictions, targets):
        """ Add a batch of predictions for evaluation.
        Args:
            predictions: the model output array. Shape (N, num_class)
            targets: the ground truths. Shape (N, num_class) for multi-label or (N,) for multi-class
        """

        assert len(predictions) == len(targets)

        num_samples = len(predictions)
        target_mat = _targets_to_mat(targets, predictions.shape[1])

        filtered_preds = self.prediction_filter.filter(predictions, 'vec')
        n_correct_predictions = np.multiply(filtered_preds, target_mat).sum(1)  # shape (N,)
        n_predictions = filtered_preds.sum(1)  # shape (N,)
        n_predictions[n_predictions == 0] = 1  # To avoid zero-division. If n_predictions==0, num should be zero as well.
        self.sample_precision_sum += (n_correct_predictions / n_predictions).sum()
        self.num_sample += num_samples

    def get_report(self, **kwargs):
        return {f'precision_{self.prediction_filter.identifier}': float(self.sample_precision_sum) / self.num_sample if self.num_sample else 0.0}

    def reset(self):
        super(SamplePrecisionEvaluator, self).reset()
        self.num_sample = 0
        self.sample_precision_sum = 0


class SampleRecallEvaluator(Evaluator):
    """
    Recall evaluator for both multi-class and multi-label classification, with average='samples'

    This evaluator demands less memory, by no storing the original predictions and targets.
    """

    def __init__(self, prediction_filter: PredictionFilter):
        super().__init__()
        self.prediction_filter = prediction_filter

    def add_predictions(self, predictions, targets):
        """ Add a batch of predictions for evaluation.
        Args:
            predictions: the model output array. Shape (N, num_class)
            targets: the ground truths. Shape (N, num_class) for multi-label or (N,) for multi-class
        """

        assert len(predictions) == len(targets)

        num_samples = len(predictions)
        target_mat = _targets_to_mat(targets, predictions.shape[1])

        filtered_preds = self.prediction_filter.filter(predictions, 'vec')
        n_correct_predictions = np.multiply(filtered_preds, target_mat).sum(1)  # shape (N,)
        n_gt = target_mat.sum(1)  # shape (N,)
        n_gt[n_gt == 0] = 1  # To avoid zero-division. for sample with zero labels, here we default recall to zero
        self.sample_recall_sum += (n_correct_predictions / n_gt).sum()
        self.num_sample += num_samples

    def get_report(self, **kwargs):
        return {f'recall_{self.prediction_filter.identifier}': float(self.sample_recall_sum) / self.num_sample if self.num_sample else 0.0}

    def reset(self):
        super(SampleRecallEvaluator, self).reset()
        self.num_sample = 0
        self.sample_recall_sum = 0
