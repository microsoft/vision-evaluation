import numpy as np
import sklearn.metrics as sm

from .evaluators import MemorizingEverythingEvaluator, PrecisionRecallCurveMixin
from vision_evaluation.prediction_filters import TopKPredictionFilter


class RetrievalEvaluator(MemorizingEverythingEvaluator):
    def add_predictions(self, predictions, targets):
        """ Add a batch of predictions for evaluation.
        Args:
            predictions: the model output array. Shape (N, M) where N is the number of queries and M is the number of items in the dataset
            targets: the ground truths. Shape (N, M) where N is the number of queries and M is the number of items in the dataset
        """
        super(RetrievalEvaluator, self).add_predictions(predictions, targets)


class PrecisionAtKEvaluator(RetrievalEvaluator):
    """
    Precision evaluator for both multi-class and multi-label classification
    """

    def __init__(self, k):
        super(PrecisionAtKEvaluator, self).__init__(TopKPredictionFilter(k))
        self.k = k

    def _get_id(self):
        return f'precision_at_{self.k}'

    def calculate_score(self, average='macro', filter_out_zero_tgt=False):
        return super(PrecisionAtKEvaluator, self).calculate_score(average=average, filter_out_zero_tgt=filter_out_zero_tgt)

    def _calculate(self, targets, predictions, average):
        """
        There is a special case that needs to be handled appropriately.
        Due to filtering conditions, there are occasions where only 1 class remains in targets/predictions and sklearn interpretes this as an invalid configuration for multilabel.
        Since when average == 'samples' is only calculated for multilabel, sm.recall_score throws an exception. e.g.:
        predictions = np.array([[True]])
        targets = np.array([[True]])
        recall_score(targets, predictions, average='samples') throws the following error:
        ValueError: Samplewise metrics are not available outside of multilabel classification.
        """
        if targets.shape[1] == 1 and average == 'samples':
            targets = np.append(targets, np.zeros((targets.shape[0], 1)), axis=1)
            predictions = np.append(predictions, np.zeros((predictions.shape[0], 1)), axis=1)
        return sm.precision_score(targets, predictions, average=average)


class RecallAtKEvaluator(RetrievalEvaluator):
    """
    Recall evaluator for both multi-class and multi-label classification
    """

    def __init__(self, k):
        super(RecallAtKEvaluator, self).__init__(TopKPredictionFilter(k))
        self.k = k

    def _get_id(self):
        return f'recall_at_{self.k}'

    def _calculate(self, targets, predictions, average):
        """
        There is a special case that needs to be handled appropriately.
        Due to filtering conditions, there are occasions where only 1 class remains in targets/predictions and sklearn interpretes this as an invalid configuration for multilabel.
        Since when average == 'samples' is only calculated for multilabel, sm.recall_score throws an exception. e.g.:
        predictions = np.array([[True]])
        targets = np.array([[True]])
        recall_score(targets, predictions, average='samples') throws the following error:
        ValueError: Samplewise metrics are not available outside of multilabel classification.
        """
        if targets.shape[1] == 1 and average == 'samples':
            targets = np.append(targets, np.zeros((targets.shape[0], 1)), axis=1)
            predictions = np.append(predictions, np.zeros((predictions.shape[0], 1)), axis=1)
        return sm.recall_score(targets, predictions, average=average)


class MeanAveragePrecisionAtK(RetrievalEvaluator):
    """
    MeanAveragePrecision @ K as defined here:
    https://stackoverflow.com/questions/54966320/mapk-computation
    https://medium.com/@misty.mok/how-mean-average-precision-at-k-map-k-can-be-more-useful-than-other-evaluation-metrics-6881e0ee21a9
    MAP@K is the average of AveP(K) over all queries. Hence it uses average="samples".
    AveP(K) =  Num / Den, where:
    Num = sum_i^k P(i) * rel(i), where:
    P(i) is the Precision @ i
    rel(i) is an indicator function where rel(i) = 1 if position i is relevant, and rel(i) = 0 otherwise.
    Den = min(K, number of relevant images)
    """

    def __init__(self, k):
        super(MeanAveragePrecisionAtK, self).__init__()
        self.k = k

    def _get_id(self):
        return f'map_at_{self.k}'

    def calculate_score(self, average='samples'):
        assert average == 'samples'
        if self.k == 0:
            return 0.0
        return self._calculate()

    def _calculate(self):
        assert self.all_predictions.shape == self.all_targets.shape
        if self.all_predictions.size == 0:
            return 0.0
        return np.mean([self._average_precision_at_k(preds, targets) for preds, targets in zip(self.all_predictions, self.all_targets)])

    def _average_precision_at_k(self, predictions, targets):
        total_pos_gt = np.sum(targets)
        if total_pos_gt == 0:
            return 0.0
        rank = min(self.k, len(predictions))
        top_k_pred_indices = np.argpartition(-predictions, rank - 1)[:rank]
        # sort score, gt by top_k
        targets = targets[top_k_pred_indices]
        sum = 0.0
        num_hits = 0.0
        for idx, tgt in enumerate(targets):
            if tgt:
                num_hits += 1.0
                sum += num_hits / (idx + 1.0)
        return sum / min(rank, total_pos_gt)


class PrecisionRecallCurveNPointsEvaluator(PrecisionRecallCurveMixin, RetrievalEvaluator):
    """
    N-point interpolatedprecision-recall curve, averaged over samples
    """

    def calculate_score(self, average='samples', filter_out_zero_tgt=False):
        assert average == 'samples'
        return super(PrecisionRecallCurveNPointsEvaluator, self).calculate_score(average=average, filter_out_zero_tgt=filter_out_zero_tgt)

    def _calculate(self, targets, predictions, average):
        assert average == 'samples'
        n_samples = predictions.shape[0]
        recall_thresholds = np.linspace(1, 0, self.n_points, endpoint=True).tolist()
        precision_averaged = np.zeros(self.n_points)
        for i in range(n_samples):
            precision_interp = self._calc_precision_recall_interp(predictions[i, :], targets[i, :], recall_thresholds)
            precision_averaged += precision_interp
        precision_averaged /= n_samples
        return precision_averaged

    def _get_id(self):
        return f'PR_Curve_{self.n_points}_point_interp'
