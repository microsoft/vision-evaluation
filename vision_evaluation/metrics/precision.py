import sklearn.metrics as sm
from .metrics import Metrics, MetricsRegistry
from .prediction_filters import TopKPredictionFilter, ThresholdPredictionFilter
from .states import AllPredictionsState


@MetricsRegistry.register('precision', ('classification_multiclass', 'classification_multilabel'))
class PrecisionMetrics(Metrics):
    """
    Precision evaluator for both multi-class and multi-label classification
    """

    def __init__(self, context, k=None, threshold=None, average='macro'):
        super().__init__(context)
        assert k or threshold
        if k is not None:
            self.prediction_filter = TopKPredictionFilter(k)
        else:
            self.prediction_filter = ThresholdPredictionFilter(threshold)
        self._average = average
        self._all_predictions = AllPredictionsState(context)

    @property
    def key_name(self):
        return f'precision' + self.prediction_filter.identifier if self.prediction_filter else ''

    def compute(self):
        predictions, targets = self._all_predictions.get()
        predictions = self.prediction_filter.filter(predictions, 'vec')
        return sm.precision_score(targets, predictions, average=self._average)
