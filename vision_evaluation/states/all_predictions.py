import numpy as np
from .state import Delegate, State


class AllPredictionsState(State):
    def __init__(self, context):
        super().__init__(context, AllPredictionsDelegate())


class AllPredictionsDelegate(Delegate):
    def add_predictions(self, predictions, targets):
        self._predictions = np.append(self._predictions, predictions, axis=0) if self._predictions is not None else predictions.copy()
        self._targets = np.append(self._targets, targets, axis=0) if self._targets is not None else targets.copy()

    def compute(self):
        return self._predictions, self._targets

    def reset(self):
        self._predictions = None
        self._targets = None
