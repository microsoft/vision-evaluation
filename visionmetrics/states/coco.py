from .state import State, MergerRegistry
from .all_predictions import AllPredictionsState


class CocoMeanAveragePrecisionState(State):
    def __init__(self, context, ious):
        self._all_predictions_state = AllPredictionsState(context)
        delegate = PycocotoolsDelegate(self._all_predictions_state)
        self._ious = ious
        super().__init__(context, delegate)

    def set_delegate(self, delegate):
        self._delegate = delegate

    @property
    def ious(self):
        return self._ious

    def add_predictions(self, predictions, targets):
        pass

    def compute(self):
        mean_aps = self._delegate(self.ious)
        return mean_aps[0] if len(mean_aps) == 1 else mean_aps


class PycocotoolsDelegate:
    def __init__(self, all_predictions_state):
        self._all_predictions_state = all_predictions_state

    def __call__(self, ious):
        predictions, targets = self._all_predictions_state.get()

        # TODO
        return [0, 1, 2]


class FromOtherStateDelegate:
    def __init__(self, state):
        self._state = state

    def __call__(self, ious):
        s = self._state.get()
        return [s[iou] for iou in ious]


@MergerRegistry.register(CocoMeanAveragePrecisionState, CocoMeanAveragePrecisionState)
def merge_coco_mean_average_state(context, state1, state2):
    new_state = CocoMeanAveragePrecisionState(context, state1.ious + state2.ious)
    from_other_state_delegate = FromOtherStateDelegate(new_state)
    state1.delegate = from_other_state_delegate
    state2.delegate = from_other_state_delegate
    state1.parent = new_state
    state2.parent = new_state
