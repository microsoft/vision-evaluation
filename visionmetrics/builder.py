from .evaluators import ClassificationMulticlassEvaluator, ClassificationMultilabelEvaluator, ObjectDetectionEvaluator
from .metrics import MetricsRegistry
from .states import StateContext


class EvaluatorBuilder:
    EVALUATOR_CLASSES = {'classification_multiclass': ClassificationMulticlassEvaluator,
                         'classification_multilabel': ClassificationMultilabelEvaluator,
                         'object_detection': ObjectDetectionEvaluator}

    def __init__(self, task_type):
        if task_type not in ['classification_multiclass', 'classification_multilabel', 'object_detection']:
            raise ValueError

        self._task_type = task_type
        self._metrics = {}
        self._context = StateContext()
        self._created = False

    def add(self, metrics_name, *, custom_key_name=None, **kwargs):
        metrics_class = MetricsRegistry.get(self._task_type, metrics_name)
        metrics = metrics_class(self._context, **kwargs)
        key_name = custom_key_name or metrics.key_name
        if key_name in self._metrics:
            raise ValueError(f"The key name is duplicated: {key_name} in {self._metrics.keys()}")
        self._metrics[key_name] = metrics
        return key_name

    def get_available_metrics_names(self):
        return MetricsRegistry.get_names(self._task_type)

    def create(self):
        if self._created:
            raise RuntimeError("This builder object cannot create multiple evaluators. Please re-create the builder.")
        self._created = True

        evaluator_class = self.EVALUATOR_CLASSES[self._task_type]
        return evaluator_class(self._context, self._metrics.values())
