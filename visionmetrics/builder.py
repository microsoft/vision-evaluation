from .evaluators import ClassificationMulticlassEvaluator, ClassificationMultilabelEvaluator, ObjectDetectionEvaluator
from .metrics import create_metrics, get_metrics_names


class EvaluatorBuilder:
    EVALUATOR_CLASSES = {'classification_multiclass': ClassificationMulticlassEvaluator,
                         'classification_multilabel': ClassificationMultilabelEvaluator,
                         'object_detection': ObjectDetectionEvaluator}

    def __init__(self, task_type):
        if task_type not in ['classification_multiclass', 'classification_multilabel', 'object_detection']:
            raise ValueError

        self._task_type = task_type
        self._metrics = {}

    def add(self, metrics_name, *, custom_key_name=None, **kwargs):
        metrics = create_metrics(self._task_type, metrics_name, **kwargs)
        key_name = custom_key_name or metrics.key_name
        if key_name in self._metrics:
            raise ValueError(f"The key name is duplicated: {key_name} in {self._metrics.keys()}")
        self._metrics[key_name] = metrics

    def get_available_metrics_names(self):
        return get_metrics_names(self._task_type)

    def create(self):
        evaluator_class = self.EVALUATOR_CLASSES[self._task_type]
        return evaluator_class(self._metrics)
