import collections


class Metrics:
    def __init__(self, context):
        pass

    @property
    def key_name(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def add_predictions(self, predictions, targets):
        pass


class MetricsRegistry:
    _registry = collections.defaultdict(dict)

    @classmethod
    def register(cls, name, task_type):
        def wrapper(c):
            task_type_list = [task_type] if not isinstance(task_type, (tuple, list)) else task_type
            for t in task_type_list:
                cls._registry[t][name] = c
            return c
        return wrapper

    @classmethod
    def get(cls, task_type, name):
        return cls._registry[task_type][name]

    @classmethod
    def get_names(cls, task_type):
        return cls._registry[task_type].keys()


def create_metrics(task_type, metrics_name, **kwargs):
    metrics_class = MetricsRegistry.get(task_type, metrics_name)
    return metrics_class(**kwargs)


def get_metrics_names(task_type):
    return MetricsRegistry.get_names(task_type)
