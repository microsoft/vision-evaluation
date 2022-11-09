import collections


class Metrics:
    def __init__(self, context):
        self.reset()

    @property
    def key_name(self):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def add_predictions(self, predictions, targets):
        pass

    def reset(self):
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
        return list(cls._registry[task_type].keys())
