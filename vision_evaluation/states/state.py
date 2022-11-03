import itertools


class StateContext:
    def __init__(self):
        self.states = []

    def add_state(self, state):
        self.states.append(state)

    def merge_states(self):
        # Merge equivalent states
        for s1, s2 in itertools.combinations(self.states, 2):
            if s1 == s2:
                s2.delegate = CopyDelegate(s1)
                s2.parent = s1

        states_with_no_parent = [s for s in self.states if not s.parent]
        for s1 in states_with_no_parent:
            for s2 in self.states:
                if s1 is not s2:
                    merger = MergerRegistry.get(s1.__class__, s2.__class__)
                    if merger:
                        merger(self, s1, s2)


class Delegate:
    def __init__(self):
        self.reset()

    def add_predictions(self, predictions, targets):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def get(self):
        if self._cached_result is None:
            self._cached_result = self.compute()
        return self._cached_result

    def reset(self):
        self._cached_result = None


class CopyDelegate(Delegate):
    def __init__(self, state):
        super().__init__()
        self._state = state

    def add_predictions(self, predictions, targets):
        pass

    def get(self):
        return self._state.get()

    def reset(self):
        pass


class State:
    def __init__(self, context, delegate=None):
        context.add_state(self)
        self._parent = None
        self._delegate = delegate

    @property
    def delegate(self):
        return self._delegate

    @delegate.setter
    def delegate(self, value):
        self._delegate = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    def add_predictions(self, predictions, targets):
        self._delegate.add_predictions(predictions, targets)

    def reset(self):
        self._delegate.reset()


class MergerRegistry:
    _registry = {}

    @classmethod
    def register(cls, state1_class, state2_class):
        def wrapper(f):
            cls._registry[(state1_class, state2_class)] = f
            return f
        return wrapper

    def get(cls, state1_class, state2_class):
        return cls._registry[(state1_class, state2_class)]
