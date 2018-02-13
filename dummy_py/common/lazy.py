class Lazy:
    def __init__(self, fn):
        assert callable(fn)
        self._fn = fn
        self._v = None

    @property
    def value(self):
        if self._fn is not None:
            self._v = self._fn()
            self._fn = None
        return self._v


class LazyProperty:
    def __init__(self, method):
        self._method = method
        self._name = self._method.__name__

    def __get__(self, instance, owner):
        if hasattr(instance, self._name):
            return getattr(instance, self._name)

        v = self._method(instance)
        setattr(instance, self._name, v)
        return v
