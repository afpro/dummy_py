class Lazy:
    """
    lazy wrapper

    >>> x = 10
    >>> v = Lazy(lambda: x + 1)
    >>> v.value
    11
    """
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
    """
    lazy property

    >>> class Test:
    >>>     def __init__(self, x):
    >>>         self._x = x
    >>>     @LazyProperty
    >>>     def p(self):
    >>>         return self._x * 2
    >>> test = Test(10)
    >>> test.p
    20
    """
    def __init__(self, method):
        self._method = method
        self._name = self._method.__name__

    def __get__(self, instance, owner):
        if hasattr(instance, self._name):
            return getattr(instance, self._name)

        v = self._method(instance)
        setattr(instance, self._name, v)
        return v
