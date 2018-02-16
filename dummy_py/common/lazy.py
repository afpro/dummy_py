import numpy as np

from .sequence import Sequence

__all__ = [
    'Lazy',
    'LazyProperty',
    'LazyMapList',
]


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
        v = self._method(instance)
        setattr(instance, self._name, v)
        return v


class _FakeLazy:
    def __init__(self, v):
        self.value = v


class LazyMapList:
    """
    lazy map source list to target list with map_fn

    >>> a = [1, 2, 3]
    >>> b = LazyMapList(a, lambda _: _ * 2)
    >>> len(b)
    3
    >>> b[0]
    2
    """

    def __init__(self, source_data, map_fn):
        self._data = Sequence(source_data).map(lambda _: Lazy(lambda: map_fn(_))).to_list()

    def __len__(self):
        return self.len()

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        self._data[key] = _FakeLazy(value)

    def len(self):
        return len(self._data)

    def get(self, index):
        return self._data[index].value

    def shuffle(self):
        np.random.shuffle(self._data)

    def random_item(self):
        return self.get(np.random.randint(self.len()))
