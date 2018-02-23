from queue import Queue

__all__ = [
    'Sequence',
]


class _IterableFactory:
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        return self._fn(*self._args, **self._kwargs).__iter__()


class _CachedIterable:
    def __init__(self, iterable, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._cache = []
        self._iter = iterable.__iter__()

    def __iter__(self):
        for v in self._cache:
            yield v

        if self._iter is not None:
            while True:
                try:
                    v = next(self._iter)
                except StopIteration:
                    self._iter = None
                    break
                self._cache.append(v)
                yield v


def _seq_of(fn):
    return Sequence(_IterableFactory(fn))


class Sequence:
    """
    sequence process, avoid ugly ')))))' in source code
    """

    def __init__(self, iterable):
        """
        :param iterable: data source
        """
        if iterable is None:
            raise RuntimeError('iterable is None')
        self._iter = iterable

    def __iter__(self):
        return self._iter.__iter__()

    def cached(self):
        if isinstance(self._iter, (list, tuple)):
            return self
        return Sequence(_CachedIterable(self))

    def map(self, fn):
        """
        :param fn: map function (item)->item
        :return: new Sequence
        """
        return _seq_of(lambda: map(fn, self))

    def filter(self, fn):
        """
        :param fn: filter function (item)->bool
        :return: new Sequence
        """
        return _seq_of(lambda: filter(fn, self))

    def flat_map(self, fn=None):
        def inner():
            for v in self:
                if fn is not None:
                    v = fn(v)
                yield from v

        return _seq_of(inner)

    def strip(self):
        """
        shortcut for strip string items
        """
        return self.map(lambda _: _.strip())

    def first(self, default=None):
        for v in self:
            return v
        return default

    def take(self, n: 'int'):
        def inner():
            i = 0
            for v in self:
                if i >= n:
                    break
                yield v
                i += 1

        return _seq_of(inner)

    def drop(self, n: 'int'):
        if n <= 0:
            return self

        def inner():
            q = Queue(maxsize=n + 1)
            for v in self:
                q.put_nowait(v)
                if q.qsize() > n:
                    yield q.get()

        return _seq_of(inner)

    def for_each(self, fn):
        """
        shortcut for for .. in ...: fn(...)
        """
        for v in self:
            fn(v)

    def to(self, wrapper):
        """
        take all items to collection or something like that
        :param wrapper: could be reduce function or list/tuple etc
        :return: reduced result
        """
        if wrapper is None:
            raise RuntimeError('wrapper is None')
        return wrapper(self)

    def to_list(self):
        return self.to(list)

    def to_tuple(self):
        return self.to(tuple)

    def to_dict(self):
        return self.to(dict)

    def sum(self, **kwargs):
        return self.to(lambda _: sum(_, **kwargs))

    def min(self, **kwargs):
        return self.to(lambda _: max(_, **kwargs))

    def max(self, **kwargs):
        return self.to(lambda _: max(_, **kwargs))
