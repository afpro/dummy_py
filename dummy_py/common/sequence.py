__all__ = [
    'Sequence',
]


class Sequence:
    """
    sequence process, avoid ugly ')))))' in source code
    """

    def __init__(self, iterable, reusable=False):
        """
        :param iterable: data source
        :param reusable: if data source could be iterate multi times
        """
        if iterable is None:
            raise RuntimeError('iterable is None')
        self._iter = iterable
        self._reusable = reusable

    def map(self, fn):
        """
        :param fn: map function (item)->item
        :return: new Sequence
        """
        return Sequence(map(fn, self._iter), self._reusable)

    def filter(self, fn):
        """
        :param fn: filter function (item)->bool
        :return: new Sequence
        """
        return Sequence(filter(fn, self._iter), self._reusable)

    def to(self, wrapper=None):
        """
        take all items to collection or something like that
        :param wrapper: could be reduce function or list/tuple etc
        :return: reduced result
        """
        if self._iter is None:
            raise RuntimeError('result already been taken')
        v = self._iter
        if not self._reusable:
            self._iter = None
        if wrapper is not None:
            v = wrapper(v)
        return v

    def to_iterator(self):
        return self.to()

    def to_list(self):
        return self.to(list)

    def to_tuple(self):
        return self.to(tuple)

    def to_dict(self):
        return self.to(dict)
