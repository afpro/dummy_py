__all__ = [
    'echo',
    'TakeFn',
    'take_fn',
]


def echo(anything):
    """
    echo, return param
    """
    return anything


class TakeFn:
    """
    take_fn, utils for map

    >>> v = [(0,1), (2,3), (4,5)]
    >>> list(map(take_fn[0], v))
    [0, 2, 4]
    """

    def __getitem__(self, item):
        def inner(_):
            return _[item]

        return inner


take_fn = TakeFn()
