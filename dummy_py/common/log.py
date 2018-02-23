from time import time
from typing import Dict, Callable
from collections import defaultdict

from dummy_py.common import Once

__all__ = [
    'TimeReducedLog',
]


class TimeReducedLog:
    """
    normally used by ML loss reduce log
    """

    def __init__(self,
                 print_method: 'Callable' = print,
                 reduce_fns: 'Dict[str, Callable]' = None,
                 default_reduce_fn: 'Callable' = None):
        """
        :param print_method: if you used a 3rd log lib, provide a log method
        :param reduce_fns: initial reduce functions, see add_reduce_fn
        """
        self._print_method = print_method
        self._reduce_fns = {}
        self._items = defaultdict(list)
        self._start_time = None
        self._default_reduce_fn = default_reduce_fn

        if reduce_fns is not None:
            for name, reduce_fn in reduce_fns.items():
                self.add_reduce_fn(name, reduce_fn)

    def add_reduce_fn(self, name: 'str', fn: 'Callable'):
        """
        add a reduce function

        :param name: item name
        :param fn: (cost:float, values: list[any]) -> any
        """
        if name in self._reduce_fns:
            raise RuntimeError('dup name: {}'.format(name))
        if not callable(fn):
            raise RuntimeError('invalid fn of {}: {}'.format(name, fn))
        self._reduce_fns[name] = fn

    def print(self, fmt: 'str', *args, **kwargs):
        """
        shortcut for print_method(fmt.format(*args, **kwargs))
        """
        if len(args) > 0 or len(kwargs) > 0:
            fmt = fmt.format(*args, **kwargs)
        return self._print_method(fmt)

    def start(self):
        """
        start time counting
        """
        if self._start_time is not None:
            raise RuntimeError('dup start')
        self._start_time = time()

    def item(self, name: 'str', value):
        """
        provide a value to item list
        :param name: item name
        :param value: item value
        """
        if self._start_time is None:
            raise RuntimeError('not start')
        if self._default_reduce_fn is None and name not in self._reduce_fns:
            raise RuntimeError('invalid name(not in reduce_fns): {}'.format(name))
        self._items[name].append(value)

    def stop(self):
        """
        stop time counting and print log
        """
        if self._start_time is None:
            raise RuntimeError('not start')
        # prepare
        cost = self.current_cost
        items = self._items
        # reset data
        self._items = defaultdict(list)
        self._start_time = None
        # print log
        print_title = Once(lambda: self.print('after {:.3}s:', cost))
        for name, values in items.items():
            if len(values) == 0:
                continue
            print_title()
            reduced = self._reduce_fns.get(name, self._default_reduce_fn)(cost, values)
            self.print('  {}: {}', name, reduced)

    @property
    def current_cost(self):
        """
        :return: current time cost in seconds
        :rtype: float
        """
        if self._start_time is None:
            raise RuntimeError('not start')
        return time() - self._start_time
