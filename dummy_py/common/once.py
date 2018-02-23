from typing import Callable

__all__ = [
    'Once',
]


class Once:
    def __init__(self, method: 'Callable'):
        self._method = method
        self._invoked = False
        self._return_value = None

    def __call__(self, *args, **kwargs):
        if not self._invoked:
            self._return_value = self._method(*args, **kwargs)
            self._invoked = True
            self._method = None
        return self._return_value
