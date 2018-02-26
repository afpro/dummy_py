from typing import TYPE_CHECKING, Union
import tensorflow as tf

__all__ = [
    'NameScope',
    'var_scope_or_name',
]

if TYPE_CHECKING:
    var_scope_or_name = Union[tf.VariableScope, str]


class NameScope(tf.name_scope):
    """
    simple tf.name_scope wrapper, for additional variable scope, support None in values
    """

    def __init__(self, name, default_name=None, values=None):
        super().__init__(name,
                         default_name=default_name,
                         values=[_ for _ in values if _ is not None] if values is not None else None)

    def __enter__(self):
        self._cur_name = super().__enter__()  # type: str
        return self

    @property
    def scope_name(self) -> 'str':
        """
        :return: current scope name
        """
        return self._cur_name

    def var_scope(self, scope_or_name: 'var_scope_or_name' = None, **kwargs) -> 'tf.variable_scope':
        """
        create variable scope
        :param scope_or_name: desired scope name, None for current name scope
        :param kwargs: param to tf.variable_scope
        :return: created variable scope
        """
        if scope_or_name is None:
            scope_or_name = self._cur_name.strip('/')
        return tf.variable_scope(scope_or_name, **kwargs)

    @staticmethod
    def create_name_scope_fn(default_name_pattern: 'str' = None):
        """
        create a function for NameScope creation
        :param default_name_pattern: for example 'my_prefix_{}'
        :return: function: (str, str, optional[list]) -> NameScope
        """

        def inner(name, default_name=None, values=None):
            if default_name_pattern is not None:
                default_name = default_name_pattern.format(default_name)
            return NameScope(name, default_name, values)

        return inner
