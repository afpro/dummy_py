# Copyright 2018, afpro <admin@afpro.net>.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from threading import local
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
    _tls = local()

    @classmethod
    def _stack_list(cls, create=True):
        v = None
        try:
            v = cls._tls.stack
        except AttributeError:
            if create:
                v = []
                cls._tls.stack = v
        return v

    @classmethod
    def _del_stack_list(cls):
        del cls._tls.stack

    def __init__(self, name, default_name=None, values=None):
        super().__init__(name,
                         default_name=default_name,
                         values=[_ for _ in values if _ is not None] if values is not None else None)

    def __enter__(self):
        self._cur_name = super().__enter__()  # type: str
        self._stack_list().append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        v = self._stack_list(create=False)
        if v is None or len(v) == 0 or v[-1] is not self:
            raise RuntimeError('exit with wrong state')
        if len(v) == 1:
            self._del_stack_list()
        else:
            del v[-1]
        return super().__exit__(exc_type, exc_val, exc_tb)

    @property
    def scope_name(self) -> 'str':
        """
        :return: current scope name
        """
        return self._cur_name

    @classmethod
    def current(cls) -> 'NameScope':
        s = cls._stack_list(create=False)
        if s is None or len(s) == 0:
            raise RuntimeError('not within NameScope(...) scope')
        return s[-1]

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

    @classmethod
    def get_variable(cls, name: 'str', **kwargs):
        with cls.current().var_scope():
            return tf.get_variable(name, **kwargs)
