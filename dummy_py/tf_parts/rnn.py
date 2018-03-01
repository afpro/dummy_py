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
import tensorflow as tf

from dummy_py.tf_utils.name_scope import NameScope
from ._type_hint import *

__all__ = [
    'LSTM',
    'GRU',
    'MGU',
]


def _dense(x, w, b):
    if isinstance(x, tuple):
        x = tf.concat(x, axis=-1)
    return tf.matmul(x, w) + b


class Base:
    def __init__(self, input_size: 'int', output_size: 'int',
                 name: 'str' = None,
                 dtype: 'tf.DType' = tf.float32):
        self._input_size = input_size
        self._output_size = output_size
        self._dtype = dtype

        with NameScope(name, self.type_name) as ns:
            self._build_ns = ns.scope_name
            self._build(ns)

    @property
    def type_name(self) -> 'str':
        raise NotImplementedError

    @property
    def input_size(self) -> 'int':
        return self._input_size

    @property
    def output_size(self) -> 'int':
        return self._output_size

    @property
    def dtype(self) -> 'tf.DType':
        return self._dtype

    @property
    def _call_default_ns(self) -> 'str':
        return '{}call'.format(self._build_ns)

    def state_size(self, batch: 'int' = None):
        return batch, self.output_size

    def _build(self, ns: 'NameScope'):
        raise NotImplementedError


class LSTM(Base):
    @property
    def type_name(self) -> 'str':
        return 'lstm'

    def _build(self, ns: 'NameScope'):
        with ns.var_scope():
            self._w_fioc = tf.get_variable('w_fioc', dtype=self.dtype,
                                           shape=(self.input_size + self.output_size, self.output_size * 4))
            self._b_fioc = tf.get_variable('b_fioc', shape=(self.output_size * 4,), dtype=self.dtype)

    def __call__(self, x: 'tf_input', h: 'tf_input', c: 'tf_input',
                 name: 'str' = None) -> 'typing.Tuple[tf.Tensor, tf.Tensor]':
        with NameScope(name, self._call_default_ns, [x, h, c, self._w_fioc, self._b_fioc]):
            x = tf.convert_to_tensor(x, self.dtype)
            h = tf.convert_to_tensor(h, self.dtype)
            c = tf.convert_to_tensor(c, self.dtype)
            t_fioc = _dense((x, h), self._w_fioc, self._b_fioc)
            ft, it, ot, ct_hat = tf.split(t_fioc, 4, axis=-1)
            ft = tf.sigmoid(ft)
            it = tf.sigmoid(it)
            ot = tf.sigmoid(ot)
            ct_hat = tf.tanh(ct_hat)
            ct = ft * c + it * ct_hat
            ht = ot * tf.tanh(ct)
        return ht, ct


class GRU(Base):
    @property
    def type_name(self) -> 'str':
        return 'gru'

    def _build(self, ns: 'NameScope'):
        self._w_zr = tf.get_variable('w_zr', dtype=self.dtype,
                                     shape=(self.input_size + self.output_size, self.output_size * 2))
        self._b_zr = tf.get_variable('b_zr', dtype=self.dtype,
                                     shape=(self.output_size * 2,))
        self._w_h = tf.get_variable('w_h', dtype=self.dtype,
                                    shape=(self.input_size + self.output_size, self.output_size))
        self._b_h = tf.get_variable('b_h', dtype=self.dtype,
                                    shape=(self.output_size,))

    def __call__(self, x: 'tf_input', h: 'tf_input', name: 'str' = None) -> 'tf.Tensor':
        with NameScope(name, self._call_default_ns, [x, h, self._w_zr, self._b_zr]):
            x = tf.convert_to_tensor(x, self.dtype)
            h = tf.convert_to_tensor(h, self.dtype)
            zt, rt = tf.split(tf.sigmoid(_dense((x, h), self._w_zr, self._b_zr)),
                              2, axis=-1)
            ht_hat = tf.tanh(_dense((rt * h, x), self._w_h, self._b_h))
            ht = (1 - zt) * h + zt * ht_hat
        return ht


class MGU(Base):
    @property
    def type_name(self) -> 'str':
        return 'mgu'

    def _build(self, ns: 'NameScope'):
        with ns.var_scope():
            self._w_f = tf.get_variable('w_f', dtype=self.dtype,
                                        shape=(self.input_size + self.output_size, self.output_size))
            self._b_f = tf.get_variable('b_f', dtype=self.dtype,
                                        shape=(self.output_size,))
            self._w_h = tf.get_variable('w_h', dtype=self.dtype,
                                        shape=(self.input_size + self.output_size, self.output_size))
            self._b_h = tf.get_variable('b_h', dtype=self.dtype,
                                        shape=(self.output_size,))

    def __call__(self, x: 'tf_input', h: 'tf_input', name: 'str' = None) -> 'tf.Tensor':
        with NameScope(name, self._call_default_ns, [x, h, self._w_f, self._b_f, self._w_h, self._b_h]):
            x = tf.convert_to_tensor(x, self.dtype)
            h = tf.convert_to_tensor(h, self.dtype)
            ft = tf.sigmoid(_dense((h, x), self._w_f, self._b_f))
            ht_hat = tf.tanh(_dense((ft * h, x), self._w_h, self._b_h))
            ht = (1 - ft) * h + ft * ht_hat
        return ht
