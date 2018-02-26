from typing import TYPE_CHECKING, Union, Tuple

import tensorflow as tf
import numpy as np
from dummy_py.tf_utils.name_scope import NameScope

__all__ = [
    'LSTM',
    'GRU',
    'MGU',
]

name_scope = NameScope.create_name_scope_fn('dummy_py_rnn_{}')

if TYPE_CHECKING:
    tf_input = Union[np.ndarray, tf.Tensor]


class Base:
    def __init__(self, input_size: 'int', output_size: 'int',
                 name: 'str' = None,
                 dtype: 'tf.DType' = tf.float32):
        self._input_size = input_size
        self._output_size = output_size
        self._dtype = dtype

        with name_scope(name, self.type_name) as ns:
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
                 name: 'str' = None) -> 'Tuple[tf.Tensor, tf.Tensor]':
        with name_scope(name, 'lstm_call', [x, h, c, self._w_fioc, self._b_fioc]):
            x = tf.convert_to_tensor(x, self.dtype)
            h = tf.convert_to_tensor(h, self.dtype)
            c = tf.convert_to_tensor(c, self.dtype)
            t_fioc = tf.matmul(tf.concat((x, h), axis=-1), self._w_fioc) + self._b_fioc
            ft, it, ot, ct_hat_input = tf.split(t_fioc, 4, axis=-1)
            ct_hat = tf.tanh(ct_hat_input)
            ct = ft * c + it * ct_hat
            ht = ot * tf.tanh(ct)
        return ht, ct


class GRU(Base):
    @property
    def type_name(self) -> 'str':
        return 'gru'

    def _build(self, ns: 'NameScope'):
        self._w_zrh = tf.get_variable('w_zrh', dtype=self.dtype,
                                      shape=(self.input_size + self.output_size, self.output_size * 3))
        self._b_zrh = tf.get_variable('b_zrh', dtype=self.dtype,
                                      shape=(self.output_size * 3,))

    def __call__(self, x: 'tf_input', h: 'tf_input', name: 'str' = None) -> 'tf.Tensor':
        with name_scope(name, 'gru_call', [x, h, self._w_zrh, self._b_zrh]):
            x = tf.convert_to_tensor(x, self.dtype)
            h = tf.convert_to_tensor(h, self.dtype)
            t_zrh = tf.matmul(tf.concat((x, h), axis=-1), self._w_zrh) + self._b_zrh
            zt, rt, ht_hat_input = tf.split(t_zrh, 3, axis=-1)
            ht_hat = tf.tanh(ht_hat_input)
            ht = (1 - zt) * h + zt * ht_hat
        return ht


class MGU(Base):
    @property
    def type_name(self) -> 'str':
        return 'mgu'

    def _build(self, ns: 'NameScope'):
        with ns.var_scope():
            self._w_fh = tf.get_variable('w_fh', dtype=self.dtype,
                                         shape=(self.input_size + self.output_size, self.output_size * 2))
            self._b_fh = tf.get_variable('b_fh', dtype=self.dtype,
                                         shape=(self.output_size * 2,))

    def __call__(self, x: 'tf_input', h: 'tf_input', name: 'str' = None) -> 'tf.Tensor':
        with name_scope(name, 'mgu_call', [x, h, self._w_fh, self._b_fh]):
            x = tf.convert_to_tensor(x, self.dtype)
            h = tf.convert_to_tensor(h, self.dtype)
            t_fh = tf.matmul(tf.concat((x, h), axis=-1), self._w_fh) + self._b_fh
            ft, ht_hat_input = tf.split(t_fh, 2, axis=-1)
            ht_hat = tf.tanh(ht_hat_input)
            ht = (1 - ft) * h + ft * ht_hat
        return ht
