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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMCell, MultiRNNCell

from dummy_py.tf_parts.type_hint import *

__all__ = [
    'rnn_cell',
    'broadcast_matmul',
    'embedding',
    'cross_entropy',
    'attention',
    'mask_logits',
    'softmax',
    'Loop'
]


def rnn_cell(hiddens, cell_fn: 'typing.Type[RNNCell]' = LSTMCell):
    """
    create multi layer rnn cell

    :param hiddens: hidden sizes
    :param cell_fn: rnn cell type or rnn cell factory
    :return:
    """
    hiddens = [int(_) for _ in hiddens]
    assert len(hiddens) > 0
    if len(hiddens) == 1:
        return cell_fn(hiddens[0])
    return MultiRNNCell([cell_fn(hidden) for hidden in hiddens])


def broadcast_matmul(x: 'tf.Tensor', y: 'tf.Tensor'):
    """
    broadcast matmul x and y
    for example, x with shape [A, B, C, D, E, F], y with shape [F, G]
    result is a matrix with shape [A, B, C, D, E, G]

    :param x: mat x
    :param y: mat y
    :return: matmul(x, y)
    """
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    v = tf.matmul(tf.reshape(x, (-1, x_shape[-1])), y)
    return tf.reshape(v, shape=tf.concat((x_shape[:-1], y_shape[-1:]), axis=0))


def embedding(ids, in_size, out_size, name=None, reuse=None):
    """
    create a embedding lookup param and do embedding lookup
    :param ids: embedding lookup id
    :param in_size: id range
    :param out_size: output vector size
    :param name: variable scope name
    :param reuse: reuse variable
    :return: embedding lookup result
    """
    with tf.variable_scope(name, default_name='embedding', values=[ids], reuse=reuse):
        return tf.nn.embedding_lookup(
            params=tf.get_variable(name='param',
                                   dtype=tf.float32,
                                   shape=(in_size, out_size)),
            ids=ids,
            name='output')


def cross_entropy(logits: 'tf.Tensor', labels: 'tf.Tensor', seq_len: 'tf.Tensor', name=None) -> 'tf.Tensor':
    """
    calculate cross entropy with sequence length support

    :param logits: logits, as [Batch, TimeStep, NClass]
    :param labels: labels, as [Batch, TimeStep]
    :param seq_len: sequence length, as [Batch]
    :param name: operator name
    :return: tuple with 1.cross entropy, same shape like logits, pad zero, 2. scalar sum, 3. scalar mean
    """
    with tf.name_scope(name, default_name='cross_entropy', values=[logits, labels, seq_len]):
        """ return sum entropy """
        return CrossEntropyLoop.run(
            logits=logits,
            labels=labels,
            seq_len=seq_len)


def attention(query, w, key):
    """
    :param query: [seq_len, QuerySize]
    :param w: [KeySize, QuerySize]
    :param key: [N, KeySize]
    :return: [N, seq_len]
    """
    attn_logits = tf.matmul(tf.matmul(key, w), query, transpose_b=True)
    attn_vec = tf.nn.softmax(attn_logits)
    return attn_vec


def mask_logits(logits: 'tf_input', mask: 'tf_input', dtype: 'tf.DType' = None, name: 'str' = None) -> 'tf.Tensor':
    """
    :param logits:  input logits shape=[Batch, SequenceLen, NClass]
    :param mask: mask, same shape as logits (or with broadcast)
    :param dtype: target dtype
    :param name: node name
    :return: masked logits
    """
    with tf.name_scope(name, default_name='mask_logits'):
        logits = tf.convert_to_tensor(logits, dtype=dtype)
        mask = tf.convert_to_tensor(mask, dtype=dtype)

        shift = tf.reduce_min(tf.multiply(logits, mask), axis=-1, keep_dims=True)
        masked = tf.multiply(logits - shift, mask)
        with_offset = masked - (1 - mask) * 1e6

    return with_offset


def softmax(a: 'np.ndarray'):
    """
    calculate softmax with last dim

    :param a: input data
    :return: softmax result
    """
    e = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)


class DictToProperty:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, item):
        if self._data is None:
            raise KeyError
        if not isinstance(self._data, dict):
            raise KeyError
        if item not in self._data:
            raise KeyError
        return self._data[item]

    def __getattr__(self, item):
        if isinstance(self._data, dict):
            if item not in self._data:
                raise KeyError
            return self._data[item]

        if hasattr(self._data, item):
            return getattr(self._data, item)

        raise KeyError

    @staticmethod
    def wrap(data):
        if not isinstance(data, DictToProperty):
            data = DictToProperty(data)
        return data


class Loop:
    """
    abstraction for tf.while_loop

    args and extra could be used as dict or visit data as property
    """

    @classmethod
    def loop_vars(cls, extra):
        """
        :param extra: extra from invoke method
        :return: a dict[str, tf.Tensor]
        """
        raise NotImplementedError

    @classmethod
    def shape_invariants(cls, extra):
        return None

    @classmethod
    def loop_cond(cls, args, extra):
        """
        loop cond, quit loop with return false tensor

        :param args: dict[str, tf.Tensor] from loop_vars
        :param extra: extra from invoke method
        :return: tf.Tensor with bool type, () shape, quit loop with false value
        """
        raise NotImplementedError

    @classmethod
    def loop_body(cls, args, extra):
        """
        loop body

        :param args: dict[str, tf.Tensor] from loop_vars
        :param extra: extra from invoke method
        :return: dict[str, tf.Tensor], each entry in this dict will reflect on args
        """
        raise NotImplementedError

    @classmethod
    def reduce_result(cls, args, extra):
        """
        after loop finished, reduce args and extras as result for invoke method

        :param args: dict[str, tf.Tensor] from loop_vars
        :param extra: extra from invoke method
        :return: anything you want invoke method return
        """
        return args

    @classmethod
    def invoke(cls, extra=None, name=None, debug=False):
        """
        :param extra: a dict with extra data for while loop
        :param name: operator name
        :param debug: with debug on, while loop parallel is disabled
        :return: result from reduce_result
        """

        extra = DictToProperty.wrap(extra)

        loop_var_dict = cls.loop_vars(extra)
        loop_var_names = list(loop_var_dict.keys())
        loop_var_tensors = [loop_var_dict[name] for name in loop_var_names]

        def loop_vars_to_args(loop_vars):
            return {
                loop_var_names[i]: loop_vars[i]
                for i in range(len(loop_var_names))
            }

        def mock_cond(*args):
            args = loop_vars_to_args(args)
            return cls.loop_cond(DictToProperty.wrap(args), extra)

        def mock_body(*args):
            args = loop_vars_to_args(args)
            result = cls.loop_body(DictToProperty.wrap(args), extra)
            for key, value in result.items():
                args[key] = value
            return [args[_] for _ in loop_var_names]

        si = cls.shape_invariants(extra)
        if isinstance(si, dict):
            si = [si[_] for _ in loop_var_names]

        result_vars = tf.while_loop(
            cond=mock_cond,
            body=mock_body,
            loop_vars=loop_var_tensors,
            shape_invariants=si,
            parallel_iterations=1 if debug else 10,
            name=name)
        return cls.reduce_result(DictToProperty.wrap(loop_vars_to_args(result_vars)), extra)


class CrossEntropyLoop(Loop):
    """
    a loop for calculate cross entropy. could be a tutorial for Loop class
    """

    @classmethod
    def loop_vars(cls, extra):
        return {
            'i': tf.constant(0, tf.int32),
            'cent': tf.TensorArray(extra.logits.dtype, extra.batch_size),
            'sum': tf.constant(0, tf.float32),
            'count': tf.constant(0, tf.float32),
        }

    @classmethod
    def loop_cond(cls, args, extra):
        return args.i < extra.batch_size

    @classmethod
    def loop_body(cls, args, extra):
        length = extra.seq_len[args.i]
        row = extra.logits[args.i, :length]
        label = extra.labels[args.i, :length]
        log_sum_exp = tf.reduce_logsumexp(row, 1)

        class InnerLoop(Loop):
            @classmethod
            def loop_vars(cls, _extra):
                return {
                    'i': tf.constant(0, tf.int32),
                    'cent': tf.TensorArray(extra.logits.dtype, extra.max_seq_len),
                    'sum': tf.constant(0, tf.float32),
                    'count': tf.constant(0, tf.float32),
                }

            @classmethod
            def loop_cond(cls, _args, _extra):
                return _args.i < extra.max_seq_len

            @classmethod
            def loop_body(cls, _args, _extra):
                def true_fn():
                    _cent = log_sum_exp[_args.i] - row[_args.i, label[_args.i]]
                    return _cent, _args.sum + _cent, _args.count + 1

                def false_fn():
                    return tf.constant(0, tf.float32), _args.sum, _args.count

                cent, s, c = tf.cond(
                    _args.i < length,
                    true_fn=true_fn,
                    false_fn=false_fn)

                return {
                    'i': _args.i + 1,
                    'cent': _args.cent.write(_args.i, cent),
                    'sum': s,
                    'count': c,
                }

            @classmethod
            def reduce_result(cls, _args, _extra):
                return _args.cent.stack(), _args.sum, _args.count

        inner = InnerLoop.invoke()
        return {
            'i': args.i + 1,
            'cent': args.cent.write(args.i, inner[0]),
            'sum': args.sum + inner[1],
            'count': args.count + inner[2],
        }

    @classmethod
    def reduce_result(cls, args, extra):
        return args.cent.stack(), args.sum, args.sum / (args.count + 1e-9)

    @classmethod
    def run(cls, logits, labels, seq_len, batch_size=None):
        return cls.invoke({
            'logits': logits,
            'labels': labels,
            'seq_len': seq_len,
            'batch_size': batch_size or tf.shape(seq_len)[0],
            'max_seq_len': tf.shape(logits)[1],
        })

    @classmethod
    def test(cls):
        # random logits
        logits = np.random.rand(3, 4, 10).astype(np.float32)
        labels = np.random.randint(0, 10, size=[3, 4], dtype=np.int32)
        seq_len = np.random.randint(2, 5, size=[3], dtype=np.int32)

        # with tf
        with tf.Graph().as_default(), tf.Session() as session:
            s = cls.run(tf.constant(logits), tf.constant(labels), tf.constant(seq_len))
            s = session.run(s)  # type: np.ndarray
        print(s)

        # with np
        v = np.zeros(shape=s[0].shape, dtype=s[1].dtype)
        for batch in range(3):
            for t in range(seq_len[batch]):
                label = labels[batch, t]
                logit = logits[batch, t]
                e = np.exp(logit)
                p = e[label] / np.sum(e)
                cent = - np.log(p)
                v[batch, t] = cent
        print(v)

        # check
        assert np.abs(np.sum(s[0]) - np.sum(v)) < 1e-3
