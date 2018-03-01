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
from tensorflow.python.layers.normalization import batch_normalization

from dummy_py.tf_utils import broadcast_matmul
from dummy_py.tf_utils.name_scope import NameScope
from ._type_hint import *

__all__ = [
    'sub_layer',
    'multi_head_attention',
    'encoder',
    'decoder',
    'get_decoder_self_attn_mask',
    'get_pos_encoding',
    'batch_pos_encoding',
]


def _non_or(v, ctor):
    if v is not None:
        return v
    if callable(ctor):
        return ctor()
    return ctor


def _simple_dense(x, in_size, out_size,
                  bias=True,
                  dtype=tf.float32,
                  name=None):
    with NameScope(name, 'dense', [x]) as ns:
        x = broadcast_matmul(x, ns.get_variable('w', shape=(in_size, out_size), dtype=dtype))
        if bias:
            x = x + ns.get_variable('b', shape=(out_size,), dtype=dtype)
        return x


def _reshape_n_head(v, n_head, name=None):
    return tf.concat(tf.split(v, n_head, axis=-1), axis=0, name=name)


def sub_layer(fn, x, *other_inputs, name=None, training=False, extra=None):
    """
    :param fn: sub layer body
    :param x: input x
    :param other_inputs: other input, not used as residual
    :param name: operator name
    :param training: see batch_normalization
    :param extra: extra param for fn
    :return: batch_norm(x + fn(x))
    """
    with NameScope(name, 'sl', [x, *other_inputs]) as scope:
        after_residual = x + fn(x, *other_inputs, **_non_or(extra, dict))
        with scope.var_scope():
            return batch_normalization(after_residual, epsilon=1e-6, training=training)


def multi_head_attention(q: 'tf_input',
                         k: 'tf_input',
                         v: 'tf_input',
                         d_model: 'int',
                         n_head: 'int',
                         attn_mask: 'tf_input' = None,
                         name: 'str' = None,
                         dtype: 'tf.DType' = tf.float32):
    """
    :param q: query    [B, Tq,  d_model]
    :param k: key      [B, Tkv, d_model]
    :param v: value    [B, Tkv, d_model]
    :param d_model: see paper
    :param n_head: see paper
    :param attn_mask: attention mask, [B, Tq, Tkv] (support broadcast), for avoid decoder forward depends
    :param name: operator name
    :param dtype: data dtype, default float32
    :return: see paper
    """
    with NameScope(name, 'mha', [q, k, v]):
        qk_same = q is k
        kv_same = k is v
        q = tf.convert_to_tensor(q, dtype, name='q')
        q_shape = tf.shape(q)
        if qk_same:
            k = q
            k_shape = q_shape
        else:
            k = tf.convert_to_tensor(k, dtype, name='k')
            k_shape = tf.shape(k)
        if kv_same:
            v = k
            v_shape = k_shape
        else:
            v = tf.convert_to_tensor(v, dtype, name='v')
            v_shape = tf.shape(v)
        if attn_mask is not None:
            attn_mask = tf.convert_to_tensor(attn_mask, dtype)

        def control_dependencies():
            yield tf.assert_equal(tf.shape(q_shape), [3], message="query as 3-D tensor")

            if not qk_same:
                yield tf.assert_equal(tf.shape(k_shape), [3], message="key as 3-D tensor")
                yield tf.assert_equal(q_shape[0], k_shape[0], message="query & key has same batch size (dim[0])")

            if not kv_same:
                yield tf.assert_equal(tf.shape(v_shape), [3], message="value as 3-D tensor")
                yield tf.assert_equal(k_shape[:-1], v_shape[:-1],
                                      message="key & value has same batch size(dim[0]) and seq-len(dim[1])")

        with tf.control_dependencies(control_dependencies()):
            q_dense = _simple_dense(q, d_model, d_model * n_head, dtype=dtype, name='qd', bias=False)
            k_dense = _simple_dense(k, d_model, d_model * n_head, dtype=dtype, name='kd', bias=False)
            v_dense = _simple_dense(v, d_model, d_model * n_head, dtype=dtype, name='vd', bias=False)

            q_n_head = _reshape_n_head(q_dense, n_head, name='qn')
            k_n_head = _reshape_n_head(k_dense, n_head, name='kn')
            v_n_head = _reshape_n_head(v_dense, n_head, name='vn')

            qk_n_head = tf.matmul(q_n_head, k_n_head, transpose_b=True, name='qkn')  # [B * n_head, Tq, Tkv]
            attn_n_head = tf.nn.softmax(qk_n_head / (d_model ** 0.5), name='an')
            if attn_mask is not None:
                attn_n_head = tf.split(attn_n_head, n_head, axis=0)
                attn_n_head = [_ * attn_mask for _ in attn_n_head]
                attn_n_head = tf.concat(attn_n_head, axis=0)
                attn_n_head = tf.truediv(attn_n_head, (tf.reduce_sum(attn_n_head, axis=-1, keep_dims=True) + 1e-6),
                                         name='anm')
            attn_v_n_head = tf.matmul(attn_n_head, v_n_head, name='avn')  # [B * n_head, Tq, d_model]
            attn_v = tf.concat(tf.split(attn_v_n_head, n_head, axis=0), axis=-1, name='av')  # [B, Tq, d_model * n_head]
            return _simple_dense(attn_v, d_model * n_head, d_model, dtype=dtype, name='o')


def feed_forward(x: 'tf.Tensor',
                 num_units: 'int' = None,
                 name: 'str' = None, dtype: 'tf.DType' = None):
    """
    :param x: input
    :param num_units: output size
    :param name: operator name
    :param dtype: data type, default to input dtype
    :return: max(0, x*w0+b0)*w1 + b1
    """
    with NameScope(name, 'ff', [x]) as ns:
        x = tf.convert_to_tensor(x, dtype, name='x')
        x = broadcast_matmul(x, ns.get_variable('w1', shape=(num_units, num_units), dtype=dtype))
        x = x + ns.get_variable('b1', shape=(num_units,), dtype=dtype)
        x = tf.nn.relu(x)
        x = broadcast_matmul(x, ns.get_variable('w2', shape=(num_units, num_units), dtype=dtype))
        x = x + ns.get_variable('b2', shape=(num_units,), dtype=dtype)
    return x


def encoder(x: 'tf_input',
            pos_encoding: 'tf_input',
            n_head: 'int',
            n_stack: 'int',
            d_model: 'int',
            training: 'bool' = False,
            context: 'typing.Iterable[tf_input]' = None,
            attn_mask: 'tf_input' = None,
            keep_prob: 'typing.Union[tf_input, float]' = None,
            name: 'str' = None,
            dtype: 'tf.DType' = tf.float32):
    """
    encoder, input size and pos encoding size must be d_model

    :param x: input (eg. word embedding), [Batch, TimeStep, d_model]
    :param pos_encoding: pos encoding, could be learned or sin/cos curve, same shape as input(support broadcast)
    :param n_head: see paper
    :param n_stack: see paper
    :param d_model: see paper
    :param training: see batch_normalization
    :param context: previous step data, from return value
    :param attn_mask: self attention mask (similar to seq_len)
    :param keep_prob: add dropout to each layer, 1 means no drop, 0 means drop all
    :param name: operator name
    :param dtype: data type, default float32
    :return: (encoder result, current context)
    """
    with NameScope(name, 'encoder', [x, pos_encoding, keep_prob] + _non_or(context, list)):
        x = tf.convert_to_tensor(x, dtype)
        pos_encoding = tf.convert_to_tensor(pos_encoding, dtype)

        def control_dependencies():
            x_shape = tf.shape(x)
            x_dims = tf.shape(x_shape)
            yield tf.assert_equal(x_dims, [3])
            yield tf.assert_equal(x_shape[2], d_model)

            pe_shape = tf.shape(pos_encoding)
            pe_dims = tf.shape(pe_shape)
            yield tf.assert_equal(pe_dims, [3])

            yield tf.Assert(tf.logical_or(tf.equal(pe_shape[0], x_shape[0]),
                                          tf.equal(pe_shape[0], 1)),
                            [pe_dims])
            yield tf.Assert(tf.logical_or(tf.equal(pe_shape[1], x_shape[1]),
                                          tf.equal(pe_shape[1], 1)),
                            [pe_dims])
            yield tf.Assert(tf.logical_or(tf.equal(pe_shape[2], x_shape[2]),
                                          tf.equal(pe_shape[2], 1)),
                            [pe_dims])

        with tf.control_dependencies(control_dependencies()):
            v = x + pos_encoding
            v_list = []
            if context is not None:
                context = [tf.convert_to_tensor(_, dtype) for _ in context]
                assert len(context) == n_stack
            for i in range(n_stack):
                v_list.append(v)
                self_attn_q = v
                self_attn_kv = v if context is None else tf.concat((v, context[i]), axis=-1)
                v = sub_layer(multi_head_attention, self_attn_q, self_attn_kv, self_attn_kv, d_model, n_head, attn_mask,
                              name='stack_{}_sa'.format(i),
                              training=training,
                              extra={'dtype': dtype})
                v = sub_layer(feed_forward, v, d_model,
                              name='stack_{}_ff'.format(i),
                              training=training,
                              extra={'dtype': dtype})
                if keep_prob is not None:
                    v = tf.nn.dropout(v, keep_prob)
    return v, v_list


def decoder(x: 'tf_input',
            pos_encoding: 'tf_input',
            encoder_output: 'tf_input',
            n_head: 'int',
            n_stack: 'int',
            d_model: 'int',
            enc_attn_mask: 'tf_input' = None,
            dec_attn_mask: 'tf_input' = None,
            training: 'bool' = False,
            keep_prob: 'typing.Union[tf_input, float]' = None,
            context: 'typing.Iterable[tf_input]' = None,
            name: 'str' = None,
            dtype: 'tf.DType' = tf.float32):
    """
    :param x: input (eg. word embedding), [Batch, TimeStep, d_model]
    :param pos_encoding: pos encoding, could be learned or sin/cos curve, same shape as input(support broadcast)
    :param encoder_output: encoder output, [Batch, EncoderTimeStep, EncoderDModel]
    :param n_head: see paper
    :param n_stack: see paper
    :param d_model: see paper
    :param context: previous step data, from return value
    :param enc_attn_mask: attn mask for encoder output, for variant length encoder output
    :param dec_attn_mask: attn mask for self attention, see multi_head_attention method
    :param training: see batch_normalization
    :param keep_prob: add dropout to each layer, 1 means no drop, 0 means drop all
    :param name: operator name
    :param dtype: data type, default float32
    :return: (decoder result, current context)
    """
    with NameScope(name, 'decoder',
                   [x, pos_encoding, enc_attn_mask, dec_attn_mask, keep_prob] + _non_or(context, list)):
        x = tf.convert_to_tensor(x, dtype)
        pos_encoding = tf.convert_to_tensor(pos_encoding, dtype)
        encoder_output = tf.convert_to_tensor(encoder_output, dtype)

        def control_dependencies():
            x_shape = tf.shape(x)
            x_dims = tf.shape(x_shape)
            yield tf.assert_equal(x_dims, [3])
            yield tf.assert_equal(x_shape[2], d_model)

            pe_shape = tf.shape(pos_encoding)
            pe_dims = tf.shape(pe_shape)
            yield tf.assert_equal(pe_dims, [3])

            yield tf.Assert(tf.logical_or(tf.equal(pe_shape[0], x_shape[0]),
                                          tf.equal(pe_shape[0], 1)),
                            [pe_dims])
            yield tf.Assert(tf.logical_or(tf.equal(pe_shape[1], x_shape[1]),
                                          tf.equal(pe_shape[1], 1)),
                            [pe_dims])
            yield tf.Assert(tf.logical_or(tf.equal(pe_shape[2], x_shape[2]),
                                          tf.equal(pe_shape[2], 1)),
                            [pe_dims])

            eo_shape = tf.shape(encoder_output)
            eo_dims = tf.shape(eo_shape)
            yield tf.equal(eo_dims, 3)
            yield tf.equal(eo_shape[0], x_shape[0])

        if context is not None:
            context = [tf.convert_to_tensor(_, dtype) for _ in context]
            assert len(context) == n_stack
        with tf.control_dependencies(control_dependencies()):
            v = x + pos_encoding
            v_list = []
            for i in range(n_stack):
                v_list.append(v)
                self_attn_q = v
                self_attn_kv = v if context is None else tf.concat((v, context[i]), axis=-1)
                v = sub_layer(multi_head_attention, self_attn_q, self_attn_kv, self_attn_kv, d_model, n_head,
                              name='stack_{}_sa'.format(i),
                              training=training,
                              extra={'attn_mask': dec_attn_mask,
                                     'dtype': dtype})
                v = sub_layer(multi_head_attention, v, encoder_output, encoder_output, d_model, n_head,
                              name='stack_{}_a'.format(i),
                              training=training,
                              extra={'attn_mask': enc_attn_mask,
                                     'dtype': dtype})
                v = sub_layer(feed_forward, v, d_model,
                              name='stack_{}_ff'.format(i),
                              training=training,
                              extra={'dtype': dtype})
                if keep_prob is not None:
                    v = tf.nn.dropout(v, keep_prob)
    return v, v_list


def get_decoder_self_attn_mask(t_q: 'int', dtype=np.float32):
    """
    get decoder self attention mask
    :param t_q: decoder query length
    :param dtype: mask dtype, default float32
    :return: [t_q, t_q] matrix
    """
    return np.transpose(np.triu(np.ones((t_q, t_q), dtype)), (1, 0))


def get_pos_encoding(d_model: 'int', pos: 'int', dtype=np.float32):
    """
    get pos encoding vector
    :param d_model: output vector size
    :param pos: position
    :param dtype: pos encoding dtype, default float32
    :return: pos encoding vector, shape (d_model,)
    """
    v = np.power(np.array(pos / 10000, dtype), np.arange(0, 2, 2 / d_model, dtype))
    v[0::2] = np.sin(v[0::2])
    v[1::2] = np.cos(v[1::2])
    return v


def batch_pos_encoding(seq_len: 'int', d_model: 'int', dtype=np.float32):
    """
    :param seq_len: max sequence len
    :param d_model: see paper
    :param dtype: data type, default float32
    :return: [1, seq_len, d_model] pos encoding
    """
    v = np.ndarray(shape=(1, seq_len, d_model), dtype=dtype)
    for i in range(seq_len):
        v[0, i, :] = get_pos_encoding(d_model, i, dtype)
    return v
