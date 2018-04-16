import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

from dummy_py.tf_parts.type_hint import *
from dummy_py.tf_utils import Loop

__all__ = [
    'state_shape',
    'zero_state',
    'decode',
]


def state_shape(num_layers, hidden_size, batch_size: 'tf_input' = None):
    """
    :param num_layers: decoder layer count
    :param hidden_size: decoder rnn hidden size (LSTM)
    :param batch_size: batch size
    """
    return batch_size, num_layers, 2, hidden_size


def zero_state(num_layers: 'int',
               hidden_size: 'int',
               batch_size: 'int',
               dtype: 'tf.DType' = tf.float32):
    """
    :param num_layers: decoder layer count
    :param hidden_size: decoder rnn hidden size (LSTM)
    :param batch_size: batch size
    :param dtype: output dtype, default float32
    :return: zero state as tensor
    """
    return tf.zeros(shape=state_shape(num_layers, hidden_size, batch_size), dtype=dtype)


def decode(num_layers: 'int',
           hidden_size: 'int',
           encoded: 'tf_input',
           encode_seq_len: 'tf_input',
           decode_input: 'tf_input',
           decode_seq_len: 'tf_input',
           state_input: 'tf_input',
           dtype: 'tf.DType' = tf.float32):
    """
    :param num_layers: decoder layer count
    :param hidden_size: decoder rnn hidden size (LSTM)
    :param encoded: output from decoder, shape=[Batch, EncoderSeqLength, *]
    :param encode_seq_len: encoder sequence length shape=[Batch]
    :param decode_input: decoder input, shape=[Batch, DecoderSeqLength, *]
    :param decode_seq_len: decoder sequence length shape=[Batch]
    :param state_input: decoder initial state shape=[Batch, num_layers, 2, hidden_size]
    :param dtype: output dtype, default float32
    :return: output shape=[Batch, DecoderSeqLength, hidden_size], states shape=[Batch, num_layers, 2, hidden_size]
    """
    assert num_layers > 0
    assert hidden_size > 0

    encoded = tf.convert_to_tensor(encoded, dtype)  # type: tf.Tensor
    encode_seq_len = tf.convert_to_tensor(encode_seq_len, tf.int32)  # type: tf.Tensor
    decode_input = tf.convert_to_tensor(decode_input, dtype)  # type: tf.Tensor
    decode_seq_len = tf.convert_to_tensor(decode_seq_len, tf.int32)  # type: tf.Tensor
    state_input = tf.convert_to_tensor(state_input, dtype)  # type: tf.Tensor

    assert encoded.shape.ndims == 3
    assert encoded.shape[-1] is not None
    assert encode_seq_len.shape.ndims == 1
    assert decode_input.shape.ndims == 3
    assert decode_seq_len.shape.ndims == 1
    assert state_input.shape.ndims == 4

    encoded_shape = tf.shape(encoded)
    encode_seq_len_shape = tf.shape(encode_seq_len)
    decode_input_shape = tf.shape(decode_input)
    decode_seq_len_shape = tf.shape(decode_seq_len)
    state_input_shape = tf.shape(state_input)

    max_encode_seq_len = tf.reduce_max(encode_seq_len)
    max_decode_seq_len = tf.reduce_max(decode_seq_len)

    batch_size = encoded_shape[0]

    class StepsLoop(Loop):
        @classmethod
        def loop_vars(cls, extra):
            return {
                'step': tf.constant(0, dtype=tf.int32),
                'state': state_input,
                'output': tf.TensorArray(dtype, size=max_decode_seq_len),
            }

        @classmethod
        def shape_invariants(cls, extra):
            return {
                'step': tf.TensorShape([]),
                'state': tf.TensorShape([None, num_layers, None, hidden_size]),
                'output': tf.TensorShape(None),
            }

        @classmethod
        def loop_cond(cls, args, extra):
            return args.step < max_decode_seq_len

        @classmethod
        def loop_body(cls, args, extra):
            # noinspection PyShadowingNames
            def calc():
                v = decode_input[:, args.step, :]
                state = []
                for layer in range(num_layers):
                    with tf.variable_scope('L{}'.format(layer)):
                        c = args.state[:, layer, 0, :]
                        h = args.state[:, layer, 1, :]
                        w = tf.get_variable('w',
                                            dtype=dtype,
                                            shape=(encoded.shape[-1], hidden_size))
                        attn_v = _LuongGeneralAttnLoop.run(encoded, encode_seq_len, h, w, batch_size, dtype)
                        cell = LSTMCell(hidden_size)
                        v, s = cell(attn_v, LSTMStateTuple(c=c, h=h))
                        state.append(tf.stack([s.c, s.h], axis=0))
                return v, tf.stack(state, axis=1)

            def dummy():
                return tf.zeros(shape=(batch_size, hidden_size), dtype=dtype), args.state

            v, s = tf.cond(args.step < decode_seq_len[args.step],
                           true_fn=calc, false_fn=dummy)

            return {
                'step': args.step + 1,
                'state': s,
                'output': args.output.write(args.step, v),
            }

        @classmethod
        def reduce_result(cls, args, extra):
            return tf.transpose(args.output.stack(), perm=(1, 0, 2)), args.state

    with tf.control_dependencies([
        tf.assert_equal(batch_size, encode_seq_len_shape[0], message='unify batch size'),
        tf.assert_equal(batch_size, decode_input_shape[0], message='unify batch size'),
        tf.assert_equal(batch_size, decode_seq_len_shape[0], message='unify batch size'),
        tf.assert_equal(batch_size, state_input_shape[0], message='unify batch size'),
        tf.assert_equal(num_layers, state_input_shape[1], message='check state input'),
        tf.assert_equal(2, state_input_shape[2], message='check state input'),
        tf.assert_equal(hidden_size, state_input_shape[3], message='check state input'),
        tf.assert_less_equal(max_encode_seq_len, encoded_shape[1], message='avoid encode seq len exceed'),
        tf.assert_less_equal(max_decode_seq_len, decode_input_shape[1], message='avoid decode seq len exceed'),
    ]):
        return StepsLoop.invoke()


class _LuongGeneralAttnLoop(Loop):
    @classmethod
    def loop_vars(cls, extra):
        return {
            'i': tf.constant(0, dtype=tf.int32),
            'v': tf.TensorArray(extra.dtype, size=extra.b)
        }

    @classmethod
    def loop_cond(cls, args, extra):
        return args.i < extra.b

    @classmethod
    def loop_body(cls, args, extra):
        e = extra.e[args.i, :extra.es[args.i]]
        c = extra.c[args.i, :, :extra.es[args.i]]
        s = tf.nn.softmax(c)
        v = tf.matmul(s, e)
        return {
            'v': args.v.write(args.i, v),
        }

    @classmethod
    def reduce_result(cls, args, extra):
        return tf.squeeze(args.v.stack(), axis=1)

    @classmethod
    def run(cls,
            e: 'tf.Tensor', es: 'tf.Tensor', h: 'tf.Tensor',
            w: 'tf.Tensor',
            batch_size: 'tf_input',
            dtype: 'tf.DType',
            **kwargs):
        # e  [B, ESL, NE]
        # es [B]
        # h  [B, ND]
        # w  [NE, ND]

        # c [B, NE]
        c = tf.matmul(h, w, transpose_b=True)

        # c [B, 1, NE]
        c = tf.expand_dims(c, axis=1)

        # c [B, 1, ESL]
        c = tf.matmul(c, e, transpose_b=True)

        return cls.invoke(extra={
            'e': e,
            'es': es,
            'c': c,
            'b': batch_size,
            'dtype': dtype,
        }, **kwargs)
