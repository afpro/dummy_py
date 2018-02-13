import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

__all__ = [
    'rnn_cell',
    'broadcast_matmul',
    'embedding',
    'cross_entropy',
    'attention',
    'softmax',
    'Loop'
]


def rnn_cell(hiddens, cell_fn=LSTMCell):
    return MultiRNNCell([cell_fn(hidden) for hidden in hiddens])


def broadcast_matmul(x, y):
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    v = tf.matmul(tf.reshape(x, (-1, x_shape[-1])), y)
    return tf.reshape(v, shape=tf.concat((x_shape[:-1], y_shape[-1:]), axis=0))


def embedding(ids, in_size, out_size, name=None):
    with tf.variable_scope(name, default_name='embedding', values=[ids]):
        return tf.nn.embedding_lookup(
            params=tf.get_variable(name='param',
                                   dtype=tf.float32,
                                   shape=(in_size, out_size)),
            ids=ids,
            name='output')


def cross_entropy(logits, labels, seq_len, name=None):
    with tf.name_scope(name, default_name='cross_entropy', values=[logits, labels, seq_len]):
        """ return (sum entropy , mean entropy) """
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


def softmax(a):
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
    @classmethod
    def loop_vars(cls, extra):
        raise NotImplementedError

    @classmethod
    def loop_cond(cls, args, extra):
        raise NotImplementedError

    @classmethod
    def loop_body(cls, args, extra):
        raise NotImplementedError

    @classmethod
    def reduce_result(cls, args, extra):
        return args

    @classmethod
    def invoke(cls, extra=None, name=None, debug=False):
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

        result_vars = tf.while_loop(
            cond=mock_cond,
            body=mock_body,
            loop_vars=loop_var_tensors,
            parallel_iterations=1 if debug else 10,
            name=name)
        return cls.reduce_result(DictToProperty.wrap(loop_vars_to_args(result_vars)), extra)


class CrossEntropyLoop(Loop):
    @classmethod
    def loop_vars(cls, extra):
        return {
            'i': tf.constant(0, tf.int32),
            'cent_sum': tf.constant(0, tf.float32),
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
                    'cent_sum': tf.constant(0, tf.float32),
                }

            @classmethod
            def loop_cond(cls, _args, _extra):
                return _args.i < length

            @classmethod
            def loop_body(cls, _args, _extra):
                return {
                    'i': _args.i + 1,
                    'cent_sum': _args.cent_sum + log_sum_exp[_args.i] - row[_args.i, label[_args.i]],
                }

            @classmethod
            def reduce_result(cls, _args, _extra):
                return _args.cent_sum

        return {
            'i': args.i + 1,
            'cent_sum': args.cent_sum + InnerLoop.invoke()
        }

    @classmethod
    def reduce_result(cls, args, extra):
        return args.cent_sum

    @classmethod
    def run(cls, logits, labels, seq_len, batch_size=None):
        return cls.invoke({
            'logits': logits,
            'labels': labels,
            'seq_len': seq_len,
            'batch_size': batch_size or tf.shape(seq_len)[0],
        })

    @classmethod
    def test(cls):
        # random logits
        logits = np.random.rand(3, 4, 10).astype(np.float32)
        labels = np.random.randint(0, 10, size=[3, 4], dtype=np.int32)
        seq_len = np.random.randint(0, 4, size=[3], dtype=np.int32)

        # with tf
        with tf.Graph().as_default(), tf.Session() as session:
            s = cls.run(tf.constant(logits), tf.constant(labels), tf.constant(seq_len))
            s = session.run(s)

        # with np
        cent_sum = 0
        for batch in range(3):
            for t in range(seq_len[batch]):
                label = labels[batch, t]
                logit = logits[batch, t]
                e = np.exp(logit)
                p = e[label] / np.sum(e)
                cent = - np.log(p)
                cent_sum += cent

        # check
        assert np.abs(s - cent_sum) < 1e-3