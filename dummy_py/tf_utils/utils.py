import os
import typing

import tensorflow as tf

__all__ = [
    'save',
    'load',
    'gpu_growth_config',
    'tf_session_fn',
]


def load(session: 'tf.Session', saver: 'tf.train.Saver', save_path: 'str'):
    """
    load a check point
    for example, save_path as /some/path/my-model
    will check file /some/path/my-model.ckpt and load latest checkpoint stored in this file
    maybe /some/path/my-model[.global step]

    :param session: tensorflow session
    :param saver: saver instance
    :param save_path: save path
    :return: true for success load
    """
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)

    latest_checkpoint = tf.train.latest_checkpoint(save_dir, '{}.ckpt'.format(save_name))
    if latest_checkpoint is None:
        return None
    saver.restore(session, latest_checkpoint)
    return latest_checkpoint


def save(session: 'tf.Session', saver: 'tf.train.Saver', global_step: 'typing.Union[int, tf.Tensor]', save_path: 'str'):
    """
    save a checkpoint
    for example, save_path as /some/path/my-model
    will save checkpoint as /some/path/my-model[.global step]
    and /some/path/my-model.ckpt

    :param session:  tensorflow session
    :param saver: saver instance
    :param global_step: global step, could be None
    :param save_path: save path
    :return a string for saved model path
    """
    save_name = os.path.basename(save_path)
    return saver.save(session, save_path,
                      global_step=global_step,
                      latest_filename='{}.ckpt'.format(save_name))


def gpu_growth_config(in_config: 'tf.ConfigProto' = None):
    """
    create a config, could be used in tf.Session(config=...)

    :param in_config: input config
    :return: config with gpu growth support and optimization on
    """
    config = in_config if isinstance(in_config, tf.ConfigProto) else tf.ConfigProto()
    config.graph_options.optimizer_options.do_function_inlining = True
    config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L1
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    config.gpu_options.allow_growth = True
    return config


def tf_session_fn(f, gpu_growth_session=True):
    """
    tensorflow session wrapper, invoke f under seperate graph/session, prepend tf.Session at argument
    :param f: actual method
    :param gpu_growth_session: whether or not use gpu growth config
    :return: wrapped method
    """
    def inner(*args, **kwargs):
        config = gpu_growth_config() if gpu_growth_session else None
        with tf.Graph().as_default(), tf.Session(config=config) as session:
            return f(session, *args, **kwargs)

    return inner
