import os
import tensorflow as tf

__all__ = [
    'save',
    'load',
    'gpu_growth_config',
]


def load(session, saver, save_path):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path)

    latest_checkpoint = tf.train.latest_checkpoint(save_dir, '{}.ckpt'.format(save_name))
    if latest_checkpoint is None:
        print(f'no checkpoint found')
        return False
    saver.restore(session, latest_checkpoint)
    print('restored {}'.format(latest_checkpoint))
    return True


def save(session, saver, global_step, save_path):
    save_name = os.path.basename(save_path)
    saver.save(session, save_path,
               global_step=global_step,
               latest_filename='{}.ckpt'.format(save_name))
    print('saved {}'.format(save_path))


def gpu_growth_config(in_config=None):
    config = in_config if isinstance(in_config, tf.ConfigProto) else tf.ConfigProto()
    config.graph_options.optimizer_options.do_function_inlining = True
    config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L1
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
    config.gpu_options.allow_growth = True
    return config
