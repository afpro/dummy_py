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
import errno
import logging
import sys
from multiprocessing import Process

import tensorflow as tf
from tensorboard.backend import application
from tensorboard.plugins.core import core_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.plugins.scalar import scalars_plugin
from werkzeug import serving

__all__ = [
    'reduce_log',
    'create_serve_fun',
    'serve_with_process',
]


def reduce_log():
    """
    reduce tensorboard log to error level
    """
    _logger = logging.getLogger('werkzeug')
    _logger.setLevel(logging.ERROR)


def create_serve_fun(log_dir: 'str', host='0.0.0.0', port=8000):
    """
    create tensorboard serve fun
    :param log_dir: log dir (as tf.summary.FileWriter)
    :param host: host ip
    :param port: host port
    :return: serve fun
    """

    def _handle_error(_, client_address):
        exc_info = sys.exc_info()
        e = exc_info[1]
        if isinstance(e, IOError) and e.errno == errno.EPIPE:
            tf.logging.warn('EPIPE caused by %s:%d in HTTP serving' % client_address)
        else:
            tf.logging.error('HTTP serving error', exc_info=exc_info)

    tb = application.standard_tensorboard_wsgi(
        logdir=log_dir,
        purge_orphaned_data=True,
        reload_interval=5,
        plugins=[
            core_plugin.CorePlugin,
            scalars_plugin.ScalarsPlugin,
            graphs_plugin.GraphsPlugin,
        ])
    server = serving.make_server(host=host, port=port, app=tb, threaded=True)
    server.daemon_threads = True
    server.handle_error = _handle_error
    return server.serve_forever


def serve_with_process(log_dir: 'str', host='0.0.0.0', port=8000):
    """
    run tensorboard in child daemon process
    :param log_dir: log dir (as tf.summary.FileWriter)
    :param host: host ip
    :param port: host port
    """
    p = Process(target=create_serve_fun(log_dir, host, port))
    p.daemon = True
    p.start()
