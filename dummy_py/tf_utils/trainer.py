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
import typing
from time import time

import tensorflow as tf

__all__ = [
    'TimerTask',
    'TrainStep',
    'train',
    'BaseTrainStep',
]


class TimerTask:
    """
    timer task for train loop
    """

    def __init__(self, interval: 'float', as_finalizer: 'bool', f: 'typing.Callable', *args, **kwargs):
        """
        :param interval: in seconds
        :param as_finalizer: execute on finish
        :param f: target function
        :param args: arguments for f
        :param kwargs: named arguments for f
        """
        self._interval = interval
        self._as_finalizer = as_finalizer
        self._f = lambda: f(*args, **kwargs)

    @property
    def interval(self) -> 'float':
        return self._interval

    @property
    def as_finalizer(self):
        return self._as_finalizer

    def __call__(self):
        self._f()


class TimerTaskStatus:
    """
    internal used class, for tracking TimerTask timestamp
    """

    def __init__(self, task: 'TimerTask', t: 'float'):
        self.task = task
        self.t = t
        self.just_executed = False

    def proceed(self, t: 'float'):
        if self.t + self.task.interval <= t:
            self.task()
            self.just_executed = True
            self.t = t
        else:
            self.just_executed = False

    def finalize(self):
        if self.task.as_finalizer and not self.just_executed:
            self.task()
            self.just_executed = True
            self.t = time()


class TrainStep:
    """
    train step interface
    """

    def finished(self) -> 'bool':
        """
        :return: if this train step is finished
        """
        raise NotImplemented

    def step(self):
        raise NotImplemented


def train(step: 'TrainStep', timer_tasks: 'typing.Iterable[TimerTask]'):
    """
    train loop
    :param step: train step
    :param timer_tasks: timer tasks executed in loop
    """
    now = time()
    tasks = [TimerTaskStatus(_, now) for _ in timer_tasks]

    while not step.finished():
        step.step()
        now = time()
        for task in tasks:
            task.proceed(now)

    for task in tasks:
        task.finalize()


class BaseTrainStep(TrainStep):
    """
    basic implements of TrainStep, with abstract data source methods
    """

    def __init__(self, session: 'tf.Session', train_op: 'tf.Tensor',
                 file_writer: 'tf.summary.FileWriter' = None,
                 global_step: 'tf.Tensor' = None,
                 summary: 'tf.Tensor' = None):
        """
        :param session: tensorflow session
        :param train_op: train operator tensor
        :param file_writer: (optional) summary file writer
        :param global_step: (optional) global step tensor, used by summary log
        :param summary: (optional) summary
        """
        if file_writer is not None and summary is not None:
            if global_step is None:
                raise RuntimeError("global_step is needed while file_writer and summary provided")

        self._session = session
        self._train_op = train_op
        self._file_writer = file_writer
        self._global_step = global_step
        self._summary = summary
        self._finished = False

    def next_batch(self):
        raise NotImplemented

    def feed_dict_of_batch_data(self, batch_data):
        raise NotImplemented

    def finished(self) -> 'bool':
        return self._finished

    def step(self):
        batch_data = self.next_batch()
        if batch_data is None:
            self._finished = True
            return

        feed_dict = self.feed_dict_of_batch_data(batch_data)
        del batch_data

        if self._file_writer is not None and self._summary is not None:
            _, s, gs = self._session.run(
                fetches=(self._train_op, self._summary, self._global_step),
                feed_dict=feed_dict)
            self._file_writer.add_summary(s, gs)
        else:
            self._session.run(fetches=self._train_op,
                              feed_dict=feed_dict)
        del feed_dict
