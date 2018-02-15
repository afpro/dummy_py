# DummyPy

afpro's python library for dummies, include

* python common utils (eg. lazy property)
* tensorflow utils (eg. load/save, gpu growth config)
* common models (eg. google attention model)

## `dummy_py.common`

common python utils

* lazy property
* io utils
* simplify map/filter method as Sequence class

## `dummy_py.tf_utils`

include tensorflow util codes

* load checkpoint
* save checkpoint
* create `tf.ConfigProto` for gpu memory growth support
* create multi layer LSTM rnn cell
* create embedding lookup structure
* calculate cross entropy with sequence length support
* general attention calculation
* numpy softmax implements
* tensorflow while loop abstraction as `Loop` class
* `tf_session_fn` decorator

## `dummy_py.tf_parts`

include tensorflow model elements

* google attention model [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)