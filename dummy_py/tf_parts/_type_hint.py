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

import numpy as np
import tensorflow as tf

if not typing.TYPE_CHECKING:
    __all__ = []
else:
    __all__ = [
        'typing',
        'tf_const_input',
        'tf_input',
        'tf_shape',
        'var_scope_or_name',
    ]
    tf_const_input = typing.Union[np.ndarray, list, tuple, int, float]
    tf_input = typing.Union[tf.Tensor, tf_const_input]
    tf_shape = typing.Union[tf.TensorShape, tf_const_input]
    var_scope_or_name = typing.Union[tf.VariableScope, str]
