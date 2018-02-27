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
__all__ = [
    'echo',
    'TakeFn',
    'take_fn',
]


def echo(anything):
    """
    echo, return param
    """
    return anything


class TakeFn:
    """
    take_fn, utils for map

    >>> v = [(0,1), (2,3), (4,5)]
    >>> list(map(take_fn[0], v))
    [0, 2, 4]
    """

    def __getitem__(self, item):
        def inner(_):
            return _[item]

        return inner


take_fn = TakeFn()
