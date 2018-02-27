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
from copy import copy
from inspect import getmembers

__all__ = [
    'OrmField',
    'OrmObject',
]

_orm_dict_field = '_$dummy_py_orm_dict'


class OrmField:
    """
    OrmField, similar to property
    """

    def __init__(self, field_type, field_name,
                 field_default_value=None,
                 strict_assign=False):
        """
        :param field_type: field type (eg. int)
        :param field_name: field name (name in dict)
        :param field_default_value: default value if not present
        :param strict_assign: whether check value type while assign
        """
        super().__init__()
        assert isinstance(field_type, type)
        assert isinstance(field_name, str)
        assert field_default_value is None or isinstance(field_default_value, field_type)
        self.field_type = field_type
        self.field_name = field_name
        self.field_default_value = field_default_value
        self.strict_assign = strict_assign
        self.is_orm_object = issubclass(field_type, OrmObject)

    def __get__(self, instance, owner):
        return self.get_value(instance) if instance is not None else self

    def __set__(self, instance, value):
        self.set_value(instance, value)

    def __delete__(self, instance):
        self.clear(instance)

    def clear(self, instance):
        if not hasattr(instance, _orm_dict_field):
            return
        d = getattr(instance, _orm_dict_field)
        if self.field_name in d:
            del d[self.field_name]

    def has_been_set(self, instance):
        if not hasattr(instance, _orm_dict_field):
            return False
        d = getattr(instance, _orm_dict_field)
        return self.field_name in d

    def get_value(self, instance):
        if not hasattr(instance, _orm_dict_field):
            return self.field_default_value

        d = getattr(instance, _orm_dict_field)
        return d.get(self.field_name, self.field_default_value)

    @staticmethod
    def inner_dict(instance):
        if hasattr(instance, _orm_dict_field):
            d = getattr(instance, _orm_dict_field)
        else:
            d = {}
            setattr(instance, _orm_dict_field, d)
        return d

    def set_value(self, instance, value, ensure_type=False):
        if value is None:
            if self.is_orm_object and self.has_been_set(instance):
                delattr(instance, _orm_dict_field)
            else:
                OrmField.inner_dict(instance)[self.field_name] = value
            return

        d = OrmField.inner_dict(instance)
        if ensure_type or self.strict_assign:
            if self.is_orm_object:
                v = d.get(self.field_name, None)
                if v is None:
                    v = self.field_type()
                v.orm_dict = value
                d[self.field_name] = v
            else:
                d[self.field_name] = self.field_type(value)
        else:
            d[self.field_name] = value


class OrmObject:
    """
    super class to a orm object

    >>> class P(OrmObject):
    >>>     a = OrmField(int, 'a', 0)
    >>>     b = OrmField(str, 'b', 'def-a')
    >>> class Q(OrmObject):
    >>>     a = OrmField(int, 'a', 1)
    >>>     b = OrmField(P, 'b', None)
    >>> q = Q()
    >>> q.orm_dict # {}
    >>> q.orm_dict = {
    >>>     'a': 1,
    >>>     'b': {
    >>>         'a': 2,
    >>>         'b': 'b',
    >>>      },
    >>> }
    >>> q.a, q.b.a, q.b.b
    1 2 b
    """

    @property
    def orm_fields(self):
        for n, t in getmembers(type(self)):
            if isinstance(t, OrmField):
                yield n, t

    @property
    def orm_dict(self):
        d = getattr(self, _orm_dict_field, None)
        if d is not None:
            d = copy(d)
        return d

    @orm_dict.setter
    def orm_dict(self, d):
        if d is not None and not isinstance(d, dict):
            raise RuntimeError('value is not a dict')

        if d is None or len(d) == 0:
            if hasattr(self, _orm_dict_field):
                getattr(self, _orm_dict_field).clear()
            return

        for n, t in self.orm_fields:
            if n not in d:
                t.clear(self)
                continue
            t.set_value(self, d[n], ensure_type=True)
