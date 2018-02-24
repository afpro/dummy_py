import inspect

__all__ = [
    'OrmField',
    'OrmObject',
]


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
        self.backup_attribute_name = '_$ja_{}'.format(field_name)
        self.strict_assign = strict_assign
        self.is_orm_object = issubclass(field_type, OrmObject)

    def __get__(self, instance, owner):
        return self.get_value(instance) if instance is not None else self

    def __set__(self, instance, value):
        self.set_value(instance, value)

    def __delete__(self, instance):
        self.clear(instance)

    def has_been_set(self, instance):
        return hasattr(instance, self.backup_attribute_name)

    def get_value(self, instance):
        return getattr(instance, self.backup_attribute_name, self.field_default_value)

    def set_value(self, instance, value, ensure_type=False):
        if value is None:
            if self.is_orm_object and self.has_been_set(instance):
                self.clear(instance)
            else:
                setattr(instance, self.backup_attribute_name, value)
            return

        if ensure_type or self.strict_assign:
            if self.is_orm_object:
                if self.has_been_set(instance):
                    self.get_value(instance).orm_dict = value
                    return

                v = self.field_type()
                v.orm_dict = value
                value = v
            else:
                value = self.field_type(value)
        setattr(instance, self.backup_attribute_name, value)

    def clear(self, instance):
        delattr(instance, self.backup_attribute_name)


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
        for n, t in inspect.getmembers(type(self)):
            if isinstance(t, OrmField):
                yield n, t

    @property
    def orm_dict(self):
        d = {}
        for n, t in self.orm_fields:
            if not t.has_been_set(self):
                continue
            v = t.get_value(self)
            if t.is_orm_object:
                sub_dict = v.orm_dict
                if len(sub_dict) > 0:
                    d[n] = sub_dict
            else:
                d[n] = v
        return d

    @orm_dict.setter
    def orm_dict(self, d):
        if not isinstance(d, dict):
            raise RuntimeError('value is not a dict')
        for n, t in self.orm_fields:
            if n not in d:
                t.clear(self)
                continue
            t.set_value(self, d[n], ensure_type=True)
