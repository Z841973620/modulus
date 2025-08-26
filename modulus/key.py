""" Key
"""

from functools import reduce

from .constants import diff_str


class Key(object):
    """
    Class describing keys used for graph unroll.
    The most basic key is just a simple string
    however you can also add dimension information
    and even information on how to scale inputs
    to networks.

    Parameters
    ----------
    name : str
      String used to refer to the variable (e.g. 'x', 'y'...).
    size : int=1
      Dimension of variable.
    derivatives : List=[]
      This signifies that this key holds a derivative with
      respect to that key.
    scale: (float, float)
      Characteristic location and scale of quantity: used for normalisation.
    """

    def __init__(self, name, size=1, derivatives=[], base_unit=None, scale=(0.0, 1.0)):
        super(Key, self).__init__()
        self.name = name
        self.size = size
        self.derivatives = derivatives
        self.base_unit = base_unit
        self.scale = scale

    @classmethod
    def from_str(cls, name):
        split_name = name.split(diff_str)
        var_name = split_name[0]
        diff_names = Key.convert_list(split_name[1:])
        return cls(var_name, size=1, derivatives=diff_names)

    @classmethod
    def from_tuple(cls, name_size):
        split_name = name_size[0].split(diff_str)
        var_name = split_name[0]
        diff_names = Key.convert_list(split_name[1:])
        return cls(var_name, size=name_size[1], derivatives=diff_names)

    @classmethod
    def convert(cls, name_or_tuple):
        if isinstance(name_or_tuple, str):
            key = Key.from_str(name_or_tuple)
        elif isinstance(name_or_tuple, tuple):
            key = cls.from_tuple(name_or_tuple)
        elif isinstance(name_or_tuple, cls):
            key = name_or_tuple
        else:
            raise ValueError("can only convert string or tuple to key")
        return key

    @staticmethod
    def convert_list(ls):
        keys = []
        for name_or_tuple in ls:
            keys.append(Key.convert(name_or_tuple))
        return keys

    @property
    def unit(self):
        return self.base_unit / reduce(
            lambda x, y: x.base_unit * y.base_unit, self.derivatives
        )

    def __str__(self):
        diff_str = "".join(["__" + x.name for x in self.derivatives])
        return self.name + diff_str

    def __repr__(self):
        return str(self)

    def __eq__(self, obj):
        return isinstance(obj, Key) and str(self) == str(obj)

    def __hash__(self):
        return hash(str(self))


def _length_key_list(list_keys):
    length = 0
    for key in list_keys:
        length += key.size
    return length
