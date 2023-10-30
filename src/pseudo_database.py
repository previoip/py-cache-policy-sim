import typing as t
from threading import Lock, RLock
from collections import namedtuple
import sys


class ABCPseudoDatabase:
  _name: str
  _container: t.Collection
  _lock: Lock = RLock()

  def __init__(self, name, container: t.Collection):
    self._name = name
    self._container = container

  def __repr__(self):
    return f'<{self.__class__.__name__} {self._name} with {len(self._container)} records>'

  def set_container(self, container: t.Collection):
    self._container = container

  def get_container(self):
    if hasattr(self._container, 'copy'):
      return self._container.copy()
    return self._container

  def has(self, key: t.Hashable):
    with self._lock:
      return self._has(key)

  def get(self, key: t.Hashable):
    with self._lock:
      return self._get(key)

  def add(self, key: t.Hashable, entry: t.Any):
    with self._lock:
      self._add(key, entry)

  def clear(self):
    with self._lock:
      self._clear()

  def _has(self, key: t.Hashable):
    raise NotImplementedError()

  def _get(self, key: t.Hashable):
    raise NotImplementedError()

  def _add(self, key: t.Hashable, entry: t.Any):
    raise NotImplementedError()

  def _clear(self):
    raise NotImplementedError()

  def dump(self, fp: t.TextIO, delim=';'):
    raise NotImplementedError()


from pandas import DataFrame

class PandasDataFramePDB(ABCPseudoDatabase):
  """ readonly pandas dataframe object PDB """
  def __init__(self, name, container: DataFrame):
    super().__init__(name=name, container=container)
    self._index = container.index

  def _has(self, key: t.Hashable):
    return key in self._index

  def _get(self, key: t.Hashable):
    if not self._has(key):
      return None
    return self._container.loc[key].values


class ListPDB(ABCPseudoDatabase):

  def __init__(self, name, container: list=list()):
    super().__init__(name=name, container=container)
    self.field_dtypes = 'int64'
    self._cursor = 0

  def _has(self, key: t.Hashable):
    return key < len(self._container)

  def _get(self, key: t.Hashable):
    return self._container[key]

  def _add(self, key: t.Hashable, entry: t.Any):
    return self._container.append(entry)

  def _set_cursor(self):
    self._cursor = len(self._container)

  def get_container_by_cursor(self):
    return c

  def to_pd(self, use_cursor=False):
    if use_cursor:
      c = self._container[self._cursor:]
    else:
      c = self._container
    self._set_cursor()
    if hasattr(self, 'field_names'):
      # _dtypes = list(zip(self.field_names, self.field_dtypes))
      return DataFrame(data=c, columns=self.field_names)#.astype(_dtypes)
    return DataFrame(data=c)

  def dump(self, fp: t.TextIO, delim: str=';'):
    # if hasattr(self, 'field_names'):
    #   fp.writelines((delim.join(self.field_names), '\n'))
    # for entry in self._container:
    #   try:
    #     fp.writelines((delim.join(map(lambda x: str(x), iter(entry))), '\n'))
    #   except TypeError:
    #     fp.writelines((str(entry), '\n'))
    self.to_pd(use_cursor=False).to_csv(fp, sep=delim)


class TabularPDB(ListPDB):

  def __init__(self, name, container: list=list(), field_names=['timestamp', 'entry'], field_dtypes=['uint64', 'string']):
    super().__init__(name=name, container=container)
    self.field_names = field_names
    self.field_dtypes = field_dtypes
    self._entry_prototype = namedtuple(f'{self._name}Record', field_names)

  def add_entry(self, *values):
    with self._lock:
      self._add(None, self._entry_prototype(*values))
