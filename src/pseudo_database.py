import typing as t
from threading import Lock, RLock
from collections import namedtuple

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

  def _has(self, key: t.Hashable):
    return key < len(self._container)

  def _get(self, key: t.Hashable):
    return self._container[key]

  def _add(self, key: t.Hashable, entry: t.Any):
    return self._container.append(entry)

class RequestLogDatabase(ListPDB):

  _entry_prototype = namedtuple('LogRecord', ['timestamp', 'user_id', 'item_id'])

  def __init__(self, name, container: list=list()):
    super().__init__(name=name, container=container)

  def add_entry(self, timestamp, user_id, item_id):
    with self._lock:
      self._add(None, self._entry_prototype(timestamp, user_id, item_id))

