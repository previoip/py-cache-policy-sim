import typing as t
from threading import Lock, RLock


class ABCPseudoDatabase:
  _name: str
  _container: t.Collection
  _lock: Lock = RLock()

  def __init__(self, name, container: t.Collection):
    self._name = name
    self._container = container

  def __repr__(self):
    return f'<{self.__class__.__name__} {len(self._container)} records>'

  def set_container(self, container: t.Collection):
    self._container = container

  def has(self, key: t.Hashable):
    with self._lock:
      return self._has(key)

  def get(self, key: t.Hashable):
    with self._lock:
      return self._get(key)

  def _has(self, key: t.Hashable):
    raise NotImplementedError

  def _get(self, key: t.Hashable):
    raise NotImplementedError


from pandas import DataFrame

class PandasDataFramePDB(ABCPseudoDatabase):
  """ readonly pandas dataframe object PDB """
  def __init__(self, name, container: DataFrame):
    super().__init__(name=name, container=container)
    self._index = container.index

  def _has(self, key: t.Hashable):
    return key in self._index

  def _get(self, key: t.Hashable):
    if not self.has(key):
      return None
    return self._container.loc[key].values