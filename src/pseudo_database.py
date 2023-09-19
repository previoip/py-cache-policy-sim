import typing as t

class ABCPseudoDatabase:
  __name: str
  __container: t.Collection

  def __init__(self, name, container: t.Collection):
    self.__name = name
    self.__container = container

  def __repr__(self):
    return f'<PDB {len(self.__container)} records>'

  def set_container(self, container: t.Collection):
    self.__container = container

  def has(self, key: t.Hashable):
    raise NotImplementedError

  def get(self, key: t.Hashable):
    raise NotImplementedError

  def set(self, key: t.Hashable, value: t.Any):
    raise NotImplementedError
  
  def iter_keys(self):
    raise NotImplementedError


from pandas import DataFrame

class PandasDataFramePDB(ABCPseudoDatabase):
  def __init__(self, name, container: DataFrame):
    super().__init__(name=name, container=container)
    self._index = container.index
  