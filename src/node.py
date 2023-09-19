import typing as t

class Edge:

  def __repr__(self):
    return f'Edge [{self.a} --> {self.b}]'

  def __init__(self, a, b, attr=None):
    self.a = a
    self.b = b
    self.attr = attr

  def graph_repr(self):
    # mermaid diagram node repr
    return f'{id(self.a)}[{self.a}] --> {id(self.b)}[{self.b}]'


class Node:

  def _assert_type(self, o):
    assert o is None or isinstance(o, self.__class__), f'arg is not instance of {self.__class__}'

  def __repr__(self):
    return f'{self.__class__.__name__} {self.name}'

  def __init__(self, name=None, parent=None):
    self.name = str(id(self)) if name is None else name
    self._edges = []
    self._parent_index = None

    if not parent is None:
      self.parent = parent
      self.parent.append_child(self)

  @property
  def parent(self):
    if self._parent_index is None:
      return None
    return self._edges[self._parent_index].b

  @parent.setter
  def parent(self, o):
    self._assert_type(o)
    if o == self:
      raise ValueError('parent cannot refer to self')
    if not self._parent_index is None:
      self._edges.pop(self._parent_index)
    edge = Edge(self, o)
    self._edges.append(edge)
    self._parent_index = self._edges.index(edge)

  @property
  def children(self):
    return list(self._iter_children())

  @property
  def degree(self):
    return len(self.children)

  @property
  def depth(self):
    return sum(map(lambda _: 1, self._iter_parents()))

  def is_root(self):
    return self.parent is None

  def is_leaf(self):
    return self.degree == 0

  def is_internal(self):
    return self.is_leaf() and not self.is_root()

  def _iter_edges(self) -> t.Generator:
    yield from self._edges

  def _iter_parents(self) -> t.Generator:
    p = self.parent
    while not p is None:
      yield p
      p = p.parent

  def _iter_siblings(self) -> t.Generator:
    if self.parent is None:
      return
    for ch in self.parent._iter_children():
      if ch == self:
        continue
      yield ch

  def _iter_children(self) -> t.Generator:
    for n, edge in enumerate(self._edges):
      if n == self._parent_index:
        continue
      yield edge.b

  def _recurse_nodes(self) -> t.Generator:
    yield self
    for ch in self._iter_children():
      yield ch
      if ch.degree > 0:
        yield ch._recurse_nodes()

  def recurse_nodes(self) -> t.Generator:
    yield from self._recurse_nodes()

  def recurse_callback(self, callback: t.Callable, *args, **kwargs) -> t.Generator:
    for node in self._recurse_nodes():
      yield callback(node, *args, **kwargs)

  def recurse_edges(self) -> t.Generator:
    for node in self._recurse_nodes():
      for edge in node._iter_edges():
        yield edge

  def get_edge_index_by_ref(self, o):
    self._assert_type(o)
    for n, edge in enumerate(self._edges):
      if edge.b == o:
        return n
    return None

  def get_child_index_by_attr(self, value, attr):
    for n, ch in enumerate(self._iter_children()):
      if getattr(ch, attr) == value:
        return n
    return None

  def get_child_index_by_name(self, name):
    return self.get_child_index_by_attr(name, 'name')

  def get_child_by_name(self, name):
    i = self.get_child_index_by_name(name)
    if i is None:
      return
    return self.children[i]

  def spawn_child(self, name):
    return self.__class__(name=name, parent=self)

  def append_child(self, child):
    self._assert_type(child)
    if not self.get_child_by_name(child.name) is None:
      raise ValueError(f'child {child} name already exists in parent children')
    if child in self.children:
      raise ValueError(f'child {child} already exists in parent children')
    edge = Edge(self, child)
    self._edges.append(edge)

  def pop_child_by_name(self, name):
    i = self.get_child_index_by_name(name)
    if i is None:
      return
    edge = self._edges.pop(i)
    ret = edge.b
    return ret