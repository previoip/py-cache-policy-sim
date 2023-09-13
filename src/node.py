import typing as t

class Edge:

  def __repr__(self):
    return f'<Edge [{self.a} --> {self.b}]>'

  def __init__(self, a, b, attr=None):
    self.a = a
    self.b = b
    self.attr = attr


class Node:

  def _assert_type(self, o):
    assert o is None or isinstance(o, self.__class__), f'arg is not instance of {self.__class__}'

  def __repr__(self):
    return f'<Node {self.name}>'

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
    return list(self._iter_children)

  @property
  def degree(self):
    return len(self.children)

  @property
  def depth(self):
    return sum(map(lambda _: 1, self._iter_parents))

  @property
  def is_root(self):
    return self.parent is None

  @property
  def is_leaf(self):
    return self.degree == 0

  @property
  def is_internal(self):
    return self.is_leaf and not self.is_root

  @property
  def _iter_edges(self):
    yield from self._edges

  @property
  def _iter_parents(self):
    p = self.parent
    while not p is None:
      yield p
      p = p.parent

  @property
  def _iter_siblings(self):
    if self.parent is None:
      return
    for ch in self.parent._iter_children:
      if ch == self:
        continue
      yield ch

  @property 
  def _iter_children(self):
    for n, edge in enumerate(self._edges):
      if n == self._parent_index:
        continue
      yield edge.b

  def get_edge_index_by_ref(self, o):
    self._assert_type(o)
    for n, edge in enumerate(self._edges):
      if edge.b == o:
        return n
    return None

  def get_child_index_by_attr(self, value, attr):
    for n, ch in enumerate(self._iter_children):
      if getattr(ch, attr) == value:
        return n
    return None

  def get_child_index_by_name(self, name):
    return self.get_child_index_by_attr(name, 'name')

  def get_child_by_name(self, name):
    i = self.get_child_index_by_name(name)
    if i is None:
      return None
    return self.children[i]

  def append_child(self, child):
    self._assert_type(child)
    if child in self.children:
      return
    edge = Edge(self, child)
    self._edges.append(edge)

  def pop_child_by_name(self, name):
    i = self.get_child_index_by_name(name)
    if i is None:
      return
    edge = self._edges.pop(i)
    ret = edge.b
    del edge
    return ret