import typing as t
from collections import namedtuple
from dataclasses import dataclass


def _g_check_type(self, o):
  if not isinstance(o, self.__class__): 
    raise ValueError(f'{o.__class__} is not instance of {self.__class__}')

@dataclass
class EdgeStates:
  passed: bool = False

@dataclass
class EdgeAttrs:
  weight = 1
  length = 0

class Edge:
  _check_type = _g_check_type

  _attrs_prototype = EdgeAttrs
  _states_prototype = EdgeStates

  def __repr__(self) -> str:
    return f'Edge [{self.a} -- {self.b}]'

  def __init__(self, a, b, attr=None):
    self._a = a
    self._b = b
    self._attrs = self._attrs_prototype()
    self._states = self._states_prototype()
    self._mirror = None

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  @property
  def attrs(self):
    return self._attrs

  @property
  def mirror(self):
    return self._mirror

  @mirror.setter
  def mirror(self, o):
    self._check_type(o)
    self._mirror = o

  def reset(self):
    self._attrs = self._attrs_prototype()
    self._states = self._states_prototype()

  def shallow_reset(self):
    self._states.passed = False

  def is_passed(self):
    return self._states.passed

  def set_passed(self):
    self._states.passed = True

  def graph_repr(self) -> str:
    # mermaid diagram node repr
    return '{}({}) -- w:{:.01f}, L:{:.01f} --> {}({})'.format(
      id(self.a),
      self.a,
      self._attrs.weight,
      self._attrs.length,
      id(self.b),
      self.b
    )


class Node:

  _check_type = _g_check_type

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
    self._check_type(o)
    if o == self:
      raise ValueError('parent cannot refer to self')
    if not self._parent_index is None:
      self._edges.pop(self._parent_index)
    edge = self._append_edge(o)
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

  def has(self, o):
    for node in self._iter_edges():
      if node.a == self and node.b == o:
        return True
    return False

  def _append_edge(self, o):
    self._check_type(o)
    edge = Edge(self, o)
    self._edges.append(edge)
    return edge

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
        yield from ch._recurse_nodes()

  def recurse_nodes(self) -> t.Generator:
    yield from self._recurse_nodes()

  def recurse_edges(self) -> t.Generator:
    for node in self._recurse_nodes():
      for edge in node._iter_edges():
        if not edge.is_passed():
          yield edge
        edge.set_passed()
      edge.shallow_reset()

  def get_edge_index_by_ref(self, o):
    self._check_type(o)
    for n, edge in enumerate(self._edges):
      if edge.a == self and edge.b == o:
        return n
    return None

  def get_edge_by_ref(self, o):
    i = self.get_edge_index_by_ref(o)
    if i is None:
      return
    return self._edges[i]

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
    self._check_type(child)
    if not self.get_child_by_name(child.name) is None:
      raise ValueError(f'child {child} name already exists in parent children')
    if child in self.children:
      raise ValueError(f'child {child} already exists in parent children')
    self._append_edge(child)

  def pop_child_by_name(self, name):
    i = self.get_child_index_by_name(name)
    if i is None:
      return
    edge = self._edges.pop(i)
    ret = edge.b
    return ret