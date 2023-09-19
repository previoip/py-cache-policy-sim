import typing as t

class Event:
  def __init__(self, name, params=None):
    self._name = name
    self._params = {} if params is None else params

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, v):
    self._name = v

  @property
  def params(self):
    return self._params

  @params.setter
  def params(self, v):
    self._params = v


class EventManager:
  _event_prototype = Event

  def __init__(self):
    self.event_pool: t.Mapping[t.Hashable, self.__class__] = {}

  def new_event(self, name, target_host=None, params=None):
    event = self._event_prototype(name, target_host=target_host, params=params)
    self.attach_event(name, event)
    return event

  def attach_event(self, name, event):
    self.event_pool[name] = event