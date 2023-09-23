import typing as t
import queue


class Listener:
  pass

class Event:
  def __init__(self, name, params=None, event_manager=None):
    self._name = name
    self._params = {} if params is None else params
    self._event_manager = event_manager
    self._listeners = []

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
  def params(self, v: dict):
    self._params.update(v)

  @property
  def listeners(self):
    return self._listeners.copy()

  def add_listener(self, listener: Listener):
    self._listeners.append(listener)

  def set_event_manager(self, event_manager):
    self._event_manager = event_manager

class EventManager:
  _event_prototype = Event

  def __init__(self, job_scheduler=None):
    self.event_pool: t.Mapping[t.Hashable, self._event_prototype] = {}
    self.job_scheduler: queue.Queue = job_scheduler

  def new_event(self, name, target_host=None, params=None):
    event = self._event_prototype(name, target_host=target_host, params=params, event_manager=self)
    self.attach_event(name, event)
    return event

  def attach_event(self, name, event):
    self.event_pool[name] = event

  def trigger_event(self, name):
    event = self.event_pool.get(name)
    if event is None:
      return
    for listener in event.listeners:
      self.job_scheduler.put(listener)
