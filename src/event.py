import typing as t
import queue
from collections import namedtuple

EventContext = namedtuple('EventContext', ['event_target', 'event_manager'], defaults=[None, None])

def new_event_thread_worker(scheduler: queue.Queue) -> t.Callable:
  def worker():
    while True:
      listener, context = scheduler.get()
      errno = listener(context)
      scheduler.task_done()
      if errno != 0:
        pass
  return worker

class Event:
 
  def __init__(self, event_type):
    self._event_type = event_type
    self._event_listeners = []

  @property
  def listeners(self):
    return self._listeners.copy()

  @property
  def event_type(self):
    return self._event_type

  def add_listener(self, name, listener):
    self._event_listeners.append(listener)

  def run(self, scheduler, context):
    for listener in self._event_listeners:
      scheduler.put((listener, context))


class EventManager:
  _event_prototype = Event
  _context_prototype = EventContext

  def __init__(self, event_target, job_scheduler=None):
    self._event_target = event_target
    self._job_scheduler: queue.Queue = job_scheduler
    self._event_pool: t.Mapping[t.Hashable, self._event_prototype] = {}

  def set_scheduler(self, scheduler):
    self._job_scheduler = scheduler

  @staticmethod
  def new_event(event_type):
    return EventManager._event_prototype(event_type)

  def attach_event(self, event: Event):
    self._event_pool[event.event_type] = event

  def get_event(self, event_type):
    return self._event_pool.get(event_type, default=None)

  def trigger_event(self, event_type):
    event = self.get_event(event_type)
    if event is None:
      return

    event.run(
      self._job_scheduler,
      self._context_prototype(self._event_target, self)
    )