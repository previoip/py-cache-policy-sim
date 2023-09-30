import typing as t
import queue
import sys
import traceback
from warnings import warn
from threading import RLock
from src import pseudo_server 

class EventContext(t.NamedTuple):
  event_type: str
  event_target: 'pseudo_server.Server' = None
  event_manager: "EventManager" = None

def new_event_thread_worker(queue: queue.Queue) -> t.Callable:
  def worker():
    while True:
      listener, context, event_param = queue.get()

      try:
        errno = listener(context, event_param=event_param)
        if errno != 0:
          raise RuntimeError(f'custom errno: {errno}')

      except:
        traceback.print_exc(file=sys.stdout)

      finally:
        queue.task_done()

  return worker

class Event:
  _lock: RLock = RLock()
 
  def __init__(self, event_type):
    self._event_type = event_type
    self._event_listeners = []

  @property
  def listeners(self):
    with self._lock:
      return self._listeners.copy()

  @property
  def event_type(self):
    return self._event_type

  def add_listener(self, listener):
    with self._lock:
      self._event_listeners.append(listener)

  def run(self, queue, context, event_param):
    with self._lock:
      for listener in self._event_listeners:
        queue.put((listener, context, event_param))


class EventManager:
  _event_prototype = Event
  _context_prototype = EventContext
  _lock: RLock = RLock()

  def __init__(self, event_target, job_queue=None):
    self._event_target = event_target
    self._job_queue: queue.Queue = job_queue
    self._event_pool: t.Mapping[t.Hashable, self._event_prototype] = {}

  def set_queue(self, job_queue):
    self._job_queue = job_queue

  @staticmethod
  def new_event(event_type):
    return EventManager._event_prototype(event_type)

  def attach_event(self, event: Event):
    self._event_pool[event.event_type] = event

  def get_event(self, event_type):
    with self._lock:
      return self._event_pool.get(event_type)

  def trigger_event(self, event_type, event_param=None):
    with self._lock:
      event = self.get_event(event_type)
      if event is None:
        warn('{} event type does not exist'.format(event_type))
        return
      event.run(
        self._job_queue,
        self._context_prototype(event_type, self._event_target, self),
        event_param
      )