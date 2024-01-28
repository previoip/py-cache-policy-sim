import typing as t
import queue
import sys
import time
import traceback
from warnings import warn
from threading import RLock, Event as ThreadingEvent
from functools import wraps
from src import pseudo_server
from src.logger_helper import spawn_logger

event_thread_worker_sleep_controller = ThreadingEvent()
event_trace = spawn_logger('eventtrace', log_path='./log/eventtrace.txt', as_fstream=True)


class EventContext(t.NamedTuple):
  event_type: str
  event_target: 'pseudo_server.Server' = None
  event_manager: "EventManager" = None

def new_event_thread_worker(queue: queue.Queue) -> t.Callable:
  def worker():
    while True:
      listener, context, event_param = queue.get()
      # event_trace.debug('{}: {}'.format(context.event_type, context.event_target))
      try:
        errno = listener(context, event_param=event_param)
        if errno != 0:
          raise RuntimeError(f'custom errno: {errno}')

      except:
        traceback.print_exc(file=sys.stdout)

      finally:
        queue.task_done()

      # while event_thread_worker_sleep_controller.is_set():
      #   time.sleep(.1)

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

  @staticmethod
  def wrap_emitter_on_method_call(event_type):
    def wrapper(fn):
      @wraps(fn)
      def wfn(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if hasattr(self, 'event_manager'):
          self.event_manager.trigger_event(event_type)
        return result
      return wfn
    return wrapper