import typing as t
import threading
import queue

class ThreadBroker:
  def __init__(self, queue_maxsize=0):
    self._threadpool: t.Mapping[t.Hashable, threading.Thread] = {}
    self._queue = queue.Queue(queue_maxsize)
    self._worker = None

  def set_worker(self, worker_func):
    self._worker = worker_func

  def new_thread(self, func=None, daemon=True):
    if not func is None:
      self.set_worker(func)
    thread = threading.Thread(target=self._worker, daemon=daemon)
    self._threadpool[id(thread)] = thread
    return thread

  def force_start(self):
    for thread in self._threadpool.values():
      thread.start()


