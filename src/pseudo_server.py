import queue
import threading
import time
from dataclasses import dataclass
from src.node import Node
from src.cache import Cache, T_TTL, T_SIZE
from src.pseudo_database import ABCPseudoDatabase, RequestLogDatabase
from src.event import EventManager, new_event_thread_worker


@dataclass
class ServerConfig:
  cache_maxsize: T_SIZE = 1024
  cache_maxage: T_TTL   = 0
  cache_timer_func = time.time 
  job_queue_maxsize: int = 0


class Server(Node):

  def __init__(self, name=None, parent=None, job_queue=None):
    super().__init__(name=name, parent=parent)

    self.cfg: ServerConfig = ServerConfig()
    self._job_queue: queue.Queue = job_queue
    self._cache: Cache = None
    self._event_manager: EventManager = None
    self._database: ABCPseudoDatabase = None
    self._request_log_database = None
    self._worker: t.Callable = None
    self._thread: threading.Thread = None

  @property
  def job_queue(self):
    return self._job_queue

  @property
  def event_manager(self):
    return self._event_manager

  @property
  def cache(self):
    return self._cache

  @property
  def database(self):
    return self._database

  @property
  def request_log_database(self):
    return self._request_log_database

  def set_database(self, database: ABCPseudoDatabase):
    self._database = database

  def setup(self):
    self._request_log_database = RequestLogDatabase(f'{self.name}', container=list())
    self._cache = Cache()
    self._cache.configure(
      maxsize=self.cfg.cache_maxsize,
      maxage=self.cfg.cache_maxage,
      timer=self.cfg.cache_timer_func,
    )
    self._job_queue = queue.Queue(
      self.cfg.job_queue_maxsize
    )
    self._event_manager = EventManager(
      event_target=self,
      job_queue=self._job_queue
    )
    self._worker = new_event_thread_worker(self._job_queue)
    self._thread = threading.Thread(target=self._worker, daemon=True)

  def run_thread(self):
    self._thread.start()

  def block_until_finished(self):
    self._job_queue.join()