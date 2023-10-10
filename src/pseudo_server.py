import queue
import threading
import time
from dataclasses import dataclass
from src.node import Node
from src.cache import Cache, T_TTL, T_SIZE
from src.pseudo_database import ABCPseudoDatabase, TabularPDB
from src.event import EventManager, new_event_thread_worker
from src.model.model_abc import ABCRecSysModel

@dataclass
class ServerConfig:
  cache_maxsize: T_SIZE = 1024
  cache_maxage: T_TTL   = 0
  job_queue_maxsize: int = 0

@dataclass
class ServerStates:
  request_counter: int = 0
  cache_hit_counter: int = 0
  cache_miss_counter: int = 0

  def hit_ratio(self):
    return self.cache_hit_counter / self.request_counter


class ServerBuffers:
  def __init__(self):
    self.response = queue.Queue()
    self.uncached_content = queue.Queue()


class Server(Node):

  def __init__(self, name=None, parent=None, job_queue=None):
    super().__init__(name=name, parent=parent)

    self.cfg: ServerConfig = ServerConfig()
    self._server_states: ServerStates = ServerStates()
    self._server_states_lock: threading.RLock = threading.RLock()
    self._job_queue: queue.Queue = job_queue
    self._buffers: ServerBuffers = None
    self._cache: Cache = None
    self._event_manager: EventManager = None
    self._database: ABCPseudoDatabase = None
    self._request_log_database = None
    self._request_status_log_database = None
    self._timer: t.Callable = time.time
    self._worker: t.Callable = None
    self._thread: threading.Thread = None
    self._model: ABCRecSysModel = None

  @property
  def states(self):
    with self._server_states_lock:
      return self._server_states

  @property
  def buffers(self):
    return self._buffers

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

  @property
  def request_status_log_database(self):
    return self._request_status_log_database

  @property
  def timer(self):
    return self._timer

  @property
  def model(self):
    return self._model

  def has_database(self):
    return not self._database is None

  def set_jog_queue(self, job_queue):
    self._job_queue = job_queue

  def set_thread(self, thread):
    self._thread = thread

  def set_timer(self, timer):
    self._timer = timer

  def set_database(self, database: ABCPseudoDatabase):
    self._database = database

  def set_model(self, model: ABCRecSysModel):
    self._model = model

  def setup(self):
    self._buffers = ServerBuffers()
    self._request_log_database = TabularPDB(
      f'{self.name}',
      container=list(),
      field_names=['timestamp', 'user_id', 'item_id']
    )
    self._request_status_log_database = TabularPDB(
      f'{self.name}',
      container=list(),
      field_names=['timestamp', 'user_id', 'item_id', 'status']
    )
    self._cache = Cache()
    self._cache.configure(
      maxsize=self.cfg.cache_maxsize,
      maxage=self.cfg.cache_maxage,
      timer=self._timer,
    )
    self._job_queue = queue.Queue(
      maxsize=self.cfg.job_queue_maxsize
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