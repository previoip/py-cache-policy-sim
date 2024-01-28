import queue
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from src.node import Node
from src.cache import Cache, T_TTL, T_SIZE
from src.pseudo_database import ABCPseudoDatabase, TabularPDB
from src.event import EventManager, new_event_thread_worker
from src.model.daisyRec.daisy.model.AbstractRecommender import AbstractRecommender
from src.model.mp_runner import RecsysFL

@dataclass
class ServerConfig:
  cache_maxsize: T_SIZE             = 1024
  cache_maxage: T_TTL               = 0
  model_config: dict                = field(default_factory=dict)
  db_req_log_fieldnames: list       = field(default_factory=lambda: ['request_id', 'timestamp', 'user_id', 'movie_id', 'rating'])
  db_req_log_fieldtypes: list       = field(default_factory=lambda: ['uint32', 'int64', 'uint32', 'uint32', 'float'])
  db_req_stat_log_fieldnames: list  = field(default_factory=lambda: ['request_id', 'timestamp', 'user_id', 'movie_id', 'rating', 'status'])
  db_req_stat_log_fieldtypes: list  = field(default_factory=lambda: ['uint32', 'int64', 'uint32', 'uint32', 'float', 'string'])
  flag_suppress_cache_on_req: bool  = False


@dataclass
class ServerStates:
  request_counter: int = 0
  cache_hit_counter: int = 0
  cache_miss_counter: int = 0

  def hit_ratio(self):
    if self.request_counter == 0:
      return 0
    return self.cache_hit_counter / self.request_counter

  def __repr__(self):
    return "total:{}\nhit:{}\nmiss:{}\n".format(self.request_counter, self.cache_hit_counter, self.cache_miss_counter)


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
    self._worker: t.Callable = None
    self._event_manager: EventManager = None

    self._timer: t.Callable = time.time
    self._buffers: ServerBuffers = None
    self._database: ABCPseudoDatabase = None
    self._request_log_database = None
    self._request_status_log_database = None
    self._cache: Cache = None
    self._recsys_runner: RecsysFL.ModelRunner = None
    self.cache_policy = None

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
  def recsys_runner(self) -> RecsysFL.ModelRunner:
    return self._recsys_runner

  def has_database(self):
    return not self._database is None

  def set_timer(self, timer):
    self._timer = timer

  def set_database(self, database: ABCPseudoDatabase):
    self._database = database

  def set_recsys_runner(self, model: AbstractRecommender):
    self._recsys_runner = model

  def set_cache_policy(self, policy_net):
    self.set_cache_policy = policy_net

  def setup(self, queue):
    self._buffers = ServerBuffers()
    self._request_log_database = TabularPDB(
      f'{self.name}',
      container=list(),
      field_names=self.cfg.db_req_log_fieldnames,
      field_dtypes=self.cfg.db_req_log_fieldtypes
    )
    self._request_status_log_database = TabularPDB(
      f'{self.name}',
      container=list(),
      field_names=self.cfg.db_req_stat_log_fieldnames,
      field_dtypes=self.cfg.db_req_stat_log_fieldtypes
    )
    self._cache = Cache()
    self._cache.configure(
      maxsize=self.cfg.cache_maxsize,
      maxage=self.cfg.cache_maxage,
      timer=self._timer,
    )
    self._event_manager = EventManager(
      event_target=self,
      job_queue=queue
    )
