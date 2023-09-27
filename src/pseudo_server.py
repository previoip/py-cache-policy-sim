from dataclasses import dataclass
import queue
from src.node import Node
from src.cache import Cache, T_TTL, T_SIZE
from src.pseudo_database import ABCPseudoDatabase
from src.event import EventManager


@dataclass
class ServerConfig:
  cache_maxsize: T_SIZE = 1024
  cache_maxage: T_TTL   = 0


class Server(Node):

  def __init__(self, name=None, parent=None, job_scheduler=None):
    super().__init__(name=name, parent=parent)

    self.cfg: ServerConfig = ServerConfig()
    self._job_scheduler: queue.Queue = job_scheduler
    self._cache: Cache = None
    self._event_manager: EventManager = None
    self._database: ABCPseudoDatabase = None

  @property
  def event_manager(self):
    return self._event_manager

  @property
  def cache(self):
    return self._cache

  @property
  def database(self):
    return self._database

  def setup(self):
    self._cache = Cache()
    self._cache.configure(
      maxsize=self.cfg.cache_maxsize,
      maxage=self.cfg.cache_maxage
    )
    self._event_manager = EventManager(
      event_target=self,
      job_scheduler=self._job_scheduler
    )

  def set_database(self, database: ABCPseudoDatabase):
    self._database = database