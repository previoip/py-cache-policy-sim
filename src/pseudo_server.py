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
  queue_maxsize: int = 0


@dataclass
class ServerFlags:
  worker_interrupt: bool = False

@dataclass
class ServerBuffers:
  response_buf = None
  request_buf = None


class Server(Node):
  
  __queue_prototype = queue.LifoQueue

  def __init__(self, name=None, parent=None):
    super().__init__(name=name, parent=parent)

    self.flags: ServerFlags()
    self.cfg: ServerConfig = ServerConfig()
    self.cache: Cache = Cache()
    self.event_manager: EventManager = EventManager()
    self.database: ABCPseudoDatabase = None
    self.job_queue: queue.Queue = None
    self.buf = ServerBuffers()
    self.setup()

  def setup(self):
    self.job_queue = self.__queue_prototype(maxsize=self.cfg.queue_maxsize)
    self.cache.configure(
      maxsize=self.cfg.cache_maxsize,
      maxage=self.cfg.cache_maxage
    )

  def spawn_worker(self):
    def worker(self):
      while self.flags.worker_interrupt:
        task = self.job_queue.get()
        ...
        self.job_queue.task_done()
    return worker
