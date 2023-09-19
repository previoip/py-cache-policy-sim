from dataclasses import dataclass
from src.node import Node
from src.cache import Cache, T_TTL, T_SIZE
from src.pseudo_database import ABCPseudoDatabase
from src.event import EventManager


@dataclass
class ServerConfig:
  cache_maxsize: T_SIZE = 1024
  cache_maxage: T_TTL   = 0


class Server(Node):
  
  def __init__(self, name=None, parent=None):
    super().__init__(name=name, parent=parent)
    self.cfg: ServerConfig = ServerConfig()
    self.cache: Cache = Cache()
    self.event_manager = EventManager()
    self.database: ABCPseudoDatabase = None
    self.request_buf = None
    self.response_buf = None
    self.setup()

  def setup(self):
    self.cache.configure(
      maxsize=self.cfg.cache_maxsize,
      maxage=self.cfg.cache_maxage
    )
