import time
import typing as t
from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from threading import RLock
from src.trace import trace_fn
from src.logger_helper import spawn_logger

_logger = spawn_logger(__name__, f'log/{__name__}.log')

T_TTL = t.Union[int, float]
T_SIZE = int

CacheRecord = namedtuple('CacheRecord', field_names=['ttl', 'size', 'value'])

class Cache:
  _cache: Mapping[t.Hashable, CacheRecord]
  _lock: RLock
  _timer: t.Callable

  # @trace_fn(_logger)
  def __init__(
    self,
    maxsize: t.Optional[T_SIZE] = 1024,
    maxage: t.Optional[T_TTL] = 0,
    timer: t.Optional[t.Callable] = time.time
  ):
    self._cache = OrderedDict()
    self._lock = RLock()

    self._timer = None
    self._maxsize = 0
    self._maxage = 0

    self.configure(
      timer=timer,
      maxsize=maxsize,
      maxage=maxage
    )

  def configure(
    self,
    maxsize: T_SIZE = 1024,
    maxage: T_TTL = 0,
    timer: t.Optional[t.Callable] = time.time
  ):
    self._timer = timer
    self.maxsize = maxsize
    self.maxage = maxage

  # @trace_fn(_logger)
  def __repr__(self):
    return f'<{self.__class__.__name__}>'

  # @trace_fn(_logger)
  def __contains__(self, key):
    return self._has(key)

  # @trace_fn(_logger)
  def __iter__(self):
    yield from self.keys()

  # @trace_fn(_logger)
  def __next__(self) -> t.Hashable:
    return next(iter(self._cache))

  @property
  def maxsize(self):
    return self._maxsize

  @maxsize.setter
  def maxsize(self, o: int):
    if o < 0:
      raise ValueError
    self._maxsize = o

  @property
  def maxage(self):
    return self._maxage

  @maxage.setter
  def maxage(self, o):
    if o < 0:
      raise ValueError
    self._maxage = o

  @property
  def usage(self):
    with self._lock:
      sums = 0
      for _, s, _ in self._cache.values():
        sums += s
      return sums

  @property
  def is_full(self):
    if self.maxsize <= 0 or self.maxsize is None:
      return False
    return self.usage >= self.maxsize 

  def copy(self):
    with self._lock:
      return self._cache.copy()

  def keys(self):
    return self.copy().keys()

  def values(self):
    return self.copy().values()

  def items(self):
    return self.copy().items()

  def clear(self):
    with self._lock:
      self._clear()

  # @trace_fn(_logger)
  def _clear(self):
    self._cache.clear()

  def has(self, key):
    with self._lock:
      self._has(key)

  # @trace_fn(_logger)
  def _has(self, key):
    self._get(key, default=None) is not None

  def get(self, key, default=None):
    with self._lock:
      return self._get(key, default=default)

  # @trace_fn(_logger)
  def _get(self, key, default=None):
    try:
      val = self._cache[key]
      if self.has_expired(key):
        self.delete(key)
        raise KeyError()
    except KeyError:
      return None
    return val

  def add(self, key: t.Hashable, value: t.Any, size: t.Optional[T_SIZE] = 1, ttl: t.Optional[T_TTL] = None):
    with self._lock:
      self._add(key, value, size=size, ttl=ttl)

  # @trace_fn(_logger)
  def _add(self, key: t.Hashable, value: t.Any, size: t.Optional[T_SIZE] = 1, ttl: t.Optional[T_TTL] = None):
    if self._has(key):
      return
    self._set(key, value, size=size, ttl=ttl)

  def set(self, key: t.Hashable, value: t.Any, size: t.Optional[T_SIZE] = 1, ttl: t.Optional[T_TTL] = None):
    with self._lock:
      self._set(key, value, size=size, ttl=ttl)

  # @trace_fn(_logger)
  def _set(self, key: t.Hashable, value: t.Any, size: t.Optional[T_SIZE] = 1, ttl: t.Optional[T_TTL] = None):
    if ttl is None:
      ttl = self._maxage

    if ttl and ttl > 0:
      ttl += self._timer() + ttl

    if not self._has(key):
      self.evict()

    self._delete(key)
    self._cache[key] = CacheRecord(ttl, size, value)

  def delete(self, key: t.Hashable):
    with self._lock:
      self._delete(key)

  # @trace_fn(_logger)
  def _delete(self, key: t.Hashable):
    try:
      del self._cache[key]
    except KeyError:
      pass

  def delete_expired(self):
    with self._lock:
      self._delete_expired()

  # @trace_fn(_logger)
  def _delete_expired(self):
    timestamp = self._timer()
    for key, (ttl, _, _) in self._cache.items():
      if ttl <= timestamp:
        self._delete(key)

  def has_expired(self, key: t.Hashable):
    with self._lock:
      self._has_expired(key)

  # @trace_fn(_logger)
  def _has_expired(self, key: t.Hashable):
    timestamp = self._timer()
    cache_record = self._cache[key]
    return cache_record.ttl <= timestamp

  def evict(self):
    self.delete_expired()

  def pop(self, key: t.Optional[t.Hashable] = None):
    with self._lock:
      self._delete_expired()
      return self._pop(key)

  # @trace_fn(_logger)
  def _pop(self, key: t.Optional[t.Hashable] = None):
    if key is None:
      key = next(self)
    key, (_, _, value) = self._cache[key]
    self._delete(key)
    return (key, value)
