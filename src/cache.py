import time
import typing as t
from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

T_TTL = t.Union[int, float]

@dataclass
class CacheRecord:
  ttl: T_TTL
  sizeof: int
  value: t.Any

  def __iter__(self):
    return iter((self.ttl, self.val))


class Cache:
  _cache: Mapping[t.Hashable, CacheRecord]
  _lock: RLock
  _timer: t.Callable

  def __init__(
    self,
    maxsize: t.Optional[int] = 1024,
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
    maxsize: int = 1024,
    maxage: T_TTL = 0,
    timer: t.Optional[t.Callable] = time.time
  ):
    self._timer = timer
    self.maxsize = maxsize
    self.maxage = maxage

  def __repr__(self):
    return f'<{self.__class__.__name__}>'

  def __len__(self):
    with self._lock:
      return len(self._cache)

  def __contains__(self, key):
    return self._has(key)

  def __iter__(self):
    yield from self.keys()

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

  def copy(self):
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

  def _clear(self):
    self._cache.clear()

  def has(self, key):
    with self._lock:
      self._has(key)

  def _has(self, key):
    self._get(key, default=None) is not None

  def get(self, key, default=None):
    with self._lock:
      return self._get(key, default=default)

  def _get(self, key, default=None):
    try:
      val = self._cache[key]
      if self.is_expired(key):
        self.delete(key)
        raise KeyError
    except KeyError:
      return None
    return val

  def add(self, key: t.Hashable, value: t.Any, sizeof: t.Optional[int] = 1, ttl: t.Optional[T_TTL] = None):
    with self._lock:
      self._add(self, key, value, sizeof=sizeof, ttl=ttl)

  def _add(self, key: t.Hashable, value: t.Any, sizeof: t.Optional[int] = 1, ttl: t.Optional[T_TTL] = None):
    if self._has(key):
      return
    self._set(self, key, value, sizeof=sizeof, ttl=ttl)

  def set(self, key: t.Hashable, value: t.Any, sizeof: t.Optional[int] = 1, ttl: t.Optional[T_TTL] = None):
    if ttl is None:
      ttl = self._maxage

    if ttl and ttl > 0:
      ttl += self._timer() + ttl

    if not self._has(key):
      self.evict()

    self._delete(key)
    self._cache[key] = CacheRecord(ttl, sizeof, value)


  def delete(self, key: t.Hashable):
      ...

  def evict(self):
    ...
  
  def pop(self):
    ...

  def is_expired(self, key: t.Hashable):
    ...
