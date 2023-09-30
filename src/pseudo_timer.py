from threading import RLock

class PseudoTimer:
  _time: int = 0
  _lock: RLock = RLock()

  def set_time(self, time):
    with self._lock:
      self._time = time

  def get_time(self):
    with self._lock:
      return self._time