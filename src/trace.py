from functools import wraps
from logging import Formatter

def trace_fn(logger):
  def wrapper(fn):
    @wraps(fn)
    def wfn(*args, **kwargs):
      result = fn(*args, **kwargs)
      logger.debug(f'called: {fn.__name__}')
      return result
    return wfn
  return wrapper