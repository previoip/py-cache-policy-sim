from functools import wraps
from logging import Formatter

log_fmt = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def trace_fn(logger):
  def wrapper(fn):
    @wraps(fn)
    def wfn(*args, **kwargs):
      result = fn(*args, **kwargs)
      logger.debug(f'called: {fn.__name__}')
      return result
    return wfn
  return wrapper