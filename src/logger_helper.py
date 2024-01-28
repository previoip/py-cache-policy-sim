import logging
import os

__LOG_FMT_STR = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
__LOG_FMT = logging.Formatter(__LOG_FMT_STR)

def get_fmt() -> logging.Formatter:
  return __LOG_FMT

def set_fmt(logger: logging.Logger):
  for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler) or \
    isinstance(handler, logging.FileHandler):
      handler.setFormatter(__LOG_FMT)

def spawn_logger(name, log_path=None, level=logging.DEBUG, as_fstream=False) -> logging.Logger:
  logger = logging.getLogger(name)
  logger.setLevel(level)

  if not log_path is None:
    dirname = os.path.dirname(log_path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
  
  if as_fstream:
    for handler in logger.handlers:
      logger.removeHandler(handler)
    ch = logging.FileHandler(log_path)
    logger.addHandler(ch)

  elif not logger.hasHandlers():
    print('logger does not have handlers')
    if log_path is None:
      ch = logging.StreamHandler()
    else:
      ch = logging.FileHandler(log_path)
    logger.addHandler(ch)

  for handler in logger.handlers:
    print('logger handlers:', handler)

  set_fmt(logger)

  return logger

def init_default_logger() -> logging.Logger:
  logger = logging.getLogger()
  logger.addHandler(logging.StreamHandler())
  logger.addHandler(logging._StderrHandler(logging.ERROR))
  logger.setLevel(logging.DEBUG)
  set_fmt(logger)
  return logger