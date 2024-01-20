import os
import tqdm
import argparse
import typing as t
import numpy as np
import pandas as pd
import time
from collections import namedtuple
from src.data_loaders.ml_data_loader import ExampleDataLoader
from src.model.daisy_monkeypatch import RECSYS_MODEL_ENUM, model_dict as recsys_model_dict


class OPT_SIM_MODES:
  cache_aside = 'cache_aside'
  centralized = 'centralized'
  localized   = 'localized'
  federated   = 'federated'


class config_c:
  rand_seed           = 1337
  cutoff              = -1
  log_folder          = './log'
  log_file_fmt        = 'log_{}.csv'
  mode                = OPT_SIM_MODES.cache_aside

  model_name          = ''
  model_train_rounds  = 10

  @classmethod
  def verify(cls):
    if cls.mode == OPT_SIM_MODES.cache_aside:
      cls.model_name = None

  def __repr__(self):
    return '  mode\t: {0.mode}\n  model\t: {0.model_name}'.format(self)


# ====================================
# helpers

def pubvars(o) -> dict:
  """ Fetch public attributes and methods of class or instance inclusive as dictionary. """
  return dict(((k, i) for k, i in vars(o).items() if not k.startswith('_')))

def strvars(o):
  """ Fetch public attributes with string values as dictionary. """
  return dict(((k, i) for k, i in pubvars(o).items() if isinstance(i, str) and not i.startswith('_')))

def argparse_build(namespace) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument('--rand-seed', type=int, dest='rand_seed', default=namespace.rand_seed)
  parser.add_argument('--cutoff', type=int, dest='cutoff', default=namespace.cutoff)
  parser.add_argument('--mode', type=str, dest='mode', default=namespace.mode, choices=strvars(OPT_SIM_MODES).values())
  parser.add_argument('--model', type=str, dest='model_name', default=namespace.model_name, choices=strvars(RECSYS_MODEL_ENUM).values())
  parser.add_argument('--model-train-rounds', type=int, dest='model_train_rounds', default=namespace.model_train_rounds)
  return parser

def prepare_request_df(data_loader, sort_by='unix_timestamp') -> pd.DataFrame:
  """ Creates preconfigured view dataframe sorted based on unix timestamp. """
  user_key = data_loader.uid
  item_key = data_loader.iid
  value_key = data_loader.inter
  df_req = data_loader.df_ratings.sort_values(by=sort_by)[
    [user_key, item_key, value_key, sort_by]
  ].copy()
  df_req[[item_key]] = df_req[[item_key]].astype('uint32')
  return df_req

class RequestRecord(t.NamedTuple):
  row   : int
  index : int
  uid   : int
  iid   : int
  val   : float
  ts    : int

def iter_requests(request_df, max_n=-1) -> t.Generator[RequestRecord, None, None]:
  """ Wraps around request_df into iterable RequestRecord struct. """
  for row, record in enumerate(request_df.itertuples()):
    if max_n >= 0 and row >= max_n:
      break
    _index, user_id, item_id, value, timestamp = record
    yield RequestRecord(row, _index, user_id, item_id, value, timestamp)

def prepare_item_df(data_loader) -> pd.DataFrame:
  item_key = data_loader.iid
  df_movies = data_loader.df_movies[[item_key]].copy()
  df_movies[[item_key]] = df_movies[[item_key]].astype('int')
  df_movies['sizeof'] = 1
  df_movies.set_index(item_key, drop=False, inplace=True)
  return df_movies

def prepare_user_partition(data_loader, n_indices):
  user_key = data_loader.uid
  user_values = data_loader.df_users[user_key].unique()
  np.random.shuffle(user_values)
  return np.array_split(user_values, n_indices)

def group_partition(partition, object_list):
  if not len(partition) == len(object_list):
    raise ValueError('partition length is not equal to object_list length')
  ret = {}
  for n, items in enumerate(partition):
    for item in items:
      ret[item] = object_list[n]
  return ret

if __name__ == '__main__':

  # parse args into config namespace
  config = config_c()
  parser = argparse_build(namespace=config)
  parser.parse_args(namespace=config)
  del parser

  config.verify()
  print('{:-^36}'.format(' sim benchmark '))
  print(config)
  print('{:-^36}'.format(''))
  # process data loader
  data_loader = ExampleDataLoader()
  data_loader.default_setup()

  item_df = prepare_item_df(data_loader)

  num_request = config.cutoff if config.cutoff > -1 else data_loader.nrow
  request_df = prepare_request_df(data_loader)
  request_it = iter_requests(request_df, num_request)
  request_tq = tqdm.tqdm(request_it, total=num_request, ascii=True)

  recsys_model = recsys_model_dict.get(config.model_name)
  for request_record in request_tq:
    pass

  del recsys_model



