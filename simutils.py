import typing as t
import pandas as pd
import numpy as np

def pubvars(o) -> dict:
  """ Fetch public attributes and methods of class or instance inclusive as dictionary. """
  return dict(((k, i) for k, i in vars(o).items() if not k.startswith('_')))

def strvars(o):
  """ Fetch public attributes with string values as dictionary. """
  return dict(((k, i) for k, i in pubvars(o).items() if isinstance(i, str) and not i.startswith('_')))

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
  df_items = data_loader.df_items[[item_key]].copy()
  df_items[[item_key]] = df_items[[item_key]].astype('int')
  df_items['sizeof'] = 1
  df_items.set_index(item_key, drop=False, inplace=True)
  return df_items

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
