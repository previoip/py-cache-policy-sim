from src.pseudo_server import Server
from src.data_examples.ml_data_loader import ExampleDataLoader
from src.pseudo_database import PandasDataFramePDB
from src.event import event_thread_worker_sleep_controller
from src.model.randomized_svd import RandomizedSVD
# from src.pseudo_timer import PseudoTimer
import time
import tqdm
import queue
import threading
import numpy as np

sim_conf = {
  'general': {
    'rand_seed': 1337,
    'trial_cutoff': 10000,
    'dump_logs': True,
    'round_at_n_iter': 1000,
  },

  'loader_conf': {
    'item_key'  : 'movie_id',
    'user_key'  : 'user_id',
    'value_key' : 'rating'
  },
 
  'network_conf': {
    'num_edge': 5, # number of user-end (edge/client) groups, previously `num_cl`
    'cache_ttl': 99999,
    'base_server_alloc_frac': .8,
    'edge_server_alloc_frac': .2,
  }
}


def prepare_request_df(data_loader, sort_by='unix_timestamp'):
  user_key = sim_conf['loader_conf']['user_key']
  item_key = sim_conf['loader_conf']['item_key']
  value_key = sim_conf['loader_conf']['value_key']
  df_req = data_loader.df_ratings.sort_values(by=sort_by)[
    [user_key, item_key, value_key, sort_by]
  ].copy()
  df_req[[item_key]] = df_req[[item_key]].astype('int')
  return df_req

def iter_requests(request_df):
  for row, (_index, record) in enumerate(request_df.iterrows()):
    user_id, item_id, value, timestamp = record
    yield row, _index, user_id, item_id, value, timestamp

def prepare_movie_df(data_loader):
  item_key = sim_conf['loader_conf']['item_key']
  df_movies = data_loader.df_movies[[item_key]].copy()
  df_movies[[item_key]] = df_movies[[item_key]].astype('int')
  df_movies['sizeof'] = 1
  df_movies.set_index(item_key, drop=False, inplace=True)
  return df_movies

def prepare_user_partition(data_loader, n_indices):
  user_key = sim_conf['loader_conf']['user_key']
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

  # ================================================
  # bootstrap
  
  from bootstrap import set_default_event, EventParamContentRequest

  np.random.seed(sim_conf['general']['rand_seed'])


  # pseudo_timer = PseudoTimer()

  data_loader = ExampleDataLoader()
  data_loader.download()
  data_loader.load()

  item_df = prepare_movie_df(data_loader)
  item_total_size = item_df['sizeof'].sum()

  edge_users_partition = prepare_user_partition(data_loader, sim_conf['network_conf']['num_edge'])

  # ================================================
  # server sim setup

  base_server = Server('base_server')
  base_server.set_database(PandasDataFramePDB('movie_db', item_df))
  base_server.set_timer(lambda: 0)
  base_server.cfg.cache_maxsize = item_total_size * sim_conf['network_conf']['base_server_alloc_frac']
  base_server.cfg.cache_maxage = sim_conf['network_conf']['cache_ttl']
  base_server.setup()
  set_default_event(base_server.event_manager)

  for n in range(sim_conf['network_conf']['num_edge']):
    edge_server = base_server.spawn_child(f'edge_server_{n}')
    edge_server.set_timer(lambda: 0)
    edge_server.cfg.cache_maxsize = item_total_size * sim_conf['network_conf']['edge_server_alloc_frac']
    edge_server.cfg.cache_maxage = sim_conf['network_conf']['cache_ttl']
    edge_server.setup()
    set_default_event(edge_server.event_manager)

  user_to_edge_server_map = group_partition(edge_users_partition, base_server.children)

  # ================================================
  # begin

  print()
  print('='*48)
  print()
  print('starting sim')
  print()

  _round_at_n_iter = sim_conf['general']['round_at_n_iter']
  _max_req = sim_conf['general']['trial_cutoff']
  _it_request = iter_requests(prepare_request_df(data_loader))
  _tqdm_it_request = tqdm.tqdm(_it_request, total=_max_req, ascii=True)

  for server in base_server.recurse_nodes():
    server.run_thread()

  for row, _index, user_id, item_id, value, timestamp in _tqdm_it_request:
    if _max_req != 0 and row >= _max_req: break
  
    _tqdm_it_request.set_description(f'performing request: user_id:{user_id} item_id:{item_id}'.ljust(55))

    edge_server = user_to_edge_server_map.get(user_id)
    content_request_param = EventParamContentRequest(edge_server, timestamp, user_id, item_id, 1)
    edge_server.event_manager.trigger_event('OnContentRequest', event_param=content_request_param)

    if row % _round_at_n_iter == 0 and row != 0:
      # round checkpoint routine

      _tqdm_it_request.set_description(f'pausing... round ckpt: {(row + 1) // _round_at_n_iter}'.ljust(55))
      event_thread_worker_sleep_controller.set()

      for server in base_server.recurse_nodes():
        server.block_until_finished()

      # todo: train recsys/fl model on main thread
      # todo: invoke cache subroutine based on recommended contents

      time.sleep(1)
      event_thread_worker_sleep_controller.clear()

  print('waiting for processes to finish...')
  for server in base_server.recurse_nodes():
    server.block_until_finished()

  print()
  print('='*48)
  print()
  print('sim finished, results:')

  print()
  print('cache usage statistics')
  for server in base_server.recurse_nodes():
    print('\t-', server, server.request_log_database, f'| cache usage {server.cache.frac_usage*100:.02f}%', end='')
    print(f' | hit ratio {server.states.hit_ratio():.02f}')

  if sim_conf['general']['dump_logs']:
    print()
    print('dumping request logs...')
    for server in base_server.recurse_nodes():
      with open('log/req_log_' + server.name + '.csv', 'w') as fo:
        server.request_log_database.dump(fo)
      with open('log/req_stat_log_' + server.name + '.csv', 'w') as fo:
        server.request_status_log_database.dump(fo)

  print()
  print('mermaid graph repr:')
  for edges in base_server.recurse_edges():
    print(edges.graph_repr())