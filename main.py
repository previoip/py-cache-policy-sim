from src.pseudo_server import Server
from src.data_examples.ml_data_loader import ExampleDataLoader
from src.pseudo_database import PandasDataFramePDB
from src.event import event_thread_worker_sleep_controller
# from src.model.randomized_svd import RandomizedSVD
from src.model.daisy_monkeypatch import RECSYS_MODEL_ENUM
import os
import json
import time
import tqdm
import queue
import threading
import numpy as np

CONF_HIST_FILENAME = 'hist.json'
SAVE_HIST = True
PRUNE_HIST = True

sim_conf = {
  'conf_name': 'cache_aside_test',
  'conf_prfx': 'cat',
  'conf_ts': time.strftime('%Y-%m-%dT%H%M'),

  'general': {
    'rand_seed': 1337,
    'recsys_model_name': RECSYS_MODEL_ENUM.mf,
    'trial_cutoff': 10000,
    'dump_logs': True,
    'round_at_n_iter': 5000,
    'log_folder': './log',
    'filename_template_log_req': 'log_request_{}.csv',
    'filename_template_log_req_stat': 'log_request_stat_{}.csv',
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

def inject_daisy_config(data_loader, daisy_config):
  user_key = sim_conf['loader_conf']['user_key']
  user_values = data_loader.df_users[user_key].unique()
  daisy_config['user_num'] = len(user_values)
  daisy_config['item_num'] = len(prepare_movie_df(data_loader))


if __name__ == '__main__':

  # ================================================
  # bootstrap
  
  from bootstrap import set_default_event, EventParamContentRequest

  data_loader = ExampleDataLoader()
  data_loader.download()
  data_loader.load()

  sim_conf['loader_conf']['user_key'] = data_loader.uid
  sim_conf['loader_conf']['item_key'] = data_loader.iid
  sim_conf['loader_conf']['value_key'] = data_loader.inter

  item_df = prepare_movie_df(data_loader)
  item_total_size = item_df['sizeof'].sum()

  edge_users_partition = prepare_user_partition(data_loader, sim_conf['network_conf']['num_edge'])

  # ================================================
  # daisy model inits

  from src.model.daisyRec.daisy.utils.config import init_seed
  from src.model.daisy_monkeypatch import init_config, build_model_constructor

  daisy_config = init_config(data_loader, algo_name=sim_conf['general']['recsys_model_name'])

  daisy_config['seed'] = sim_conf['general']['rand_seed']
  init_seed(daisy_config['seed'], True)

  inject_daisy_config(data_loader, daisy_config)

  model_constructor = build_model_constructor(daisy_config)

  model = model_constructor()
  print(model)
  exit()

  sim_conf.update({'daisy_config': daisy_config})

  # ================================================
  # server sim setup

  base_server = Server('base_server')
  base_server.set_database(PandasDataFramePDB('movie_db', item_df))
  base_server.set_timer(lambda: 0)
  base_server.set_model(model_constructor())
  base_server.cfg.cache_maxsize = item_total_size * sim_conf['network_conf']['base_server_alloc_frac']
  base_server.cfg.cache_maxage = sim_conf['network_conf']['cache_ttl']

  for n in range(sim_conf['network_conf']['num_edge']):
    edge_server = base_server.spawn_child(f'edge_server_{n}')
    edge_server.set_timer(lambda: 0)
    edge_server.set_model(model_constructor())
    edge_server.cfg.cache_maxsize = item_total_size * sim_conf['network_conf']['edge_server_alloc_frac']
    edge_server.cfg.cache_maxage = sim_conf['network_conf']['cache_ttl']

  for server in base_server.recurse_nodes():
    server.setup()
    set_default_event(server.event_manager)

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

  # ================================================
  # finishing

  print()
  print('='*48)
  print()
  print('sim finished, results:')

  sim_conf['results'] = {}

  print()
  print('cache usage statistics')

  hit_ratios = []
  for server in base_server.recurse_nodes():
    print('\t-', server, server.request_log_database, f'| cache usage {server.cache.frac_usage*100:.02f}%', end='')
    print(f' | hit ratio {server.states.hit_ratio():.02f}')
    hit_ratios.append({'server': server.name, 'values': {'hit': server.states.hit_ratio()}})

  sim_conf['results'].update({'hit_ratios': hit_ratios})

  if sim_conf['general']['dump_logs']:
    print()
    print('dumping request logs...')

    log_folder = sim_conf['general']['log_folder']
    if not os.path.exists(log_folder) and not os.path.isdir(log_folder):
      os.makedirs(log_folder)
    filename_template_log_req = sim_conf['general']['filename_template_log_req'] 
    filename_template_log_req_stat = sim_conf['general']['filename_template_log_req_stat'] 
    filename_template_log_req = sim_conf['conf_ts'] + '_' + filename_template_log_req
    filename_template_log_req_stat = sim_conf['conf_ts'] + '_' + filename_template_log_req_stat
    filename_template_log_req = sim_conf['conf_prfx'] + '_' + filename_template_log_req
    filename_template_log_req_stat = sim_conf['conf_prfx'] + '_' + filename_template_log_req_stat
    
    log_files = []

    for server in base_server.recurse_nodes():
      filename_log_req = filename_template_log_req.format(server.name)
      filename_log_req_stat = filename_template_log_req_stat.format(server.name)
      with open(log_folder + '/' + filename_log_req , 'w') as fo:
        server.request_log_database.dump(fo)
        log_files.append({'server': server.name, 'type': 'request', 'fp': filename_log_req})
      with open(log_folder + '/' + filename_log_req_stat, 'w') as fo:
        server.request_status_log_database.dump(fo)
        log_files.append({'server': server.name, 'type': 'request_stats', 'fp': filename_log_req_stat})

    sim_conf['results'].update({'log_files': log_files})

  print()
  print('mermaid graph repr:')
  for edges in base_server.recurse_edges():
    print(edges.graph_repr())

  if PRUNE_HIST:
    if os.path.exists(CONF_HIST_FILENAME) and \
      os.path.isfile(CONF_HIST_FILENAME):
      with open(CONF_HIST_FILENAME, 'w') as fo:
        fo.write('[]')

  if SAVE_HIST:
    if not os.path.exists(CONF_HIST_FILENAME) and \
      not os.path.isfile(CONF_HIST_FILENAME):
      with open(CONF_HIST_FILENAME, 'w') as fo:
        fo.write('[]')
    with open(CONF_HIST_FILENAME, 'r') as fo:
      conf_hist = json.load(fo)
    conf_hist.append(sim_conf)
    with open(CONF_HIST_FILENAME, 'w') as fo:
      json.dump(conf_hist, fo, indent=2)
