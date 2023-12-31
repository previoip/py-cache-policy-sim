from src.pseudo_server import Server
from src.data_examples.ml_data_loader import ExampleDataLoader
from src.pseudo_database import PandasDataFramePDB
from src.event import event_thread_worker_sleep_controller
from src.model.daisy_monkeypatch import RECSYS_MODEL_ENUM
from collections import Counter
from enum import Enum, auto
import os
import json
import time
import tqdm
import queue
import threading
import numpy as np
import pandas as pd
import argparse


CONF_HIST_FILENAME = 'hist.json'
SAVE_HIST = True
PRUNE_HIST = False # set to false to be able to evaluate multiple configuration

class SIM_MODE_ENUM:
  cache_aside = 'cache_aside'
  centralized = 'centralized'
  localized = 'localized'
  federated = 'federated'

sim_conf = {
  'conf_name': 'cache_aside_4div10_alloc',
  'conf_prfx': 'cat',
  'conf_ts': time.strftime('%Y-%m-%dT%H%M'),

  'general': {
    'rand_seed': 1337,
    'mode': SIM_MODE_ENUM.cache_aside,
    'recsys_model_name': RECSYS_MODEL_ENUM.mf,
    'recsys_model_topk_frac': .75,
    'trial_cutoff': 100_000,
    'dump_logs': True,
    'dump_configs': True,
    'print_mermaid_md': False,
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
    'num_edge': 3, # number of user-end (edge/client) groups, previously `num_cl`
    'cache_ttl': 99999,
    'base_server_alloc_frac': 1,
    'edge_server_alloc_frac': .4,
  }
}

def argparse_setup():
  parser = argparse.ArgumentParser()
  subargp = parser.add_subparsers()
  for conf_key in sim_conf.keys():
    conf_val = sim_conf.get(conf_key)

    if not isinstance(conf_val, dict):
      if not isinstance(conf_val, (int, float, str, bool)):
        continue
      parser.add_argument(f'--{conf_key}', default=conf_val, type=type(conf_val))

    else:
      # subparser = subargp.add_parser(conf_key)
      for subconf_key in conf_val.keys():
        if not isinstance(conf_val.get(subconf_key), (int, float, str, bool)):
          continue
        # subparser.add_argument(f'--{subconf_key}', default=conf_val.get(subconf_key))
        parser.add_argument(f'--{subconf_key}', default=conf_val.get(subconf_key), type=type(conf_val.get(subconf_key)))

  return parser

def argparse_conf_unflatten(parsed_args):
  parsed_args = vars(parsed_args)
  unflattened = dict()
  for conf_key in sim_conf.keys():
    conf_val = sim_conf.get(conf_key)
    if isinstance(conf_val, dict):
      unflattened[conf_key] = dict()
      for subconf_key in conf_val.keys():
        parsed_val = parsed_args.get(subconf_key)
        if parsed_val is None:
          parsed_val = sim_conf[conf_key].get(subconf_key)
        unflattened[conf_key][subconf_key] = parsed_val
    else:
      parsed_val = parsed_args.get(conf_key)
      if parsed_val is None:
        parsed_val = sim_conf.get(conf_key)
      unflattened[conf_key] = parsed_val
  return unflattened


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
  for row, record in enumerate(request_df.itertuples()):
    _index, user_id, item_id, value, timestamp = record
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

def update_daisy_config_user_item_num(data_loader, daisy_config):
  user_key = sim_conf['loader_conf']['user_key']
  user_values = data_loader.df_users[user_key].unique()
  daisy_config.update({'user_num': len(user_values)})
  daisy_config.update({'item_num': len(prepare_movie_df(data_loader))}) 

def override_daisy_config(daisy_config):
  # custom daiy config overrides embedded as dict
  daisy_config.update({
    "test_size": 0.0,
    'epochs': 5,
    'gpu': '', # or set to '0' if gpu is available
    # 'topk': int(len(item_df) * sim_conf['general']['recsys_model_topk_frac'])
  })
  
# pandas helper

def fn_get_reqlog_container(server):
  return server.request_log_database.get_container()  

def fn_get_all_reqlog_container(servers):
  req_log_dfs = []
  for server in servers:
    req_log = fn_get_all_reqlog_container(server)
    if req_log is None or len(req_log) == 0:
      continue
    req_log_dfs.append(req_log.to_pd())
  return req_log

# server model train runner helper

def fn_server_get_user_ids(server):
  ls_request_log_database = server.request_log_database.get_container()
  if ls_request_log_database is None:
    return []
  ls_users_ids = list(set([int(i.user_id) for i in ls_request_log_database]))
  return ls_users_ids

def fn_train_df_on_server(server, train_runner, req_log_df):
  train_set, test_set, test_ur, total_train_ur = train_runner.split(req_log_df)

  recsys_model = server.model
  recsys_config = server.cfg.model_config
  recsys_config.update({'train_ur': total_train_ur})
  runner_preset = train_runner.get_train_runner(recsys_config)
  model_runner = runner_preset()
  model_runner(recsys_model, recsys_config, train_set)



if __name__ == '__main__':

  parser = argparse_setup()
  parsed_args = parser.parse_args()

  sim_conf.update(argparse_conf_unflatten(parsed_args))
  print(sim_conf)

  sim_mode = sim_conf['general']['mode']

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
  item_total_count = len(item_df)

  edge_users_partition = prepare_user_partition(data_loader, sim_conf['network_conf']['num_edge'])

  # ================================================
  # daisy model inits

  from src.model.daisyRec.daisy.utils.config import init_seed
  from src.model.daisy_monkeypatch import init_config, build_model_constructor

  daisy_config = init_config(data_loader, algo_name=sim_conf['general']['recsys_model_name'])

  daisy_config['seed'] = sim_conf['general']['rand_seed']
  init_seed(daisy_config['seed'], True)

  update_daisy_config_user_item_num(data_loader, daisy_config)
  override_daisy_config(daisy_config)

  model_constructor, train_runner = build_model_constructor(daisy_config)

  # ================================================
  # server sim setup

  base_server = Server('base_server')
  base_server.set_database(PandasDataFramePDB('movie_db', item_df))
  base_server.set_timer(lambda: 0)
  base_server.set_model(model_constructor())
  base_server.cfg.model_config = daisy_config.copy()
  base_server.cfg.cache_maxsize = item_total_size * sim_conf['network_conf']['base_server_alloc_frac']
  base_server.cfg.cache_maxage = sim_conf['network_conf']['cache_ttl']

  for n in range(sim_conf['network_conf']['num_edge']):
    edge_server = base_server.spawn_child(f'edge_server_{n}')
    edge_server.set_timer(lambda: 0)
    edge_server.set_model(model_constructor())
    edge_server.cfg.model_config = daisy_config.copy()
    edge_server.cfg.cache_maxsize = item_total_size * sim_conf['network_conf']['edge_server_alloc_frac']
    edge_server.cfg.cache_maxage = sim_conf['network_conf']['cache_ttl']

  for server in base_server.recurse_nodes():
    server.setup()
    set_default_event(server.event_manager)

    # if server.name == 'base_server':
    #   continue

    if sim_mode != SIM_MODE_ENUM.cache_aside:
      server.cfg.flag_suppress_cache_on_req = True

  user_to_edge_server_map = group_partition(edge_users_partition, base_server.children)

  # ================================================
  # begin

  print()
  print('='*48)
  print()
  print('starting sim')
  print()

  # prepare request iterover
  _round_at_n_iter = sim_conf['general']['round_at_n_iter']
  _max_req = sim_conf['general']['trial_cutoff']
  _it_request = iter_requests(prepare_request_df(data_loader))
  _tqdm_it_request = tqdm.tqdm(_it_request, total=_max_req, ascii=True)

  # run thread per server
  # todo: use threadpool or threadexec
  for server in base_server.recurse_nodes():
    server.run_thread()

  # iterate over the request dataframe iterator
  for row, _index, user_id, item_id, value, timestamp in _tqdm_it_request:
    if _max_req != 0 and row >= _max_req: break
  
    # make sure all job queue is exhausted before continuing
    for server in base_server.recurse_nodes():
      server.block_until_finished()

    _tqdm_it_request.set_description(f'performing request: user_id:{user_id} item_id:{item_id}'.ljust(55))

    # perform request based on edge-to-user random uid mapping
    edge_server = user_to_edge_server_map.get(user_id)
    content_request_param = EventParamContentRequest(row, edge_server, timestamp, user_id, item_id, value, 1)
    edge_server.event_manager.trigger_event('OnContentRequest', event_param=content_request_param)


    if sim_mode == SIM_MODE_ENUM.cache_aside:
      # todo: additionnal for cache_aside mode tasks, 
      # note for cache_aside caching is preformed after each request
      pass

    elif row % _round_at_n_iter == 0 and row != 0:
      # round checkpoint routine

      _tqdm_it_request.set_description(f'pausing... round ckpt: {(row + 1) // _round_at_n_iter}'.ljust(55))
      for server in base_server.recurse_nodes():
        server.block_until_finished()
      event_thread_worker_sleep_controller.set()

      to_be_cached_items_by_server = dict()
      for server in base_server.recurse_nodes():
        to_be_cached_items_by_server[server.name] = list()

      # case cetralized learning
      if sim_mode == SIM_MODE_ENUM.centralized:
        servers = filter(lambda x: x.name != 'base_server' and x.is_leaf(), base_server.recurse_nodes())
        req_log_dfs = map(lambda server: server.request_log_database.to_pd(use_cursor=True), servers)
        req_log_dfs = filter(lambda df: not df is None or len(df) != 0, req_log_dfs)
        req_log_dfs = list(req_log_dfs)
        if len(req_log_dfs) == 0:
          continue
        req_log_df = pd.concat(req_log_dfs)
        # train combined request log df into base_server
        fn_train_df_on_server(base_server, train_runner, req_log_df)

        for server in base_server.recurse_nodes():
          to_be_cached_items = list()
          user_ids = fn_server_get_user_ids(server)
          for sel_user_id in user_ids:
            ranks = base_server.model.full_rank(sel_user_id)
            if isinstance(ranks, np.ndarray):
              ranks = ranks.flatten()
            ranks = list(ranks)
            to_be_cached_items.extend(ranks)
          topk_n = int(server.cfg.cache_maxsize * sim_conf['general'].get('recsys_model_topk_frac', 1))
          to_be_cached_items = Counter(to_be_cached_items).most_common(topk_n)
          to_be_cached_items = [i[0] for i in to_be_cached_items]
          to_be_cached_items_by_server[server.name].extend(to_be_cached_items)
      # esac cetralized learning

      # case localized learning
      elif sim_mode == SIM_MODE_ENUM.localized:
        servers = filter(lambda x: True, base_server.recurse_nodes())
        for server in servers:
          req_log_df = server.request_log_database.to_pd(use_cursor=True)
          if req_log_df is None or len(req_log_df) == 0:
            continue
          # train request log df on each server respectively
          fn_train_df_on_server(server, train_runner, req_log_df)

        for server in base_server.recurse_nodes():
          to_be_cached_items = list()
          user_ids = fn_server_get_user_ids(server)
          for sel_user_id in user_ids:
            ranks = server.model.full_rank(sel_user_id)
            if isinstance(ranks, np.ndarray):
              ranks = ranks.flatten()
            ranks = list(ranks)
            to_be_cached_items.extend(ranks)
          topk_n = int(server.cfg.cache_maxsize * sim_conf['general'].get('recsys_model_topk_frac', 1))
          to_be_cached_items = Counter(to_be_cached_items).most_common(topk_n)
          to_be_cached_items = [i[0] for i in to_be_cached_items]
          to_be_cached_items_by_server[server.name].extend(to_be_cached_items)
      # esac localized learning

      # case federated learning
      elif sim_mode == SIM_MODE_ENUM.federated:
        servers = filter(lambda x: x.name != 'base_server' and x.is_leaf(), base_server.recurse_nodes())
        for server in servers:
          req_log_df = server.request_log_database.to_pd(use_cursor=True)
          if req_log_df is None or len(req_log_df) == 0:
            continue
          # train request log df on each server
          fn_train_df_on_server(server, train_runner, req_log_df)

        # perform federated learning from each edge servers
        base_server.model.fl_agg(list(map(lambda server: server.model, servers)))
        base_server.model.fl_delegate_to(list(map(lambda server: server.model, servers)))

        for server in base_server.recurse_nodes():
          to_be_cached_items = list()
          user_ids = fn_server_get_user_ids(server)
          for sel_user_id in user_ids:
            ranks = server.model.full_rank(sel_user_id)
            if isinstance(ranks, np.ndarray):
              ranks = ranks.flatten()
            ranks = list(ranks)
            to_be_cached_items.extend(ranks)
          topk_n = int(server.cfg.cache_maxsize * sim_conf['general'].get('recsys_model_topk_frac', 1))
          to_be_cached_items = Counter(to_be_cached_items).most_common(topk_n)
          to_be_cached_items = [i[0] for i in to_be_cached_items]
          to_be_cached_items_by_server[server.name].extend(to_be_cached_items)
      # esac federated learning

      else:
        raise ValueError(f'sim_mode is not valid enum value')

      # distribute to be cached items to servers
      print()
      for server in base_server.recurse_nodes():
        to_be_cached_items = to_be_cached_items_by_server.get(server.name)
        print('caching item for', server.name, 'with', len(to_be_cached_items), 'items.')
        if to_be_cached_items is None or len(to_be_cached_items) == 0:
          continue
        for cache_item_id in to_be_cached_items:
          cache_param = EventParamContentRequest(
            request_id=None,
            client=server,
            timestamp=timestamp,
            user_id=user_id,
            item_id=cache_item_id,
            rating=None,
            item_size=1,
          )
          server.event_manager.trigger_event('SubCache', event_param=cache_param)

      event_thread_worker_sleep_controller.clear()

  print('waiting for all processes to finish...')
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

  server_node_depths = {}
  for node in base_server.recurse_nodes():
    server_node_depths[node.name] = node.depth + 1
  
  sim_conf['results'].update({'depths': server_node_depths})

  if sim_conf['general']['dump_configs']:

    print()
    print('dumping server configs...')

    model_configs = []
    for server in base_server.recurse_nodes():
      model_config = server.cfg.model_config
      if model_config.get('train_ur') is not None:
        model_config['train_ur'] = None
      model_configs.append({'server': server.name, 'type': 'daisy_config/json', 'value': model_config})
    if sim_mode != SIM_MODE_ENUM.cache_aside:
      sim_conf['results'].update({'model_configs': model_configs})

  if sim_conf['general']['print_mermaid_md']:
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
      json.dump(conf_hist, fo, indent=2, default=lambda o: str(o))
