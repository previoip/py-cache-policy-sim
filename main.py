from queue import Queue
from config import config, build_parser, SIM_MODE_ENUM
import numpy as np
import pandas as pd
import tqdm
from collections import Counter, namedtuple
from tqdm.contrib.logging import logging_redirect_tqdm
from simutils import prepare_item_df, prepare_request_df, prepare_user_partition, iter_requests, group_partition
from src.data_loaders.ml_data_loader import ExampleDataLoader
from src.model.mp_runner import RecsysFL
from src.pseudo_database import PandasDataFramePDB
from src.pseudo_server import Server
from src.logger_helper import init_default_logger
from threading import Thread

# ================================================================
#  caching assist centralized/federated using recommender system 
# ================================================================

# logger handler
logger = init_default_logger()

# helper: run job queue in main thread instead whenever 
#         event is presumably triggered


if __name__ == '__main__':

  # ================================================================
  # 0. preparation and preprocessing
  # ================================================================

  # setup config and parse args into namespace 
  parser = build_parser(namespace=config)
  parser.parse_args(namespace=config)
  del parser

  config.verify()

  print('{:-^36}'.format(' sim benchmark '))
  print('  mode\t: {0.global_mode}'.format(config))
  print('  model\t: {0.recsys_name}'.format(config))
  print('{:-^36}'.format(''))

  # process data loader
  data_loader = ExampleDataLoader()
  data_loader.default_setup()

  # parse recsys config based on selected available models
  daisy_config = RecsysFL.init_daisy_config(data_loder=data_loader, algo_name=config.recsys_name)
  daisy_config['seed'] = config.global_rand_seed
  daisy_config['epochs'] = config.recsys_epochs

  print(daisy_config)

  # setup request iterator
  num_request = config.global_cutoff if config.global_cutoff > -1 else data_loader.nrow
  item_df = prepare_item_df(data_loader)
  item_len = len(item_df)
  request_df = prepare_request_df(data_loader)
  request_it = iter_requests(request_df, num_request)
  request_tqdm = tqdm.tqdm(request_it, total=num_request, ascii=True)
  round_mod = (num_request // config.recsys_round) - 1

  # ================================================================
  # 1. prep pseudo-server nodes
  # ================================================================

  # 1.0 various data loader inserts
  db_req_log_fieldnames = ['request_id', data_loader.tid, data_loader.uid, data_loader.iid, data_loader.inter]
  db_req_stat_log_fieldnames = ['request_id', data_loader.tid, data_loader.uid, data_loader.iid, data_loader.inter, 'status']

  # 1.1 event handler imports and setups
  # import default events from bootstrap
  from bootstrap import set_default_event, EventParamContentRequest


  # 1.2 define namespace and unload-train procedure

  class CacheItemPool:
    """ namespace for caching item pool """
    _item_proto = namedtuple('CacheItem', field_names=['uid', 'iid', 'val'])
    cache_item_pool = dict()

    @classmethod
    def convert_cache_item(cls, uid, iid, val):
      return cls._item_proto(uid, iid, val)

    @classmethod
    def map_convert_cache_item(cls, items):
      return list(map(lambda x: cls._item_proto(*x), items))

    @classmethod
    def clear_cache_pool(cls, server_name):
      cls.cache_item_pool[server_name] = []

    @classmethod
    def check_null(cls, server_name):
      if not cls.cache_item_pool.get(server_name):
        cls.clear_cache_pool(server_name)

    @classmethod
    def append_item(cls, server_name, item):
      cls.check_null(server_name)
      cls.cache_item_pool[server_name].append(item)

    @classmethod
    def extend_item(cls, server_name, items):
      cls.check_null(server_name)
      if isinstance(items, np.ndarray):
        items = items.flatten()
      items = list(items)
      cls.cache_item_pool[server_name].extend(item)

    @classmethod
    def get_most_common_items(cls, server_name, topn):
      items = cls.cache_item_pool.get(server_name)
      if items is None:
        return []
      items = Counter(items).most_common(topn)
      items = [i[0] for i in items]
      return items

  class CacheStrategies:
    """ namespace for caching strategies methods """

    @classmethod
    def fetch_servers(cls, root_node_server, predicate):
      servers = filter(lambda x: predicate(x), root_node_server.recurse_nodes())
      return list(servers)

    @classmethod
    def fetch_request_logs_df(cls, root_node_server, predicate):
      servers = cls.fetch_servers(root_node_server, predicate)
      dfs = map(lambda server: server.request_log_database.to_pd(use_cursor=True), servers)
      dfs = [i for i in dfs if len(i) > 0]
      if len(dfs) == 0:
        return None
      return pd.concat(dfs)



  # import event worker constructor
  from src.event import new_event_thread_worker

  # event worker
  job_queue = Queue()
  worker = new_event_thread_worker(job_queue)
  event_thread = Thread(target=worker, daemon=True)
  event_thread.start()

  # 1.3 server node setups
  # pre-setup base server -> has item pseudo-database
  base_server = Server('base_server')
  base_server.set_database(PandasDataFramePDB('item_db', item_df))
  base_server.set_timer(lambda: 0)
  base_server.cfg.cache_maxsize = int(config.netw_bs_alloc * item_len)
  base_server.cfg.db_req_log_fieldnames = db_req_log_fieldnames
  base_server.cfg.db_req_stat_log_fieldnames = db_req_stat_log_fieldnames

  base_server_runner = RecsysFL.new_recsys_runner(daisy_config)
  base_server.set_recsys_runner(base_server_runner)
  _, _, _, ur = base_server.recsys_runner.split(request_df)
  base_server.cfg.cache_maxage = 999

  # pre-setup edge server
  for i in range(1, config.netw_num_edge + 1):
    edge_server = base_server.spawn_child('edge_server_{:02}'.format(i))

    edge_server_runner = RecsysFL.new_recsys_runner(daisy_config)
    edge_server.set_recsys_runner(base_server_runner)
    _, _, _, ur = edge_server.recsys_runner.split(request_df)
    edge_server.set_timer(lambda: 0)
    edge_server.cfg.cache_maxsize = int(config.netw_es_alloc * item_len)
    edge_server.cfg.cache_maxage = 999
    edge_server.cfg.db_req_log_fieldnames = db_req_log_fieldnames
    edge_server.cfg.db_req_stat_log_fieldnames = db_req_stat_log_fieldnames

  # setup server defaults
  for server in base_server.recurse_nodes():
    server.setup(queue=job_queue)
    set_default_event(server.event_manager)
    if config.global_mode != SIM_MODE_ENUM.cache_aside:
      server.cfg.flag_suppress_cache_on_req = True

  # partition uids based on number of edge servers
  users_partitions = prepare_user_partition(data_loader, config.netw_num_edge)
  edge_users_map = group_partition(users_partitions, base_server.children)

  # ================================================================
  # 2. request routine
  # ================================================================
  with logging_redirect_tqdm(loggers=[logger], tqdm_class=request_tqdm):
    for req in request_tqdm:

      target_server = edge_users_map.get(req.uid)
      request_tqdm.set_description("user_id:{:>5} item_id:{:>5} server:{:<15}".format(req.uid, req.iid, str(target_server)))
      request_param = EventParamContentRequest(req.row, target_server, req.ts, req.uid, req.iid, req.val, 1)
      target_server.event_manager.trigger_event('OnContentRequest', event_param=request_param)

      # wait for jobs to be exhausted on each request
      # !caveat: there are no performance benefit compared
      # with using only main thread. but the program was 
      # written this way so rewiring everything would take
      # some additional time
      job_queue.join()

      # 2.1 cache logic flow
      # training rounds
      if config.global_mode == SIM_MODE_ENUM.cache_aside:
        pass
      
      elif round_mod != 0 and req.row > 0 and req.row % round_mod == 0:
        if config.global_mode == SIM_MODE_ENUM.centralized:
          df = CacheStrategies.fetch_request_logs_df(base_server, lambda x: x.name == 'base_server')
          base_server.recsys_runner.train(df)

        elif config.global_mode == SIM_MODE_ENUM.localized:
          for server in CacheStrategies.fetch_servers(base_server, lambda x: True):
            df = CacheStrategies.fetch_request_logs_df(base_server, lambda x: x.name == server.name)
            if df is None:
              continue
            server.recsys_runner.train(df)

        elif config.global_mode == SIM_MODE_ENUM.federated:
          for server in CacheStrategies.fetch_servers(base_server, lambda x: x.name != 'base_server'):
            df = CacheStrategies.fetch_request_logs_df(base_server, lambda x: x.name == server.name)
            if df is None:
              continue
            server.recsys_runner.train(df)

  # wait until queue is exhausted
  job_queue.join()

  for server in base_server.recurse_nodes():
    print(server, server.states)