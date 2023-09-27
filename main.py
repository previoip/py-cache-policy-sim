from src.pseudo_server import Server
from src.data_examples.ml_data_loader import ExampleDataLoader
from src.pseudo_database import PandasDataFramePDB
import queue
import threading
from bootstrap import dispatch_default

sim_conf = {
  'general': {
    'trial_cutoff': 100
  },

  'loader_conf': {
    'item_key'  : 'movie_id',
    'user_key'  : 'user_id',
    'value_key' : 'rating'
  },
 
  'network_conf': {
    'base_server_alloc_frac': .8,
    'edge_server_alloc_frac': .4,
    'num_edge': 5, # number of user-end (edge/client) groups, previously `num_cl`
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

def prepare_user_df(data_loader):
  user_key = sim_conf['loader_conf']['user_key']
  user_ls = data_loader.df_user[[user_key]].unique()
  user_count = len(user_ls)

def create_worker(q: queue.Queue):
  def worker():
    while True:
      task = q.get()
      errno = task()
      q.task_done()
  return worker


if __name__ == '__main__':

  # ================================================
  # bootstrap

  data_loader = ExampleDataLoader()
  data_loader.download()
  data_loader.load()


  scheduler = queue.Queue()
  runner = create_worker(scheduler)

  # ================================================
  # server sim setup

  base_server = Server('base_server')
  base_server.set_database(PandasDataFramePDB('movie_db', prepare_movie_df(data_loader)))
  base_server.setup()
  base_server.event_manager.set_scheduler(scheduler)
  dispatch_default(base_server.event_manager)

  for n in range(sim_conf['network_conf']['num_edge']):
    edge_server = base_server.spawn_child(f'edge_server_{n}')
    edge_server.setup()
    edge_server.event_manager.set_scheduler(scheduler)
    dispatch_default(edge_server.event_manager)


  # ================================================
  # begin

  thread = threading.Thread(target=runner, daemon=True)
  thread.start()

  def tw_test_fetch_from_db(pdb, key):
    def tw():
      print(key, pdb.get(key))
      return 0
    return tw

  _max_req = sim_conf['general']['trial_cutoff']
  for row, _index, user_id, item_id, value, timestamp in iter_requests(prepare_request_df(data_loader)):

    task = tw_test_fetch_from_db(base_server.database, item_id)
    scheduler.put(task)

    if _max_req != 0 and row > _max_req: break

  scheduler.join()


  i = 0
  while i < 10:
    print(i)
    i += 1
    break
  else:
    print('foo')