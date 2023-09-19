from src.pseudo_server import Server
from src.data_examples.ml_data_loader import ExampleDataLoader
from src.pseudo_database import PandasDataFramePDB

sim_conf = {
  'loader_conf': {
    'item_key'  : 'movie_id',
    'user_key'  : 'user_id',
    'value_key' : 'rating'
  },

  'network_conf': {
    'n_edge': 5,
  }
}

def prepare_request_df(data_loader):
  return data_loader.df_ratings.sort_values(by='unix_timestamp')[
    [
      sim_conf['loader_conf']['user_key'],
      sim_conf['loader_conf']['item_key'],
      sim_conf['loader_conf']['value_key']
    ]
  ]

def prepare_movie_df(data_loader):
  item_key = sim_conf['loader_conf']['item_key']
  df_movies = data_loader.df_movies[[item_key]]
  df_movies[[item_key]] = df_movies[[item_key]].astype('int')
  df_movies.set_index(item_key, drop=False, inplace=True)
  return df_movies


if __name__ == '__main__':
  data_loader = ExampleDataLoader()
  data_loader.download()
  data_loader.load()

  base_server = Server('base_server')
  # base_server.database = PandasDataFramePDB('movie_db', prepare_movie_df(data_loader))
  base_server.setup()

  for n in range(sim_conf['network_conf']['n_edge']):
    edge_server = base_server.spawn_child(f'edge_server_{n}')
    edge_server.setup()

  for _ in base_server.recurse_callback(lambda node: print(node._edges)):
    pass

  print()

  for edge in base_server.recurse_edges():
    print(edge.graph_repr())