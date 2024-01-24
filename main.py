from config import config, build_parser
from simutils import prepare_item_df, prepare_request_df, prepare_user_partition, iter_requests
from src.data_loaders.ml_data_loader import ExampleDataLoader
from src.model.daisy_monkeypatch import init_config, RecsysFL
from src.pseudo_database import PandasDataFramePDB
from src.pseudo_server import Server
import tqdm

if __name__ == '__main__':
  # parse args into config namespace
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

  daisy_config = init_config(data_loder=data_loader, algo_name=config.recsys_name)

  print(daisy_config)
  runner = RecsysFL.new_recsys_runner(daisy_config)

  print(runner.model.wref_config)

  num_request = config.global_cutoff if config.global_cutoff > -1 else data_loader.nrow
  item_df = prepare_item_df(data_loader)
  request_df = prepare_request_df(data_loader)
  request_it = iter_requests(request_df, num_request)
  request_tqdm = tqdm.tqdm(request_it, total=num_request, ascii=True)

  edge_users_partitions = prepare_user_partition(data_loader, config.netw_num_edge)

  base_server = Server('base_server')
  base_server.set_database(PandasDataFramePDB('item_db', item_df))

  for i in range(config.netw_num_edge):
    edge_server = Server('edge_server_{:02}'.format(i))


