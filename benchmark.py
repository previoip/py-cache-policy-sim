import os
import tqdm
import argparse
import typing as t
import numpy as np
import pandas as pd
import time
from src.data_loaders.ml_data_loader import ExampleDataLoader
from src.model.mp_models import RECSYS_MODEL_ENUM
from src.model.mp_runner import RecsysFL
from config import config, build_parser, SIM_MODE_ENUM
import traceback


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

  # item_df = prepare_item_df(data_loader)
  err_fo = open('debug_errors_1.txt', 'w')

  for algo_name in [
    # RECSYS_MODEL_ENUM.item2vec,
    # RECSYS_MODEL_ENUM.itemknn,
    # RECSYS_MODEL_ENUM.lightgcn,
    # RECSYS_MODEL_ENUM.mf,
    # RECSYS_MODEL_ENUM.mostpop,
    # RECSYS_MODEL_ENUM.multi_vae,
    # RECSYS_MODEL_ENUM.nfm,
    # RECSYS_MODEL_ENUM.fm,
    # RECSYS_MODEL_ENUM.neumf,
    # RECSYS_MODEL_ENUM.ngcf,
    RECSYS_MODEL_ENUM.slim
  ]:
    daisy_config = RecsysFL.init_daisy_config(data_loder=data_loader, algo_name=algo_name)
    daisy_config['epochs'] = 1
    try:
      runner = RecsysFL.new_recsys_runner(daisy_config)
      train_set, test_set, test_ur, train_ur = runner.split(data_loader.df_ratings)
      print(len(train_ur))
      print(len(test_ur))

      print(len(test_ur) + len(train_ur))

      print(len(train_set))
      print(len(test_set))
      
      # runner.train(train_set)
      # print(runner.model.predict(1, 1))
      # print(runner.model.full_rank(1))
    
    except Exception:
      err_fo.writelines([algo_name, '\n', traceback.format_exc(),'\n\n'])


  err_fo.close()

