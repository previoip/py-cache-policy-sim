import os
import tqdm
import argparse
import typing as t
import numpy as np
import pandas as pd
import time
from src.data_loaders.ml_data_loader import ExampleDataLoader
from src.model.daisy_monkeypatch import RecsysFL, RECSYS_MODEL_ENUM
from config import config, build_parser, SIM_MODE_ENUM


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
  daisy_config = RecsysFL.init_daisy_config(data_loder=data_loader, algo_name=RECSYS_MODEL_ENUM.mf)
  runner = RecsysFL.new_recsys_runner(daisy_config)
  train_set, test_set, test_ur, train_ur = runner.split(data_loader.df_ratings)

  



