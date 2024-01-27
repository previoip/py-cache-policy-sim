import argparse
import time
from simutils import strvars
from src.model.mp_models import RECSYS_MODEL_ENUM


class SIM_MODE_ENUM:
  cache_aside = 'cache_aside'
  centralized = 'centralized'
  localized   = 'localized'
  federated   = 'federated'


class config:
  name                  = 'cache_aside_4div10_alloc'
  prfx                  = 'ca'
  ts                    = time.strftime('%Y-%m-%dT%H%M')

  global_rand_seed      = 1337
  global_mode           = SIM_MODE_ENUM.cache_aside
  global_cutoff         = -1

  recsys_name           = None
  recsys_ftopk          = 0.75
  recsys_round          = 10

  path_log_dir          = './log'
  path_fmt_log_req      = 'logreq_{}.csv'
  path_fmt_log_req_stat = 'logreq_stat_{}.csv'
  path_fmt_log_hist     = 'hist_{}.json'

  netw_num_edge         = 3
  netw_bs_alloc         = 1.0
  netw_es_alloc         = 0.4

  df_uid       = None
  df_iid       = None
  df_interid   = None
  df_tsid      = None

  @classmethod
  def verify(cls):
    if cls.global_mode == SIM_MODE_ENUM.cache_aside:
      cls.recsys_name = None
    if not cls.recsys_name is None and cls.global_mode != SIM_MODE_ENUM.cache_aside:
      cls.global_mode = SIM_MODE_ENUM.centralized


def build_parser(namespace: config):
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, dest='name', default=namespace.name)
  parser.add_argument('--rand-seed', type=int, dest='global_rand_seed', default=namespace.global_rand_seed)
  parser.add_argument('--cutoff', type=int, dest='global_cutoff', default=namespace.global_cutoff)
  parser.add_argument('--mode', type=str, dest='global_mode', default=namespace.global_mode, choices=strvars(SIM_MODE_ENUM).values())
  parser.add_argument('--model', type=str, dest='recsys_name', default=namespace.recsys_name, choices=strvars(RECSYS_MODEL_ENUM).values())
  parser.add_argument('--model-train-rounds', type=int, dest='recsys_round', default=namespace.recsys_round)
  parser.add_argument('--fl-edges-count', type=int, dest='netw_num_edge', default=namespace.netw_num_edge)
  parser.add_argument('--fl-edges-alloc', type=float, dest='netw_es_alloc', default=namespace.netw_es_alloc)
  parser.add_argument('--fl-base-alloc', type=float, dest='netw_bs_alloc', default=namespace.netw_bs_alloc)
  parser.add_argument('--fmt-reqlog', type=str, dest='path_fmt_log_req', default=namespace.path_fmt_log_req)
  parser.add_argument('--fmt-reqlogst', type=str, dest='path_fmt_log_req_stat', default=namespace.path_fmt_log_req_stat)
  return parser