import os
import json
import pandas as pd
import numpy as np
import typing as t
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

CONF_HIST_FILENAME = 'hist.json'
FIG_EXPORT_PATH = './fig'

class TELECOM_CONST:
                # (unit),       (desc)
  b_m   = 50e6  # Hz            bandwidth (edge -> users)
  P_m   = 24    # W, dBm        transmission power (edge -> users)
  # G_m         # [0, 1]        channel gain range (edge -> users)
  omega = -163  # W/hz, dBm/Hz  power density over noise
  B_m   = 100e6 # Hz            bandwidth (base -> edge)
  P_cs  = 30    # W, dBm        transmission power (base -> edge)
  # G_cs        # [0, 1]        channel gain range (base -> edge)

# helpers:

def cast_to(v: t.Any, np_dtype: np.dtype):
  if isinstance(v, np.generic):
    return v.astype(np_dtype)
  return np.cast[np_dtype](v)


def parse_hitorical_json(fp, ensure_only_latests=True):
  __hist_rec_nt = namedtuple('HistRecord', field_names=['hist_name', 'hist_timestamp', 'hist_info', 'hist_net_info', 'results'])

  def __unpack_hist_rec(d):
    return __hist_rec_nt(
      hist_name = d.get('conf_name'),
      hist_timestamp = d.get('conf_ts'),
      hist_info = d.get('general'),
      hist_net_info = d.get('network_conf'),
      results = d.get('results')
    )

  with open(fp, 'r') as fo:
    hists = json.load(fo)
    hists = sorted(hists, key=lambda x: x['conf_name'] + x['conf_ts'])
  hist_names = []
  hist_selects = []

  for i, hist in enumerate(hists):
    if not ensure_only_latests or not hist['conf_name'] in hist_names:
      hist_names.append(hist['conf_name'])
      hist_selects.append(hist)

  return map(__unpack_hist_rec, hist_selects)


def read_and_concat_dfs(hist_record):
  log_folder = hist_record.hist_info['log_folder']
  _log_dfs = []
  for log_file_record in hist_record.results['log_files']:
    if log_file_record['type'] != 'request_stats':
      continue
    log_df = pd.read_csv(os.path.join(log_folder, log_file_record['fp']), sep=';')
    log_df['server'] = log_file_record['server']
    _log_dfs.append(log_df)
  log_df = pd.concat(_log_dfs)
  log_df.reset_index(inplace=True, drop=True)
  return log_df


def filter_df(df, parsed_args):
  df = df.copy()
  # store types for typecast purposes
  _df_dtypes = df.dtypes
  # filter should the filter args subgroup is set
  for filt_col in ['request_id', 'server', 'status']:
    parsed_vals = parsed_args.get(filt_col)
    # fallback value if status col is not set
    if filt_col == 'status' and parsed_vals is None:
      parsed_vals = 'cache_hit,cache_missed'
    elif parsed_vals is None:
      continue
    parsed_vals = parsed_vals.split(',') if parsed_vals.find(',') != -1 else [parsed_vals]
    df_filt = df[filt_col] == cast_to(parsed_vals.pop(0), _df_dtypes[filt_col])
    for parsed_val in parsed_vals:
      df_filt |= df[filt_col] == cast_to(parsed_val, _df_dtypes[filt_col])
    df = df[df_filt]
  return df


def argparse_setup():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--save-fig', action='store_true', help='save matplotlib figure, path is hardcoded into program const')
  subargp = parser.add_subparsers()
  subargp_filter = subargp.add_parser('filter')
  subargp_filter.add_argument('--request-id')
  subargp_filter.add_argument('--server')
  subargp_filter.add_argument('--status')
  return parser


if __name__ == '__main__':

  parser = argparse_setup()
  parsed_args = vars(parser.parse_args())

  if parsed_args.get('save_fig', False):
    os.makedirs(FIG_EXPORT_PATH, exist_ok=True)

  config_hist_records = parse_hitorical_json(CONF_HIST_FILENAME)

  for config_hist_record in config_hist_records:
    print()
    print('evaluating:', config_hist_record.hist_name, config_hist_record.hist_timestamp)

    log_df = read_and_concat_dfs(config_hist_record)
    log_df = filter_df(log_df, parsed_args)

    # Vxy predicate
    log_df['Vxy'] = np.where((log_df['status'].apply(lambda x: str(x).lower().endswith('hit'))), 1, 0)

    # sort again by timestamp
    log_df.sort_values(by='timestamp', inplace=True)

    # check preprocessed df
    print()
    print('log dataframe:')
    print(log_df)