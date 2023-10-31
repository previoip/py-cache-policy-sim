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

class TELECOM_EVAL:

  class PARAMS:
    uniform_hy = 1 # assumes all content have same sizes

  class DR_CONST:
                  # (unit),       (desc)
    b_m   = 50e6  # Hz            bandwidth (edge -> users)
    P_m   = 24    # W, dBm        transmission power (edge -> users)
    # G_m         # [0, 1]        channel gain range (edge -> users)
    omega = -163  # W/hz, dBm/Hz  power density over noise
    B_m   = 100e6 # Hz            bandwidth (base -> edge)
    P_cs  = 30    # W, dBm        transmission power (base -> edge)
    # G_cs        # [0, 1]        channel gain range (base -> edge)

  @staticmethod
  def _content_delivery(Vxy, hy, rx, R):
    return (Vxy * hy / rx) + ((1 - Vxy) * ((hy / R) + (hy / rx)))

  @classmethod
  def content_delivery(cls, df):
    return cls._content_delivery(df['Vxy'], df['hy'], df['rx'], df['R'])

  @staticmethod
  def _communication_energy_cost(dxy, P_m, dy, P_cs):
    return (dxy * P_m) + (dy * P_cs)

  @classmethod
  def communication_energy_cost(cls, df):
    return cls._communication_energy_cost(df['dxy'], cls.DR_CONST.P_m, df['dy'], cls.DR_CONST.P_cs)


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
  eval_results = dict()

  do_save_figs = parsed_args.get('save_fig', False)

  if do_save_figs:
    os.makedirs(FIG_EXPORT_PATH, exist_ok=True)

  config_hist_records = parse_hitorical_json(CONF_HIST_FILENAME)

  for config_hist_record in config_hist_records:
    print()
    print('evaluating:', config_hist_record.hist_name, config_hist_record.hist_timestamp)

    eval_results[config_hist_record.hist_name] = dict()

    log_df = read_and_concat_dfs(config_hist_record)
    log_df = filter_df(log_df, parsed_args)

    server_names_ls = log_df['server'].unique()

    # sort again by timestamp for good measure
    log_df.sort_values(by='timestamp', inplace=True)

    # check df prior preproc
    print()
    print('log dataframe:')
    print(log_df)


    # eval Vxy predicate
    log_df['Vxy'] = np.where((log_df['status'].apply(lambda x: str(x).lower().endswith('hit'))), 1, 0)

    # sizeof as hy per content_id (or uniform)
    if 'sizeof' in log_df.columns:
      log_df['hy'] = log_df['sizeof']
    else:
      log_df['hy'] = TELECOM_EVAL.PARAMS.uniform_hy

    # !!! experiment: R transmission rate is proportional to inverse node depth attribute
    log_df['R'] = log_df['server'].apply(lambda x: 1 / float(config_hist_record.results['depths'].get(x)))

    # !!! experiment: rx downlink transmission rate is proportional to R squared
    log_df['rx'] = log_df['R'] ** 2

    # reeval for dy
    log_df['dy'] = log_df['hy'] / log_df['R']

    # ================================================
    # begin

    log_df['dxy'] = TELECOM_EVAL.content_delivery(log_df)
    log_df['Ecom'] = TELECOM_EVAL.communication_energy_cost(log_df)

    # check evaluated df
    print()
    print('evaluated log dataframe:')
    print(log_df)

    # server aggregates
    agg_df = log_df[['server', 'dxy', 'Ecom']]
    agg_gr = agg_df.groupby('server')
    agg_df = log_df[['dxy', 'Ecom']]

    per_server_metrics = agg_gr.agg(np.average)
    print()
    print('average metrics per server:')
    print(per_server_metrics)

    overall_server_metrics = agg_df.agg(np.average)
    print()
    print('average metrics overall:')
    print(overall_server_metrics)
    print()

    eval_results[config_hist_record.hist_name]['dxy'] = overall_server_metrics['dxy']
    eval_results[config_hist_record.hist_name]['Ecom'] = overall_server_metrics['Ecom']

# see aggregates in plt figures

plt.rcParams["figure.figsize"] = (8, 4)

# > plot for dxy delay

d_plot_dxy = dict()
for config_name in eval_results.keys():
  val = eval_results[config_name]['dxy']
  d_plot_dxy[config_name] = val

plt.bar(*zip(*d_plot_dxy.items()))
if do_save_figs:
  plt.savefig(os.path.join(FIG_EXPORT_PATH, 'dxy.png'))
else:
  plt.show()
plt.close()

# > plot for Ecom consumption

d_plot_dxy = dict()
for config_name in eval_results.keys():
  val = eval_results[config_name]['Ecom']
  d_plot_dxy[config_name] = val

plt.bar(*zip(*d_plot_dxy.items()))
if do_save_figs:
  plt.savefig(os.path.join(FIG_EXPORT_PATH, 'Ecom.png'))
else:
  plt.show()
plt.close()