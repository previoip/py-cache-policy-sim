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
  # param naming convention: {kind; c, rng}_{varname}_{[super]}_{[sub]}

  class PARAMS:
    seed = 1337
    uniform_hy = 1      # kb           assumes all content have same sizes

    class USER_PARAM:
      val_h__xm           = 1           # kb           assumes all content have same sizes
      rng_gamma_r_xm      = (0.3, 1   ) # (0.03, 0.1)  
      rng_gamma_dl_xm     = (5.0, 10.0)

    class SIM_PARAM:
      val_delta__m        = 10e9
      val_delta__bs       = 50e9
      val_delta__CNC      = 100e9
      val_c__xm           = 1500e3
      val_b__m            = 20e6      # Hz            bandwidth (edge -> users)
      val_B__CS           = 100e6     # Hz    bandwidth (base -> edge)
      val_omega__         = -163      # W/hz, dBm/Hz  power density over noise
      # b_m   = 50e6  
      # P_m   = 24    # W, dBm        transmission power (edge -> users)
      # G_m           # [0, 1]        channel gain range (edge -> users)
      # B_m   = 100e6 # Hz            
      # P_cs  = 30    # W, dBm        transmission power (base -> edge)
      # G_cs          # [0, 1]        channel gain range (base -> edge)

  @staticmethod
  def _request_delay(h, R):
    return h / R

  @classmethod
  def request_delay(cls, df):
    return cls._request_delay(df['h'], df['R'])

  @staticmethod
  def _content_delivery(Vxy, h, rx, R):
    return (Vxy * h / rx) + ((1 - Vxy) * ((h / R) + (h / rx)))

  @classmethod
  def content_delivery(cls, df):
    return cls._content_delivery(df['Vxy'], df['h'], df['rx'], df['R'])

  @staticmethod
  def _communication_energy_cost(dxy, P_m, dy, P_cs):
    return (dxy * P_m) + (dy * P_cs)

  # @classmethod
  # def communication_energy_cost(cls, df):
  #   return cls._communication_energy_cost(df['dxy'], cls.DR_CONST.P_m, df['dy'], cls.DR_CONST.P_cs)


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
    log_df = pd.read_csv(os.path.join(log_folder, log_file_record['fp']), sep=';', index_col=0)
    log_df['server'] = log_file_record['server']
    _log_dfs.append(log_df)
  log_df = pd.concat(_log_dfs)
  log_df.sort_values(by=['timestamp', 'request_id'], ascending=[True, True], inplace=True)
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

def plot_helper(plt, res_dict, res_ctx, plotter=plt.bar):
  data = dict()
  for config_name in res_dict.keys():
    val = res_dict[config_name].get(res_ctx)
    if val is None:
      continue
    label = config_name.split('_')[-1]
    data[label] = val
  if data:
    plotter(*zip(*data.items()))
  return plt


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

    np.random.seed(TELECOM_EVAL.PARAMS.seed)

    eval_results[config_hist_record.hist_name] = dict()

    log_df = read_and_concat_dfs(config_hist_record)
    log_df = filter_df(log_df, parsed_args)

    log_df['timestamp'] -= log_df['timestamp'].min()

    # store server depth for future use
    log_df['depth'] = log_df['server'].apply(lambda x: config_hist_record.results['depths'].get(x))

    # sort by timestamp request_id and depth for good measure
    log_df.sort_values(by=['timestamp', 'request_id', 'depth'], ascending=[True, True, False], inplace=True)

    # eval V__xy predicate
    log_df['V__xy'] = np.where((log_df['status'].apply(lambda x: str(x).lower().endswith('hit'))), 1, 0)

    # hackish way to get edge-user download rate with even random distibution
    # scoped to avoid unecessary variable contamination
    def __1():
      # define a mask to exclude base_server
      mask_ls_edge_to_user_id = log_df['server'] != 'base_server'
      # 'serialize' as string for each edge server to user as strin to get unique pairs
      ls_edge_to_user_id = (log_df[mask_ls_edge_to_user_id]['server'] + '|' + log_df[mask_ls_edge_to_user_id]['user_id'].map(str)).sort_values().unique()
      # store the length
      len_edge_to_user_id = len(ls_edge_to_user_id) 
      # define the divisors based on the length with range func
      dl_edge_to_user_id = range(len_edge_to_user_id)
      # perform linear interpolation based on range divisors and download rate param
      lambda_lin_interp = lambda x1, y1, x2, y2, x: y1 + ((y1 - y2) / (x1 - x2) * x)
      dl_edge_to_user_id = map(lambda x: lambda_lin_interp(1, TELECOM_EVAL.PARAMS.USER_PARAM.rng_gamma_r_xm[0], len_edge_to_user_id, TELECOM_EVAL.PARAMS.USER_PARAM.rng_gamma_r_xm[1], x), dl_edge_to_user_id)
      dl_edge_to_user_id = list(dl_edge_to_user_id)
      np.random.shuffle(dl_edge_to_user_id)
      # store shuffled edge-user download rate as mappable object
      dl_edge_to_user_id = dict(zip(ls_edge_to_user_id, dl_edge_to_user_id))

      # now do the same with base-edge
      ls_base_to_edge = log_df[mask_ls_edge_to_user_id]['server'].sort_values().unique()
      len_ls_base_to_edge = len(ls_base_to_edge) 
      dl_base_to_edge = range(len_ls_base_to_edge)
      dl_base_to_edge = map(lambda x: lambda_lin_interp(1, TELECOM_EVAL.PARAMS.USER_PARAM.rng_gamma_dl_xm[0], len_ls_base_to_edge, TELECOM_EVAL.PARAMS.USER_PARAM.rng_gamma_dl_xm[1], x), dl_base_to_edge)
      dl_base_to_edge = list(dl_base_to_edge)
      np.random.shuffle(dl_base_to_edge)
      dl_base_to_edge = dict(zip(ls_base_to_edge, dl_base_to_edge))

      # to determine which rate is uzed in log_df, check whether request propagates to base server (missed)
      log_it = log_df.itertuples()
      R = []
      for n, row in enumerate(log_it):
        R.append(dl_edge_to_user_id.get(row.server + '|' + str(row.user_id), 0))
        if not row.server.startswith('base') and row.status.endswith('missed'):
          R.append(dl_base_to_edge.get(row.server, 0))
          next(log_it)
      log_df['R'] = R
      del R

    __1()
    

    # sizeof as hy per content_id (or a constant value)
    if 'sizeof' in log_df.columns:
      log_df['h'] = log_df['sizeof']
    else:
      log_df['h'] = TELECOM_EVAL.PARAMS.USER_PARAM.val_h__xm

    log_df['d_r'] = TELECOM_EVAL.request_delay(log_df)

    # print(log_df[['server', 'status', 'h', 'R', 'd_r']])

    # ================================================
    # begin

    # check evaluated df
    print()
    print('evaluated log dataframe:')
    print(log_df)


    # server aggregates
    agg_gr_serv = log_df[['server', 'd_r', 'R']].groupby('server')
    agg_gr_user = log_df[['request_id', 'd_r', 'R']].groupby('request_id')
    agg_gr_user_exl_BS = log_df[log_df['server'] != 'base_server'][['request_id', 'd_r', 'R', 'V__xy']].groupby('request_id')


    per_server_metrics = agg_gr_serv[['d_r', 'R']].agg(np.average)
    print()
    print('average metrics per server:')
    print(per_server_metrics)

    overall_server_metrics = agg_gr_user[['d_r', 'R']].agg('sum').agg(np.average)
    overall_server_metrics_exl_BS = agg_gr_user_exl_BS[['V__xy']].agg('sum').agg(np.average)
    print()
    print('average metrics overall:')
    print(overall_server_metrics)
    print()

    eval_results[config_hist_record.hist_name]['avg_d_r'] = overall_server_metrics['d_r']
    eval_results[config_hist_record.hist_name]['V__xy'] = overall_server_metrics_exl_BS['V__xy']

# see aggregates in plt figures

plt.rcParams["figure.figsize"] = (9, 3)

def show_and_close(plt, save_fpath):
  if do_save_figs:
    plt.savefig(os.path.join(FIG_EXPORT_PATH, save_fpath))
  else:
    plt.show()
  plt.close()

plt = plot_helper(plt, eval_results, 'avg_d_r')
plt.title('Average Request Delay')
show_and_close(plt, 'delay_bar.png')

plt = plot_helper(plt, eval_results, 'avg_d_r', plotter=plt.plot)
plt.title('Average Request Delay')
show_and_close(plt, 'delay.png')

plt = plot_helper(plt, eval_results, 'V__xy', plotter=plt.plot)
plt.title('Average Request Hit Rate')
show_and_close(plt, 'hit.png')



