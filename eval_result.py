import os
import json
import pandas as pd
import numpy as np
import typing as t
import matplotlib.pyplot as plt
import argparse

CONF_HIST_FILENAME = 'hist.json'

HIST_CONF_NAME = 'cache_aside_test'
HIST_INDEX = -1
FIG_EXPORT_PATH = './fig'


# helpers 
def cast_to(v: t.Any, np_dtype: np.dtype):
  if isinstance(v, np.generic):
    return v.astype(np_dtype)
  return np.cast[np_dtype](v)


# argparser for ease debugging
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default=HIST_CONF_NAME, help=f'config history name, default: {HIST_CONF_NAME}')
parser.add_argument('-i', '--index', default=HIST_INDEX, type=int, help=f'config history index, default: {HIST_INDEX}')
parser.add_argument('-s', '--save-fig', action='store_true', help='save matplotlib figure, path is hardcoded into program const')

subargp = parser.add_subparsers()
subargp_evalconf = subargp.add_parser('filter') 
subargp_evalconf.add_argument('--request-id')
subargp_evalconf.add_argument('--server')
subargp_evalconf.add_argument('--status')

if __name__ == '__main__':

  parsed_args = vars(parser.parse_args())  

  history_conf_name = parsed_args.get('name', HIST_CONF_NAME)
  history_conf_index = parsed_args.get('index', HIST_INDEX)
  save_fig = parsed_args.get('save_fig', False)


  with open(CONF_HIST_FILENAME, 'r') as fo:
    hist = json.load(fo)
  hist = list(filter(lambda x: x['conf_name'] == history_conf_name, hist))[history_conf_index]

  hist_name = hist.get('conf_name')
  hist_timestamp = hist.get('conf_ts')
  hist_info = hist.get('general')
  hist_net_info = hist.get('network_conf')
  results = hist.get('results')

  print('evaluating sim run history:')
  print(hist_name, hist_timestamp, sep=' | ')

  print()
  print('printing result history:')
  for res_key in results.keys():
    print('\tresult item:', res_key)
    for res_val in results[res_key]:
      print('\t\t-', res_val)
    print()
  
  print()
  print('loading log files...')
  log_folder = hist_info['log_folder']
  _log_dfs = []
  for log_file_record in results['log_files']:
    if log_file_record['type'] != 'request_stats':
      continue
    log_df = pd.read_csv(os.path.join(log_folder, log_file_record['fp']), sep=';')
    log_df['server'] = log_file_record['server']
    _log_dfs.append(log_df)
  log_df = pd.concat(_log_dfs)
  log_df.reset_index(inplace=True, drop=True)

  # store types for typecast purposes
  _log_df_dtypes = log_df.dtypes

  # filter should the filter args subgroup is set
  for filt_col in ['request_id', 'server', 'status']:
    parsed_vals = parsed_args.get(filt_col)
    # edge case for col status
    if filt_col == 'status' and parsed_vals is None:
      parsed_vals = 'cache_hit,cache_missed'
    elif parsed_vals is None:
      continue
    parsed_vals = parsed_vals.split(',') if parsed_vals.find(',') != -1 else [parsed_vals] 
    log_df_filt = log_df[filt_col] == cast_to(parsed_vals.pop(0), _log_df_dtypes[filt_col])
    for parsed_val in parsed_vals:
      log_df_filt |= log_df[filt_col] == cast_to(parsed_val, _log_df_dtypes[filt_col])
    log_df = log_df[log_df_filt]

  print('done loading log files.')

  print()
  print('preprocessing log files...')

  # Vxy predicate
  log_df['Vxy'] = np.where((log_df['status'].apply(lambda x: str(x).lower().endswith('hit'))), 1, 0)

  print('done preprocessing log files.')

  # check preprocessed df
  print()
  print('log dataframe:')
  print(log_df)
