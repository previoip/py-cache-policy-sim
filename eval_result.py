import os
import json
import pandas as pd
import matplotlib.pyplot as plt


CONF_HIST_FILENAME = 'hist.json'

HIST_CONF_NAME = 'cache_aside_test'
HIST_INDEX = -1


if __name__ == '__main__':
  with open(CONF_HIST_FILENAME, 'r') as fo:
    hist = json.load(fo)
  hist = list(filter(lambda x: x['conf_name'] == HIST_CONF_NAME, hist))[HIST_INDEX]

  hist_name = hist['conf_name']
  hist_timestamp = hist['conf_ts']
  hist_info = hist['general']
  hist_net_info = hist['network_conf']
  results = hist['results']

  print('evaluating sim run history', hist_name, hist_timestamp, sep=' | ')

  print()
  print('hit ratios:')
  for hit_ratio in results['hit_ratios']:
    print(hit_ratio)

  log_folder = hist_info['log_folder']
  log_dfs = []
  for log_file_record in results['log_files']:
    if log_file_record['type'] != 'request_stats':
      continue

    log_df = pd.read_csv(os.path.join(log_folder, log_file_record['fp']), sep=';')
    log_df['server'] = log_file_record['server']
    log_dfs.append(log_df)

  log_df = pd.concat(log_dfs)
  log_df.reset_index(inplace=True, drop=True)

  print(log_df)