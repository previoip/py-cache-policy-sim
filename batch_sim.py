from main import SIM_MODE_ENUM, RECSYS_MODEL_ENUM
import subprocess
import itertools

BASE_ARGS = ['python3', 'main.py']

class SIMUL_CONF:
  modes = [SIM_MODE_ENUM.cache_aside]
  edge_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  
  # v only valid if mode != cache_aside, else multiple simnul_confs 
  # v results will be redundant. leave one value to make it ran once
  models = [RECSYS_MODEL_ENUM.mf]  
  rounds = [1000] 

def proc_del_sim_results():
  import os, shutil
  if os.path.exists('hist.json') and os.path.isfile('hist.json'):
    print('removing previous hist.json')
    os.remove('hist.json')
  if os.path.exists('log') and os.path.isdir('log'):
    print('removing previous logs')
    shutil.rmtree('log')

if __name__ == '__main__':
  proc_del_sim_results()

  for mode in SIMUL_CONF.modes:
    for round in SIMUL_CONF.rounds:
      for frac in SIMUL_CONF.edge_fracs:
        args = BASE_ARGS.copy()
        args += ['--trial_cutoff', '10000'] # comment these out to simulate full dataset
        args += ['--conf_prfx', f'{mode}_r{round}']
        args += ['--conf_name', f'{mode}_r{round}_{frac}'.replace('.', '')]
        args += ['--edge_server_alloc_frac', str(frac)]
        args += ['--mode', mode]
        args += ['--round_at_n_iter', str(round)]
        subprocess.call(args)