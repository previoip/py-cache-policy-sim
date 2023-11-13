from main import SIM_MODE_ENUM, RECSYS_MODEL_ENUM
import subprocess
import itertools
from collections import namedtuple

BASE_ARGS = ['python3', 'main.py']
PRUNE_PREVIOUS_RESULTS = False
FRACS_RES = 10


class SIMUL_CONF:
  nt_template = namedtuple('SimConf', field_names=['mode', 'model', 'edge_frac', 'round'])

  modes = [SIM_MODE_ENUM.cache_aside, SIM_MODE_ENUM.localized, SIM_MODE_ENUM.centralized, SIM_MODE_ENUM.federated]
  edge_fracs = map(lambda x: x/FRACS_RES, range(1, FRACS_RES+1))
  models = [None, RECSYS_MODEL_ENUM.mf, RECSYS_MODEL_ENUM.item2vec]
  rounds = [500, 1000] 


# helpers 

def proc_del_sim_results():
  import os, shutil
  if os.path.exists('hist.json') and os.path.isfile('hist.json'):
    print('removing previous hist.json')
    os.remove('hist.json')
  if os.path.exists('log') and os.path.isdir('log'):
    print('removing previous logs')
    shutil.rmtree('log')

if __name__ == '__main__':
  if PRUNE_PREVIOUS_RESULTS:
    # proc_del_sim_results()
    pass

  perm = itertools.product(*[
    SIMUL_CONF.modes,
    SIMUL_CONF.models,
    SIMUL_CONF.edge_fracs,
    SIMUL_CONF.rounds,
  ])
  perm = map(lambda x: SIMUL_CONF.nt_template(*x), perm)
  perm = filter(lambda x: (x.mode == SIM_MODE_ENUM.cache_aside and x.model == None) or (x.mode != SIM_MODE_ENUM.cache_aside and x.model != None), perm)

  for mode, model, edge_frac, round in perm:
    args = BASE_ARGS.copy()
    args += ['--trial_cutoff', '20000'] # comment these out to simulate full dataset
    args += ['--conf_prfx', f'{mode}_{model}_r{round}']
    args += ['--conf_name', f'{mode}_{model}_r{round}_{edge_frac}'.replace('.', '')]
    args += ['--edge_server_alloc_frac', str(edge_frac)]
    if mode:
      args += ['--mode', str(mode)]
    args += ['--round_at_n_iter', str(round)]
    subprocess.call(args)