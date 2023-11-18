from main import SIM_MODE_ENUM, RECSYS_MODEL_ENUM
import subprocess
import itertools
from collections import namedtuple

BASE_ARGS = ['python', 'main.py']

PRUNE_PREVIOUS_RESULTS = True
FRACS_RES = 1
CUTOFF = 2000

ARCHIVE_FNAME = 'archive.zip'
HIST_FNAME = 'hist.json'
LOG_FOLDER = 'log'

class SIMUL_CONF:
  nt_template = namedtuple('SimConf', field_names=['mode', 'model', 'edge_frac', 'round'])

  modes = [
    # SIM_MODE_ENUM.cache_aside,
    # SIM_MODE_ENUM.localized,
    # SIM_MODE_ENUM.centralized,
    SIM_MODE_ENUM.federated
  ]
  edge_fracs = map(lambda x: x/FRACS_RES, range(1, FRACS_RES+1))

  models = [
    None,
    RECSYS_MODEL_ENUM.ease,
    RECSYS_MODEL_ENUM.fm,
    RECSYS_MODEL_ENUM.item2vec,
    RECSYS_MODEL_ENUM.itemknn,
    RECSYS_MODEL_ENUM.lightgcn,
    RECSYS_MODEL_ENUM.mf,
    RECSYS_MODEL_ENUM.mostpop,
    RECSYS_MODEL_ENUM.multi_vae,
    RECSYS_MODEL_ENUM.neumf,
    RECSYS_MODEL_ENUM.nfm,
    RECSYS_MODEL_ENUM.ngcf,
    RECSYS_MODEL_ENUM.puresvd,
    # RECSYS_MODEL_ENUM.slim, # Could not get it to work properly
  ]

  rounds = [1000] 

# helpers 

def ordered_set(iterable):
  temp = []
  for i in iterable:
    if i in temp:
      continue
    temp.append(i)
  return temp

def proc_archive_results():
  import os, shutil, zipfile, importlib

  compression = zipfile.ZIP_STORED
  if not importlib.util.find_spec('bz2') is None:
    compression = zipfile.ZIP_BZIP2
  elif not importlib.util.find_spec('zlib') is None:
    compression = zipfile.ZIP_DEFLATED

  if os.path.exists(ARCHIVE_FNAME) and os.path.isfile(ARCHIVE_FNAME):
    print('removing previous archive:', ARCHIVE_FNAME)
    os.remove(ARCHIVE_FNAME)

  with zipfile.ZipFile(ARCHIVE_FNAME, 'w', compression=compression) as zf:
    print('zipping results into', ARCHIVE_FNAME, 'compression', compression)
    for dirname, _, files in os.walk(LOG_FOLDER):
      zf.write(dirname)
      for filename in files:
        zf.write(os.path.join(dirname, filename))
    if os.path.exists(HIST_FNAME) and os.path.isfile(HIST_FNAME):
      zf.write(HIST_FNAME)


def proc_del_sim_results():
  import os, shutil, zipfile
  if os.path.exists(HIST_FNAME) and os.path.isfile(HIST_FNAME):
    print('removing previous hist.json')
    os.remove(HIST_FNAME)
  if os.path.exists(LOG_FOLDER) and os.path.isdir(LOG_FOLDER):
    print('removing previous logs')
    shutil.rmtree(LOG_FOLDER)

if __name__ == '__main__':
  if PRUNE_PREVIOUS_RESULTS:
    proc_archive_results()
    proc_del_sim_results()

  perm = itertools.product(*[
    SIMUL_CONF.modes,
    SIMUL_CONF.models,
    SIMUL_CONF.edge_fracs,
    SIMUL_CONF.rounds,
  ])
  perm = map(lambda x: SIMUL_CONF.nt_template(*x), perm)
  perm = filter(lambda x: (x.mode == SIM_MODE_ENUM.cache_aside and x.model == None) or (x.mode != SIM_MODE_ENUM.cache_aside and x.model != None), perm)
  perm = list(perm)

  print()
  print('Batched simulation configurations:')
  for i in ordered_set(map(lambda x: (x.mode, x.model), perm)):
    print('\t - ', i)
  print('rounds:', SIMUL_CONF.rounds)
  print('fraction resolution:', FRACS_RES)
  print('number of simulation:', len(perm))
  print()
  print('do you wish to continue?')
  if input('[Y/n] ') != 'Y':
    exit()

  for n, conf in enumerate(perm):
    print()
    print('running sim', f'{n+1}/{len(perm)}', 'config:', conf)
    print()

    mode, model, edge_frac, _round = conf

    args = BASE_ARGS.copy()
    args += ['--trial_cutoff', str(CUTOFF)] # comment these out to simulate full dataset
    args += ['--conf_prfx', f'{mode}_{model}_r{_round}']
    args += ['--conf_name', f'{mode}_{model}_r{_round}_{edge_frac}'.replace('.', '')]
    args += ['--edge_server_alloc_frac', str(edge_frac)]
    if mode:
      args += ['--mode', str(mode)]
    args += ['--round_at_n_iter', str(_round)]
    if model:
      args += ['--recsys_model_name', model]

    errno = subprocess.call(args)
    if errno != 0:
      print()
      print('sim exited with non-zero err code', errno)
      print('config:', conf)
      exit()