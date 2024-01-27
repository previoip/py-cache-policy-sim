import os
import yaml
from src.data_loaders.ml_data_loader import ExampleDataLoader
from src.model.mp_models import model_dict
from src.logger_helper import init_default_logger
from src.model.daisyRec.daisy.utils.splitter import TestSplitter
from src.model.daisyRec.daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from src.model.daisyRec.daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from src.model.daisyRec.daisy.utils.utils import get_ur, get_history_matrix, get_inter_matrix

DAISY_LOGGER = init_default_logger()

class RecsysFL:
  """ namespace/apis used for patched daisy recsys methods """

  @classmethod
  def infer_preset(cls, daisy_config):
    """ get preset value based on daisy config """
    algo_name = daisy_config.get('algo_name', '').lower()
    if algo_name in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
      return 1
    elif algo_name in ['multi-vae']:
      return 2
    elif algo_name in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
      if algo_name in ['lightgcn', 'ngcf']:
        return 4
      return 3
    elif algo_name in ['item2vec']:
      return 5
    raise ValueError('config algo_name does not match to any preset: {}'.format(algo_name))

  @staticmethod
  def init_daisy_config(data_loder: ExampleDataLoader, algo_name=None, stdout=None):
    """ initialize daisy config from preconfigured yaml """
    config = dict()

    # get base daisy config file
    current_path = os.path.dirname(os.path.realpath(__file__))
    rel_daisy_path = '../model/daisyRec/daisy'
    basic_init_file = os.path.join(current_path, rel_daisy_path, 'assets/basic.yaml')
    basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
    config.update(basic_conf)

    # update config from config preset by algo name
    config['algo_name'] = config['algo_name'] if algo_name is None else algo_name

    algo_name = config['algo_name']
    model_init_file = os.path.join(current_path, rel_daisy_path, f'assets/{algo_name}.yaml')
    if os.path.exists(model_init_file) and os.path.isfile(model_init_file):
      model_conf = yaml.load(
        open(model_init_file), Loader=yaml.loader.SafeLoader)
      if not model_conf is None:
        config.update(model_conf)

    # various inserts based on data_loader
    users = data_loder.df_users[data_loder.uid].unique()
    items = data_loder.df_movies[data_loder.iid].unique()
    config['user_num'] = len(users)
    config['item_num'] = len(items)
    config['UID_NAME'] = data_loder.uid
    config['IID_NAME'] = data_loder.iid
    config['INTER_NAME'] = data_loder.inter
    config['TID_NAME'] = data_loder.tid
    config['TID_NAME'] = data_loder.tid

    # various inserts for federated issues from multiple devices
    config['gpu'] = ''


    if algo_name == 'nfm':
      # todo: fix batch normalization on nfm raises ValueError "expected 2D or 3D input (got 1D input)"
      config.update({'batch_norm': False})

    config['logger'] = DAISY_LOGGER if stdout is None else stdout

    return config


  class ModelRunner:
    """ runner class for training, sampling, and other procedures. binds model into instance attr """

    def __init__(self, daisy_config):
      self.wref_config_runner = daisy_config
      self.model = None
      self.sampler = None
      self.splitter = None
      self.train_proc = None

    def set_model(self, model):
      self.model = model

    def set_sampler(self, sampler):
      self.sampler = sampler

    def set_splitter(self, splitter):
      self.splitter = splitter

    def set_train_procedure(self, train_proc: callable):
      self.train_proc = train_proc

    def split(self, df):
      train_index, test_index = self.splitter.split(df)
      train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()
      uid, iid = self.wref_config_runner['UID_NAME'], self.wref_config_runner['IID_NAME']
      test_ur = get_ur(test_set, uid=uid, iid=iid)
      train_ur = get_ur(train_set, uid=uid, iid=iid)
      self.model.wref_config.update({'train_ur': train_ur})
      return train_set, test_set, test_ur, train_ur

    def train(self, train_set):
      self.train_proc(train_set)

    def train_proc_1(self, train_set):
      self.model.fit(train_set)

    def train_proc_2(self, train_set):
      uid = self.wref_config_runner['UID_NAME']
      batch_size = self.wref_config_runner['batch_size']

      history_item_id, history_item_value, _  = get_history_matrix(train_set, self.model.wref_config, row=uid)
      self.model.wref_config.update({'history_item_id': history_item_id, 'history_item_value': history_item_value})

      train_dataset = AEDataset(train_set, yield_col=uid)
      train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

      self.model.fit(train_loader)

    def train_proc_3(self, train_set):
      sampler = self.sampler(train_set, self.model.wref_config)
      batch_size = self.model.wref_config['batch_size']

      train_samples = sampler.sampling()
      train_dataset = BasicDataset(train_samples)
      train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

      self.model.fit(train_loader)

    def train_proc_4(self, train_set):
      self.model.wref_config.update({'inter_matrix': get_inter_matrix(train_set, self.model.wref_config)})
      
      sampler = self.sampler(train_set, self.model.wref_config)
      batch_size = self.model.wref_config['batch_size']

      train_samples = sampler.sampling()
      train_dataset = BasicDataset(train_samples)
      train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
      self.model.fit(train_loader)

    def train_proc_5(self, train_set):
      sampler = self.sampler(train_set, self.model.wref_config)
      batch_size = self.wref_config_runner['batch_size']

      train_samples = sampler.sampling()
      train_dataset = BasicDataset(train_samples)
      train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

      self.model.fit(train_loader)

  @classmethod
  def new_recsys_runner(cls, daisy_config):
    runner = cls.ModelRunner(daisy_config)
    preset = cls.infer_preset(daisy_config)

    model_name = runner.wref_config_runner['algo_name'].lower()
    model_class = model_dict.get(model_name)
    model = model_class(runner.wref_config_runner.copy())
    runner.set_model(model)

    runner.set_splitter(TestSplitter(runner.wref_config_runner))

    if preset == 1:
      runner.set_train_procedure(runner.train_proc_1)
    elif preset == 2:
      runner.set_train_procedure(runner.train_proc_2)
    elif preset == 3:
      runner.set_train_procedure(runner.train_proc_3)
      runner.set_sampler(BasicNegtiveSampler)
    elif preset == 4:
      runner.set_train_procedure(runner.train_proc_4)
      runner.set_sampler(BasicNegtiveSampler)
    elif preset == 5:
      runner.set_train_procedure(runner.train_proc_5)
      runner.set_sampler(SkipGramNegativeSampler)

    return runner