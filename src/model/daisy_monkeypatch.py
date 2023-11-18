import os
import yaml
import typing as t
from functools import partial
from enum import Enum
from functools import wraps
from src.logger_helper import spawn_logger, init_default_logger
from src.data_examples.ml_data_loader import ExampleDataLoader
from src.model.model_abc import ABCRecSysModel
from src.model.daisyRec.daisy.model.AbstractRecommender import AbstractRecommender, GeneralRecommender
from src.model.daisyRec.daisy.model.EASERecommender import EASE
from src.model.daisyRec.daisy.model.FMRecommender import FM
from src.model.daisyRec.daisy.model.Item2VecRecommender import Item2Vec
from src.model.daisyRec.daisy.model.KNNCFRecommender import ItemKNNCF
from src.model.daisyRec.daisy.model.LightGCNRecommender import LightGCN 
from src.model.daisyRec.daisy.model.MFRecommender import MF
from src.model.daisyRec.daisy.model.NeuMFRecommender import NeuMF
from src.model.daisyRec.daisy.model.NFMRecommender import NFM
from src.model.daisyRec.daisy.model.NGCFRecommender import NGCF
from src.model.daisyRec.daisy.model.PopRecommender import MostPop
from src.model.daisyRec.daisy.model.PureSVDRecommender import PureSVD
from src.model.daisyRec.daisy.model.SLiMRecommender import SLiM
from src.model.daisyRec.daisy.model.VAECFRecommender import VAECF
from src.model.daisyRec.daisy.utils.splitter import TestSplitter
from src.model.daisyRec.daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler
from src.model.daisyRec.daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from src.model.daisyRec.daisy.utils.utils import ensure_dir, get_ur, get_history_matrix, build_candidates_set, get_inter_matrix

# DAISY_LOGGER = spawn_logger('daisy_log')
DAISY_LOGGER = init_default_logger()

class RECSYS_MODEL_ENUM:
  ease      = 'ease'
  fm        = 'fm'
  item2vec  = 'item2vec'
  itemknn   = 'itemknn'
  lightgcn  = 'lightgcn'
  mf        = 'mf'
  neumf     = 'neumf'
  nfm       = 'nfm'
  ngcf      = 'ngcf'
  mostpop   = 'mostpop'
  puresvd   = 'puresvd'
  slim      = 'slim'
  multi_vae = 'multi-vae'

# ==========================================
# federated aggregation method patches
# ==========================================

# Misc helpers

def agg_avg_torch_tensors(model, other_models, nn_attr_names):
  n = len(other_models) + 1
  for attr in nn_attr_names:
    model_unloaded_state = getattr(model, attr).state_dict()
    other_model_unloaded_states = map(lambda tt: getattr(tt, attr).state_dict(), other_models)
    for layer in model_unloaded_state:
      for other_state in other_model_unloaded_states:
        model_unloaded_state[layer] += other_state[layer]
    model_unloaded_state[layer] /= n
    getattr(model, attr).load_state_dict(model_unloaded_state)

def __mpd_assert_arg_type(fn):
  # asserts arg type decorator
  @wraps(fn)
  def wrapper(self, o_ls: t.Iterable):
    for o in o_ls:
      if not isinstance(self, o.__class__):
        raise TypeError('args does not have same type')
    return fn(self, o_ls)
  return wrapper

# ==========================================
# Monkey Patch: EASE

@__mpd_assert_arg_type
def __mp_EASE_flAgg(self, other_models: t.Iterable):
  print('called __mp_EASE_flAgg')
  print('reg_weight', self.reg_weight, type(self.reg_weight))
  # print('item_similarity', self.item_similarity, type(self.item_similarity))
  print('interaction_matrix', self.interaction_matrix, type(self.interaction_matrix))
  
EASE.fl_agg = __mp_EASE_flAgg

@__mpd_assert_arg_type
def __mp_EASE_flDelegate(self, other_models: t.Iterable):
  print('called __mp_EASE_flDelegate')

EASE.fl_delegate_to = __mp_EASE_flDelegate


# ==========================================
# Monkey Patch: FM

@__mpd_assert_arg_type
def __mp_FM_flAgg(self, other_models: t.Iterable):
  print('called __mp_FM_flAgg')
  print('embed_user', self.embed_user, type(self.embed_user))
  print('embed_item', self.embed_item, type(self.embed_item))
  print('u_bias', self.u_bias, type(self.u_bias))
  print('i_bias', self.i_bias, type(self.i_bias))
  print('bias_', self.bias_, type(self.bias_))

FM.fl_agg = __mp_FM_flAgg

@__mpd_assert_arg_type
def __mp_FM_flDelegate(self, other_models: t.Iterable):
  print('called __mp_FM_flDelegate')

FM.fl_delegate_to = __mp_FM_flDelegate


# ==========================================
# Monkey Patch: Item2Vec

@__mpd_assert_arg_type
def __mp_Item2Vec_flAgg(self, other_models: t.Iterable):
  print('called __mp_Item2Vec_flAgg')
  print('user_embedding', self.user_embedding, type(self.user_embedding))
  print('shared_embedding', self.shared_embedding, type(self.shared_embedding))

Item2Vec.fl_agg = __mp_Item2Vec_flAgg

@__mpd_assert_arg_type
def __mp_Item2Vec_flDelegate(self, other_models: t.Iterable):
  print('called __mp_Item2Vec_flDelegate')

Item2Vec.fl_delegate_to = __mp_Item2Vec_flDelegate


# ==========================================
# Monkey Patch: ItemKNNCF

@__mpd_assert_arg_type
def __mp_ItemKNNCF_flAgg(self, other_models: t.Iterable):
  print('called __mp_ItemKNNCF_flAgg')
  print('pred_mat', self.pred_mat, type(self.pred_mat))

ItemKNNCF.fl_agg = __mp_ItemKNNCF_flAgg

@__mpd_assert_arg_type
def __mp_ItemKNNCF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_ItemKNNCF_flDelegate')

ItemKNNCF.fl_delegate_to = __mp_ItemKNNCF_flDelegate


# ==========================================
# Monkey Patch: LightGCN

@__mpd_assert_arg_type
def __mp_LightGCN_flAgg(self, other_models: t.Iterable):
  print('called __mp_LightGCN_flAgg')
  print('restore_user_e', self.restore_user_e, type(self.restore_user_e))
  print('restore_item_e', self.restore_item_e, type(self.restore_item_e))

LightGCN.fl_agg = __mp_LightGCN_flAgg

@__mpd_assert_arg_type
def __mp_LightGCN_flDelegate(self, other_models: t.Iterable):
  print('called __mp_LightGCN_flDelegate')

LightGCN.fl_delegate_to = __mp_LightGCN_flDelegate


# ==========================================
# Monkey Patch: MF

@__mpd_assert_arg_type
def __mp_MF_flAgg(self, other_models: t.Iterable):
  print('called __mp_MF_flAgg')

  attrs = ['embed_user', 'embed_item']
  agg_avg_torch_tensors(self, other_models, attrs)

  # print('embed_user', self.embed_user, type(self.embed_user))
  # print('embed_item', self.embed_item, type(self.embed_item))


MF.fl_agg = __mp_MF_flAgg


@__mpd_assert_arg_type
def __mp_MF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_MF_flDelegate')
  for other_model in other_models:
    other_model.embed_user.load_state_dict(self.embed_user.state_dict())
    other_model.embed_item.load_state_dict(self.embed_item.state_dict())

MF.fl_delegate_to = __mp_MF_flDelegate


# ==========================================
# Monkey Patch: NeuMF

@__mpd_assert_arg_type
def __mp_NeuMF_flAgg(self, other_models: t.Iterable):
  print('called __mp_NeuMF_flAgg')
  print('embed_user_GMF', self.embed_user_GMF, type(self.embed_user_GMF)) 
  print('embed_user_MLP', self.embed_user_MLP, type(self.embed_user_MLP)) 
  print('embed_item_GMF', self.embed_item_GMF, type(self.embed_item_GMF))
  print('embed_item_MLP', self.embed_item_MLP, type(self.embed_item_MLP))
  print('predict_layer', self.predict_layer, type(self.predict_layer))


NeuMF.fl_agg = __mp_NeuMF_flAgg

@__mpd_assert_arg_type
def __mp_NeuMF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_NeuMF_flDelegate')

NeuMF.fl_delegate_to = __mp_NeuMF_flDelegate


# ==========================================
# Monkey Patch: NFM

@__mpd_assert_arg_type
def __mp_NFM_flAgg(self, other_models: t.Iterable):
  print('called __mp_NFM_flAgg')
  print('deep_layers', self.deep_layers, type(self.deep_layers))
  print('prediction', self.prediction, type(self.prediction))

NFM.fl_agg = __mp_NFM_flAgg

@__mpd_assert_arg_type
def __mp_NFM_flDelegate(self, other_models: t.Iterable):
  print('called __mp_NFM_flDelegate')

NFM.fl_delegate_to = __mp_NFM_flDelegate


# ==========================================
# Monkey Patch: NGCF

@__mpd_assert_arg_type
def __mp_NGCF_flAgg(self, other_models: t.Iterable):
  print('called __mp_NGCF_flAgg')
  print('embed_user', self.embed_user, type(self.embed_user))
  print('embed_item', self.embed_item, type(self.embed_item))
  print('gnn_layers', self.gnn_layers, type(self.gnn_layers))

NGCF.fl_agg = __mp_NGCF_flAgg

@__mpd_assert_arg_type
def __mp_NGCF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_NGCF_flDelegate')

NGCF.fl_delegate_to = __mp_NGCF_flDelegate


# ==========================================
# Monkey Patch: MostPop

@__mpd_assert_arg_type
def __mp_MostPop_flAgg(self, other_models: t.Iterable):
  print('called __mp_MostPop_flAgg')
  print('item_score', self.item_score, type(self.item_score))

MostPop.fl_agg = __mp_MostPop_flAgg

@__mpd_assert_arg_type
def __mp_MostPop_flDelegate(self, other_models: t.Iterable):
  print('called __mp_MostPop_flDelegate')

MostPop.fl_delegate_to = __mp_MostPop_flDelegate


# ==========================================
# Monkey Patch: PureSVD

@__mpd_assert_arg_type
def __mp_PureSVD_flAgg(self, other_models: t.Iterable):
  print('called __mp_PureSVD_flAgg')
  print('user_vec', self.user_vec, type(self.user_vec))
  print('item_vec', self.item_vec, type(self.item_vec))

PureSVD.fl_agg = __mp_PureSVD_flAgg

@__mpd_assert_arg_type
def __mp_PureSVD_flDelegate(self, other_models: t.Iterable):
  print('called __mp_PureSVD_flDelegate')

PureSVD.fl_delegate_to = __mp_PureSVD_flDelegate


# ==========================================
# Monkey Patch: SLiM

@__mpd_assert_arg_type
def __mp_SLiM_flAgg(self, other_models: t.Iterable):
  print('called __mp_SLiM_flAgg')
  print('A_tilde', self.A_tilde, type(self.A_tilde))
  print('w_sparse', self.w_sparse, type(self.w_sparse))

SLiM.fl_agg = __mp_SLiM_flAgg

@__mpd_assert_arg_type
def __mp_SLiM_flDelegate(self, other_models: t.Iterable):
  print('called __mp_SLiM_flDelegate')

SLiM.fl_delegate_to = __mp_SLiM_flDelegate


# ==========================================
# Monkey Patch: VAECF

@__mpd_assert_arg_type
def __mp_VAECF_flAgg(self, other_models: t.Iterable):
  print('called __mp_VAECF_flAgg')
  print('history_user_id', self.history_user_id, type(self.history_user_id))
  print('history_item_id', self.history_item_id, type(self.history_item_id))
  print('history_user_value', self.history_user_value, type(self.history_user_value))
  print('history_item_value', self.history_item_value, type(self.history_item_value))

VAECF.fl_agg = __mp_VAECF_flAgg

@__mpd_assert_arg_type
def __mp_VAECF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_VAECF_flDelegate')

VAECF.fl_delegate_to = __mp_VAECF_flDelegate



# ==========================================
# federated aggregation method patches _ end
# ==========================================

model_dict = {
  RECSYS_MODEL_ENUM.ease      : EASE,
  RECSYS_MODEL_ENUM.fm        : FM,
  RECSYS_MODEL_ENUM.item2vec  : Item2Vec,
  RECSYS_MODEL_ENUM.itemknn   : ItemKNNCF,
  RECSYS_MODEL_ENUM.lightgcn  : LightGCN,
  RECSYS_MODEL_ENUM.mf        : MF,
  RECSYS_MODEL_ENUM.neumf     : NeuMF,
  RECSYS_MODEL_ENUM.nfm       : NFM,
  RECSYS_MODEL_ENUM.ngcf      : NGCF,
  RECSYS_MODEL_ENUM.mostpop   : MostPop,
  RECSYS_MODEL_ENUM.puresvd   : PureSVD,
  RECSYS_MODEL_ENUM.slim      : SLiM,
  RECSYS_MODEL_ENUM.multi_vae : VAECF,
}

def init_config(data_loder: ExampleDataLoader, param_dict=None, algo_name=None):
  config = dict()

  current_path = os.path.dirname(os.path.realpath(__file__))
  basic_init_file = os.path.join(current_path, '../model/daisyRec/daisy/assets/basic.yaml')
  
  basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
  config.update(basic_conf)

  config['algo_name'] = config['algo_name'] if algo_name is None else algo_name
  algo_name = config['algo_name']

  model_init_file = os.path.join(current_path, f'../model/daisyRec/daisy/assets/{algo_name}.yaml')
  model_conf = yaml.load(
    open(model_init_file), Loader=yaml.loader.SafeLoader)
  if model_conf is not None:
    config.update(model_conf)

  if param_dict is not None:
    config.update(param_dict)

  config['UID_NAME'] = data_loder.uid
  config['IID_NAME'] = data_loder.iid
  config['INTER_NAME'] = data_loder.inter
  config['TID_NAME'] = data_loder.tid
  config['logger'] = DAISY_LOGGER

  return config


class _ModelTrainRunner:
  def __init__(self, daisy_config):
    self._conf_ref = daisy_config
    self.splitter = TestSplitter(daisy_config)

  def set_splitter(daisy_config):
    self.splitter = TestSplitter(daisy_config)

  def split(self, df):
    train_index, test_index = self.splitter.split(df)
    train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    self._conf_ref.update({'train_ur': total_train_ur})

    return train_set, test_set, test_ur, total_train_ur

  def _train_preset1(self):
    def wrapper(model, config, train_set):
      model.fit(train_set)
    return wrapper

  def _train_preset2(self):
    def wrapper(model, config, train_set):
      history_item_id, history_item_value, _  = get_history_matrix(train_set, config, row='user_id')
      model.config['history_item_id'], model.config['history_item_value'] = history_item_id, history_item_value
      train_dataset = AEDataset(train_set, yield_col=config['UID_NAME'])
      train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
      model.fit(train_loader)
    return wrapper

  def _train_preset3(self):
    def wrapper(model, config, train_set):
      sampler = BasicNegtiveSampler(train_set, config)
      train_samples = sampler.sampling()
      train_dataset = BasicDataset(train_samples)
      train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
      model.fit(train_loader)
    return wrapper

  def _train_preset4(self):
    def wrapper(model, config, train_set):
      model.config.update({'inter_matrix': get_inter_matrix(train_set, config)})
      sampler = BasicNegtiveSampler(train_set, config)
      train_samples = sampler.sampling()
      train_dataset = BasicDataset(train_samples)
      train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
      model.fit(train_loader)
    return wrapper

  def _train_preset5(self):
    def wrapper(model, config, train_set):
      sampler = SkipGramNegativeSampler(train_set, config)
      train_samples = sampler.sampling()
      train_dataset = BasicDataset(train_samples)
      train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
      model.fit(train_loader)
    return wrapper

  def get_train_runner(self, daisy_config):
    algo_name = daisy_config['algo_name'].lower()

    if algo_name in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
      return self._train_preset1

    elif algo_name in ['multi-vae']:
      return self._train_preset2

    elif algo_name in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
      if algo_name in ['lightgcn', 'ngcf']:
        return self._train_preset4
      return self._train_preset3

    elif algo_name in ['item2vec']:
      return self._train_preset5


def build_model_constructor(daisy_config):
  algo_name = daisy_config['algo_name'].lower()
  runner = _ModelTrainRunner(daisy_config)

  model = model_dict.get(algo_name)

  if model is None:
    raise ValueError()

  model_constructor = partial(model, daisy_config)

  return model_constructor, runner