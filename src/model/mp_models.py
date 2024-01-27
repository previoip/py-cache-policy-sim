import os
import yaml
import typing as t
from functools import partial
from enum import Enum
from functools import wraps
from src.logger_helper import spawn_logger, init_default_logger
from src.data_loaders.ml_data_loader import ExampleDataLoader
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

  n = len(other_models) + 1
  for other_model in other_models:
    self.reg_weight += other_model.reg_weight
    self.interaction_matrix += other_model.interaction_matrix

  self.reg_weight /= n
  self.interaction_matrix /= n

  # print('reg_weight', self.reg_weight, type(self.reg_weight))
  # print('interaction_matrix', self.interaction_matrix, type(self.interaction_matrix))

EASE.fl_agg = __mp_EASE_flAgg


@__mpd_assert_arg_type
def __mp_EASE_flDelegate(self, other_models: t.Iterable):
  print('called __mp_EASE_flDelegate')

  for other_model in other_models:
    other_model.reg_weight = self.reg_weight
    other_model.interaction_matrix = self.interaction_matrix.copy()

EASE.fl_delegate_to = __mp_EASE_flDelegate


# ==========================================
# Monkey Patch: FM

@__mpd_assert_arg_type
def __mp_FM_flAgg(self, other_models: t.Iterable):
  print('called __mp_FM_flAgg')
  
  attrs = ['embed_user', 'embed_item', 'u_bias', 'i_bias', 'bias_']
  agg_avg_torch_tensors(self, other_models, attrs)

  # print('embed_user', self.embed_user, type(self.embed_user))
  # print('embed_item', self.embed_item, type(self.embed_item))
  # print('u_bias', self.u_bias, type(self.u_bias))
  # print('i_bias', self.i_bias, type(self.i_bias))
  # print('bias_', self.bias_, type(self.bias_))

FM.fl_agg = __mp_FM_flAgg


@__mpd_assert_arg_type
def __mp_FM_flDelegate(self, other_models: t.Iterable):
  print('called __mp_FM_flDelegate')
  for other_model in other_models:
    other_model.embed_user.load_state_dict(self.embed_user.state_dict())
    other_model.embed_item.load_state_dict(self.embed_item.state_dict())
    other_model.u_bias.load_state_dict(self.u_bias.state_dict())
    other_model.i_bias.load_state_dict(self.i_bias.state_dict())
    other_model.bias_.load_state_dict(self.bias_.state_dict())

FM.fl_delegate_to = __mp_FM_flDelegate


# ==========================================
# Monkey Patch: Item2Vec

@__mpd_assert_arg_type
def __mp_Item2Vec_flAgg(self, other_models: t.Iterable):
  print('called __mp_Item2Vec_flAgg')
  attrs = ['user_embedding', 'shared_embedding']
  agg_avg_torch_tensors(self, other_models, attrs)

  # print('user_embedding', self.user_embedding, type(self.user_embedding))
  # print('shared_embedding', self.shared_embedding, type(self.shared_embedding))

Item2Vec.fl_agg = __mp_Item2Vec_flAgg


@__mpd_assert_arg_type
def __mp_Item2Vec_flDelegate(self, other_models: t.Iterable):
  print('called __mp_Item2Vec_flDelegate')
  for other_model in other_models:
    other_model.user_embedding.load_state_dict(self.user_embedding.state_dict())
    other_model.shared_embedding.load_state_dict(self.shared_embedding.state_dict())

Item2Vec.fl_delegate_to = __mp_Item2Vec_flDelegate


# ==========================================
# Monkey Patch: ItemKNNCF

@__mpd_assert_arg_type
def __mp_ItemKNNCF_flAgg(self, other_models: t.Iterable):
  print('called __mp_ItemKNNCF_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
  print('pred_mat', self.pred_mat, type(self.pred_mat))

ItemKNNCF.fl_agg = __mp_ItemKNNCF_flAgg


@__mpd_assert_arg_type
def __mp_ItemKNNCF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_ItemKNNCF_flDelegate')
  raise NotImplementedError('fl patch is not yet implemented')

ItemKNNCF.fl_delegate_to = __mp_ItemKNNCF_flDelegate


# ==========================================
# Monkey Patch: LightGCN

@__mpd_assert_arg_type
def __mp_LightGCN_flAgg(self, other_models: t.Iterable):
  print('called __mp_LightGCN_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
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
  attrs = ['embed_user_GMF', 'embed_user_MLP', 'embed_item_GMF', 'embed_item_MLP', 'predict_layer']
  agg_avg_torch_tensors(self, other_models, attrs)

  # print('embed_user_GMF', self.embed_user_GMF, type(self.embed_user_GMF)) 
  # print('embed_user_MLP', self.embed_user_MLP, type(self.embed_user_MLP)) 
  # print('embed_item_GMF', self.embed_item_GMF, type(self.embed_item_GMF))
  # print('embed_item_MLP', self.embed_item_MLP, type(self.embed_item_MLP))
  # print('predict_layer', self.predict_layer, type(self.predict_layer))

NeuMF.fl_agg = __mp_NeuMF_flAgg

@__mpd_assert_arg_type
def __mp_NeuMF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_NeuMF_flDelegate')
  for other_model in other_models:
    other_model.embed_user_GMF.load_state_dict(self.embed_user_GMF.state_dict())
    other_model.embed_user_MLP.load_state_dict(self.embed_user_MLP.state_dict())
    other_model.embed_item_GMF.load_state_dict(self.embed_item_GMF.state_dict())
    other_model.embed_item_MLP.load_state_dict(self.embed_item_MLP.state_dict())
    other_model.predict_layer.load_state_dict(self.predict_layer.state_dict())

NeuMF.fl_delegate_to = __mp_NeuMF_flDelegate


# ==========================================
# Monkey Patch: NFM

@__mpd_assert_arg_type
def __mp_NFM_flAgg(self, other_models: t.Iterable):
  print('called __mp_NFM_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
  print('deep_layers', self.deep_layers, type(self.deep_layers))
  print('prediction', self.prediction, type(self.prediction))

NFM.fl_agg = __mp_NFM_flAgg


@__mpd_assert_arg_type
def __mp_NFM_flDelegate(self, other_models: t.Iterable):
  print('called __mp_NFM_flDelegate')
  raise NotImplementedError('fl patch is not yet implemented')

NFM.fl_delegate_to = __mp_NFM_flDelegate


# ==========================================
# Monkey Patch: NGCF

@__mpd_assert_arg_type
def __mp_NGCF_flAgg(self, other_models: t.Iterable):
  print('called __mp_NGCF_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
  print('embed_user', self.embed_user, type(self.embed_user))
  print('embed_item', self.embed_item, type(self.embed_item))
  print('gnn_layers', self.gnn_layers, type(self.gnn_layers))

NGCF.fl_agg = __mp_NGCF_flAgg


@__mpd_assert_arg_type
def __mp_NGCF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_NGCF_flDelegate')
  raise NotImplementedError('fl patch is not yet implemented')

NGCF.fl_delegate_to = __mp_NGCF_flDelegate


# ==========================================
# Monkey Patch: MostPop

@__mpd_assert_arg_type
def __mp_MostPop_flAgg(self, other_models: t.Iterable):
  print('called __mp_MostPop_flAgg')
  n = len(other_models) + 1
  for other_model in other_models:
    self.item_score += other_model.item_score
  self.item_score /= n

  # print('item_score', self.item_score, type(self.item_score))

MostPop.fl_agg = __mp_MostPop_flAgg


@__mpd_assert_arg_type
def __mp_MostPop_flDelegate(self, other_models: t.Iterable):
  print('called __mp_MostPop_flDelegate')
  for other_model in other_models:
    other_model.item_score = self.item_score.copy()

MostPop.fl_delegate_to = __mp_MostPop_flDelegate


# ==========================================
# Monkey Patch: PureSVD

@__mpd_assert_arg_type
def __mp_PureSVD_flAgg(self, other_models: t.Iterable):
  print('called __mp_PureSVD_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
  print('user_vec', self.user_vec, type(self.user_vec))
  print('item_vec', self.item_vec, type(self.item_vec))

PureSVD.fl_agg = __mp_PureSVD_flAgg


@__mpd_assert_arg_type
def __mp_PureSVD_flDelegate(self, other_models: t.Iterable):
  print('called __mp_PureSVD_flDelegate')
  raise NotImplementedError('fl patch is not yet implemented')

PureSVD.fl_delegate_to = __mp_PureSVD_flDelegate


# ==========================================
# Monkey Patch: SLiM

@__mpd_assert_arg_type
def __mp_SLiM_flAgg(self, other_models: t.Iterable):
  print('called __mp_SLiM_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
  print('A_tilde', self.A_tilde, type(self.A_tilde))
  print('w_sparse', self.w_sparse, type(self.w_sparse))

SLiM.fl_agg = __mp_SLiM_flAgg


@__mpd_assert_arg_type
def __mp_SLiM_flDelegate(self, other_models: t.Iterable):
  print('called __mp_SLiM_flDelegate')
  raise NotImplementedError('fl patch is not yet implemented')

SLiM.fl_delegate_to = __mp_SLiM_flDelegate


# ==========================================
# Monkey Patch: VAECF

@__mpd_assert_arg_type
def __mp_VAECF_flAgg(self, other_models: t.Iterable):
  print('called __mp_VAECF_flAgg')
  raise NotImplementedError('fl patch is not yet implemented')
  print('history_user_id', self.history_user_id, type(self.history_user_id))
  print('history_item_id', self.history_item_id, type(self.history_item_id))
  print('history_user_value', self.history_user_value, type(self.history_user_value))
  print('history_item_value', self.history_item_value, type(self.history_item_value))

VAECF.fl_agg = __mp_VAECF_flAgg


@__mpd_assert_arg_type
def __mp_VAECF_flDelegate(self, other_models: t.Iterable):
  print('called __mp_VAECF_flDelegate')
  raise NotImplementedError('fl patch is not yet implemented')

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

