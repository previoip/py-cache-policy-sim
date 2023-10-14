import os
import yaml
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

model_dict = {
  'ease': EASE,
  'fm': FM,
  'item2vec': Item2Vec,
  'itemknn': ItemKNNCF,
  'lightgcn': LightGCN,
  'mf': MF,
  'neumf': NeuMF,
  'nfm': NFM,
  'ngcf': NGCF,
  'mostpop': MostPop,
  'puresvd': PureSVD,
  'slim': SLiM,
  'multi-vae': VAECF,
}


def init_config(data_loder: ExampleDataLoader, param_dict=None):
  config = dict()

  current_path = os.path.dirname(os.path.realpath(__file__))
  basic_init_file = os.path.join(current_path, '../model/daisyRec/daisy/assets/basic.yaml')
  
  basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
  config.update(basic_conf)

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

  return config
