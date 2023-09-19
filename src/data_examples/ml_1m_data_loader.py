import numpy as np
from src.utils.request_utils import get_request
from src.utils.archive_utils import unload_zip_from_io
from pathlib import Path
import pandas as pd


def _date_parser(datestr):
  __month_shortname_index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  s = datestr.split('-')
  if len(s) != 3:
    return pd.NaT
  s[0] = '{:02}'.format(int(s[0]))
  s[1] = '{:02}'.format(__month_shortname_index.index(s[1]) + 1)
  s[2], s[0] = s[0], s[2]
  return '-'.join(s)

class ExampleDataLoader:
  # MovieLens data loader
  # permalink: https://grouplens.org/datasets/movielens/latest/

  archive_url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
  archive_dir = 'data/ex2_1m'

  data_dir = {
    'users': {
      'filename': 'users.dat',
      'delimiter': '::',
      'encoding': 'latin-1',
      'headers': ['user_id', 'gender', 'age', 'occupation', 'zip_code'],
      'types': ['int', 'string', 'int', 'int', 'string']
    },
    'ratings': {
      'filename': 'ratings.dat',
      'delimiter': '::',
      'encoding': 'latin-1',
      'headers': ['user_id', 'movie_id', 'rating', 'unix_timestamp'],
      'types': ['int', 'int', 'float64', 'uint64']
    },
    'movies': {
      'filename': 'movies.dat',
      'delimiter': '::',
      'encoding': 'latin-1',
      'headers': ['movie_id', 'title', 'genres'],
      'types': ['int', 'string', 'string']
    },
  }

  data_feature_cols = {
    'users': ['user_id', 'gender', 'age', 'occupation', 'zip_code'],
    'ratings': ['rating'], # 'unix_timestamp'
    'movies' : ['movie_id'] # 'release_date', 'video_release_date'
  }

  data_occupation_map = [
    'other',
    'academic/educator',
    'artist',
    'clerical/admin',
    'college/grad student',
    'customer service',
    'doctor/health care',
    'executive/managerial',
    'farmer',
    'homemaker',
    'K-12 student',
    'lawyer',
    'programmer',
    'retired',
    'sales/marketing',
    'scientist',
    'self-employed',
    'technician/engineer',
    'tradesman/craftsman',
    'unemployed',
    'writer'
  ]

  data_genres_map = [
    'Action',
    'Adventure',
    'Animation',
    'Children\'s',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
  ]

  data_converters = {
    'users': {
      'user_id': lambda x: str(int(x)-1)
    },
    'ratings': {
      # 'unix_timestamp': lambda x: np.datetime64('1970-01-01') + np.timedelta64(x, 's'),
      'user_id': lambda x: str(int(x)-1),
      'movie_id': lambda x: str(int(x)-1),
    },
    'movies': {
      'movie_id': lambda x: str(int(x)-1),
      'release_date': _date_parser
    }
  }


  def __init__(self):
    self.df = None
    self.df_users = None
    self.df_ratings = None
    self.df_movies = None

    self.mapped_features = {
      'users': {
        'user_id': {'incr': -1}
      },
      'ratings': {
        'user_id': {'incr': -1},
        'movie_id': {'incr': -1},
      },
      'movies': {
        'movie_id': {'incr': -1},
      }
    }

    self._header_dict = {}
    self._header_dict_is_loaded = False

  def download(self, to_folder=None):
    if not to_folder or to_folder is None:
      to_folder = self.archive_dir
    else:
      self.archive_dir = to_folder
    b = get_request(self.archive_url)
    unload_zip_from_io(b, to_folder)
    b.close()
    return self

  def load(self):
    for tb_name, tb_info in self.data_dir.items():
      data_glob_path = list(Path(self.archive_dir).rglob(tb_info['filename']))
      data_file_path = next(filter(lambda x: x.is_file(), data_glob_path))
      df = pd.read_csv(
        data_file_path,
        names=tb_info['headers'],
        sep=tb_info['delimiter'],
        encoding=tb_info['encoding'],
        dtype=dict(zip(tb_info['headers'], tb_info['types'])),
        converters=self.data_converters.get(tb_name)
      )
      setattr(self, f'df_{tb_name}', df)
    return self

  def remap(self):
    self.mapped_features['ratings'] = {}
    self.mapped_features['ratings']['rating'] = {}
    __rating_max = np.argmax(self.df_ratings['rating'].to_numpy())
    __rating_min = np.argmin(self.df_ratings['rating'].to_numpy())
    __rating_len = __rating_max - __rating_min

    self.mapped_features['ratings']['rating']['max'] = __rating_max
    self.mapped_features['ratings']['rating']['min'] = __rating_min
    self.mapped_features['ratings']['rating']['len'] = __rating_len

    self.df_ratings['rating'] = self.df_ratings['rating'].apply(lambda x: (x - __rating_min) / __rating_len)

    self.df_users['occupation'] = self.df_users['occupation'].apply(lambda x: self.data_occupation_map[int(x)])

    return self

  def clean(self):
    return self


  def merge(self):
    self.df = self.df_ratings.merge(self.df_users, on='user_id')
    self.df = self.df.merge(self.df_movies, on='movie_id')
    return self

  def trunc(self):
    feature_cols = [i for j in self.data_feature_cols.values() for i in j]
    self.df = self.df[feature_cols]
    return self


  def get_feature_values(self, feature_col:str, tb_name=None):
    if tb_name is not None:
      values = getattr(self, f'df_{tb_name}')[feature_col].unique()
    else:
      values = self.df[feature_col].unique()

    return list(values)

  def _load_header_dtype(self):
    for meta in self.data_dir.values():
      for n, v in enumerate(meta['headers']):
        self._header_dict[v] = meta['types'][n] 

  def fetch_datatype(self, col_name):
    if not self._header_dict_is_loaded:
      self._load_header_dtype()
    return self._header_dict.get(col_name)