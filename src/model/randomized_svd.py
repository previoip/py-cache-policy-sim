import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from src.model.model_abc import ABCRecSysModel

class RandomizedSVD(ABCRecSysModel):
  def __init__(self, latent_factor):
    self.latent_factor = latent_factor
    self.U = None
    self.V = None
    self.s = None
    self.csr_mat = None

  def _unique(self, ls):
    t = []
    for i in ls:
      if i not in t:
        t.append(i)
    return t

  def _preproc(self, user_ls, item_ls, value_ls, csr_shape=None):

    if csr_shape is None:
      u_user_ls, u_item_ls = self._unique(user_ls), self._unique(item_ls)
      csr_shape=(len(u_user_ls) + 1, len(u_item_ls) + 1)

    self.csr_mat = sp.csr_matrix(
      (value_ls, (user_ls, item_ls))
      # shape=csr_shape
    )


  def fit(self, user_ls, item_ls, value_ls, csr_shape=None):
    self._preproc(user_ls, item_ls, value_ls, csr_shape)
    U, s, Vt = randomized_svd(
      self.csr_mat,
      n_components = self.latent_factor,
      random_state = self.rand_seed
    )

    s_Vt = sp.diags(s) * Vt
    self.U = U
    self.V = s_Vt.T
    self.s = s

  def pred(self, u, v):
    return self.U[u, :].dot(self.V[v, :])

  def rec(self, u, num=1):
    scores = self.U[u, :].dot(self.V.T)
    return np.argsort(-scores)[:num]

