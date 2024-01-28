from src.pseudo_database import TabularPDB
from src.cache import Cache
import random



class TELECOM_EVAL:
  # attr naming convention: {kind; c, rng}_{varname}_{[super]}_{[sub]}

  class PARAMS:
    seed = 1337
    uniform_hy = 1      # kb           assumes all content have same sizes

    class USER_PARAM:
      val_h__xm           = 1           # kb           assumes all content have same sizes
      rng_gamma_r_xm      = (0.3, 1   ) # (0.03, 0.1)  
      rng_gamma_dl_xm     = (5.0, 10.0)

    class SIM_PARAM:
      val_delta__m        = 10e9
      val_delta__bs       = 50e9
      val_delta__CNC      = 100e9
      val_c__xm           = 1500e3
      val_b__m            = 20e6      # Hz            bandwidth (edge -> users)
      val_B__CS           = 100e6     # Hz    bandwidth (base -> edge)
      val_omega__         = -163      # W/hz, dBm/Hz  power density over noise
      # b_m   = 50e6  
      # P_m   = 24    # W, dBm        transmission power (edge -> users)
      # G_m           # [0, 1]        channel gain range (edge -> users)
      # B_m   = 100e6 # Hz            
      # P_cs  = 30    # W, dBm        transmission power (base -> edge)
      # G_cs          # [0, 1]        channel gain range (base -> edge)

  @staticmethod
  def _request_delay(h, R):
    return h / R

  @classmethod
  def request_delay(cls, df):
    return cls._request_delay(df['h'], df['R'])

  @staticmethod
  def _content_delivery(Vxy, h, rx, R):
    return (Vxy * h / rx) + ((1 - Vxy) * ((h / R) + (h / rx)))

  @classmethod
  def content_delivery(cls, df):
    return cls._content_delivery(df['Vxy'], df['h'], df['rx'], df['R'])

  @staticmethod
  def _communication_energy_cost(dxy, P_m, dy, P_cs):
    return (dxy * P_m) + (dy * P_cs)

  @classmethod
  def communication_energy_cost(cls, df):
    return cls._communication_energy_cost(df['dxy'], cls.DR_CONST.P_m, df['dy'], cls.DR_CONST.P_cs)



class CachingRLEnv:
  """ reinforcement learning for caching policy """
  def __init__(self, server):
    self.server = server
    self.n_look_back = 1000
    self.eps_T = 100

    self.batch_size = 128
    self.gamma = 0.99
    self.eps_start = 0.9
    self.eps_end = 0.05
    self.eps_decay = 1000
    self.tau = 0.005
    self.lr = 1e-4

    self.replay_buf = TabularPDB(
      'ReplayBuffer',
      container=list(),
      field_names = ['st', 'at', 'rt', 'st1'],
      field_dtypes = ['float', 'float', 'float', 'float']
    )

  def retrieve_past_memory(self):
    df = self.server.request_log_database_status.to_pd(use_cursor=False)
    past_buf = df.tail(self.n_look_back)
    return past_buf

  def retrieve_replay_memory(self):
    replay_buf = self.replay_buf.to_pd(use_cursor=False).tail(self.n_look_back)
    return replay_buf

  def sample(self, batch_size=0):
    if batch_size <= 0:
      batch_size = self.batch_size
    return random.sample(self.retrieve_replay_memory(), batch_size)

  def eval_latency(self, df):
    st = np.where((df['status'].apply(lambda x: str(x).lower().endswith('hit') and str(x).lower().startswith('cache'))), 1, 0)
    lat = st.sum()
    return lat

  def forward(self):
    pass

  def action(self, user_cand_pop):
    ls = sorted(user_cand_pop.copy(), key=lambda x: x[1])
    length = len(ls)
    low_len = length // 2
    ls_hi, ls_lo = ls[:length-low_len], ls[length:]
    random.shuffle(ls_lo)
    return ls_lo

  def train(self, cache_cand_pop):
    s1_cache = self.server.cache
    sn_cache = Cache(self.server.cfg.cache_maxsize, self.server.cfg.cache_maxage, lambda: 0)
    sn_cache.clear()

    for _ in range(self.eps_T):
      pass

  def infer(self, cache_cands):
    pass
