class ABCRecSysModel:
  rand_seed = 1

  def fit(self, user_ls, item_ls, value_ls):
    raise NotImplementedError('fit method is not implemented')

  def pred(self, u, v):
    raise NotImplementedError('pred method is not implemented')

  def rec(self, u, n=1):
    raise NotImplementedError('rec method is not implemented')

  def aggregate(self, o: "ABCRecSysModel"):
    raise NotImplementedError('aggregate method is not implemented')
