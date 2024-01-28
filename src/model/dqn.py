
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class QNN(nn.Module):
  """ deep Q-nn impl """

  def __init__(self, observ_num, action_num):
    super(DQN, self).__init__()
    self.lin1 = nn.Linear(observ_num, 128)
    self.lin2 = nn.Linear(128, 64)
    self.lin3 = nn.Linear(64, action_num)

  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    return self.lin3(x)


class ReplayBuffer:
  """ replay buffer """

  def __init__(self, l=100):
    self.buf = deque([], maxlen=l)

  def push(self, i):
    self.buf.append(i)

if __name__ == '__main__':
  actions = []