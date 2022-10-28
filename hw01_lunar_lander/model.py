import numpy as np
import torch
from torch import device, maximum, nn
from torch.nn import functional as F
from torch.optim import Adam


SEED = 42


class DQNModel(nn.Module):
    """ Actor (Policy) Model """

    def __init__(self, state_size: int, action_size: int):
        """ Initialize parameters and build model """
        super(DQNModel, self).__init__()
        self.seed = torch.manual_seed(SEED)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """ Build a network that maps state -> action values """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

    # def __init__(self, state_dim, action_dim, hidden_size=64):
    #     super(DQNModel, self).__init__()
    #     self.fc1 = nn.Linear(state_dim, hidden_size)
    #     self.fc2 = nn.Linear(hidden_size, hidden_size)
    #     self.fc3 = nn.Linear(hidden_size, action_dim)
    #     self.bc1 = nn.BatchNorm1d(hidden_size)
    #     self.bc2 = nn.BatchNorm1d(hidden_size)
    #     # self.dropout = nn.Dropout(p=0.35)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = self.bc1(x)
    #     x = F.relu(self.fc2(x))
    #     # x = self.dropout(x)
    #     x = self.bc2(x)
    #     x = self.fc3(x)
    #     return x
