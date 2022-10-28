import os
import sys
import random

import numpy as np
import torch
from torch import device, maximum, nn
from torch.nn import functional as F
from torch.optim import Adam

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))


def set_seed(seed):
  """ Set seed """
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """ Initialize parameters and build model """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)



class Agent:
    def __init__(self):
        set_seed(42)
        model_path = SCRIPT_DIR / "agent.pkl"
        self.model = QNetwork(8, 4, 42)
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)
        
    def act(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_vals = self.model(state)

        return np.argmax(action_vals.cpu().data.numpy())

