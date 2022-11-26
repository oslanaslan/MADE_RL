import os
import sys
import random

import numpy as np
import torch
from pathlib import Path

from .train import Actor


SEED = 42
FILE_NAME = "agent.pt"
# FILE_NAME = "agent.pkl"
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))


def set_seed(seed):
  """ Set seed """
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ["PYTHONHASHSEED"] = str(seed)


class Agent:
    def __init__(self):
        model_path = SCRIPT_DIR / FILE_NAME
        set_seed(SEED)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.model = Actor(28, 8)
        self.model.load_state_dict(torch.load(model_path))
        # self.model = torch.load(model_path)
        self.model.eval().to(self.device)
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float)
            return np.clip(self.model(state).cpu().numpy(), -1, +1)

    def reset(self):
        pass
