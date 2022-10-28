import os
import random
import copy
from collections import deque, namedtuple
from typing import Tuple

from gym import make
import numpy as np
import torch
from torch import device, maximum, nn
from torch.nn import functional as F
from torch.optim import Adam

from model import DQNModel


SEED = 42
GAMMA = 0.99
INITIAL_STEPS = 2048
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 512
BUFFER_SIZE = 2048
LEARNING_RATE = 5e-4
TAU = 1e-3


class MemoryBuffer:
    """ Memory buffer for storing experience """

    def __init__(self, action_size: int, buffer_size: int,  batch_size: int) -> None:
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
          "Experience",
          field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, state: int, action: int, next_state: int, reward: float, done: bool) -> None:
        """ Add a new experience """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self) -> Tuple[int]:
        """ Sample random batch of experience """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack(
          [e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
          [e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
          [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
          [e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(
          [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
      
    def __len__(self) -> int:
        return len(self.memory)


class DQN:
    def __init__(self, state_dim: int, action_dim: int):
        self.steps = 0 # Do not change
        self.state_size = state_dim
        self.action_size = action_dim
        self.local_model = DQNModel(state_dim, action_dim)
        self.target_model = DQNModel(state_dim, action_dim)
        self.optimizer = Adam(self.local_model.parameters(), lr=LEARNING_RATE)
        self.memory = MemoryBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_model.to(self.device)
        self.target_model.to(self.device)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        self.memory.add(*transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return self.memory.sample()
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        states, actions, rewards, next_states, dones = batch
        q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + GAMMA * q_targets_next * (1 - dones)
        q_expected = self.local_model(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        pbar = zip(self.target_model.parameters(), self.local_model.parameters())

        for target_param, local_param in pbar:
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_model.eval()

        with torch.no_grad():
          action_vals = self.local_model(state)

        self.local_model.train()
        return np.argmax(action_vals.cpu().data.numpy())

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self, name: str = "agent.pkl"):
        torch.save(self.target_model.state_dict(), name)


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


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

if __name__ == "__main__":
    set_seed(SEED)
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    max_mean_rewards = -np.inf
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            cur_mean_rewards = np.mean(rewards)
            print(f"Step: {i+1}, Reward mean: {cur_mean_rewards}, Reward std: {np.std(rewards)}")
            dqn.save()

            if cur_mean_rewards > max_mean_rewards:
                max_mean_rewards = cur_mean_rewards
                dqn.save("best_iter.pkl")
