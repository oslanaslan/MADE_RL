
import os
import copy
import random
import warnings
from collections import deque

import pybullet_envs
from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm


GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 3e-4
ACTOR_LR = 2e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cuda'
BATCH_SIZE = 256
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1000000
NOISE_STD = 0.5
SEED = 42
FILE_NAME = "agent.pt"


def set_seed(seed):
  """ Set seed """
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ["PYTHONHASHSEED"] = str(seed)


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        
        self.replay_buffer = deque(maxlen=200000)
        self.step = 0

    def update(self, transition):
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)] for _ in range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)
            
            # Update critic
            with torch.no_grad():
                noise = torch.normal(
                    mean=torch.zeros_like(action),
                    std=NOISE_STD*torch.ones_like(action)
                )
                next_action = (self.target_actor(next_state) + noise).clamp(-1, 1)

                q_critic_1 = self.target_critic_1(next_state, next_action)
                q_critic_2 = self.target_critic_2(next_state, next_action)
                q_cur = reward + (1 - done) * GAMMA * torch.min(q_critic_1, q_critic_2)

            q_critic_1 = self.critic_1(state, action)
            q_critic_2 = self.critic_2(state, action)


            # Update critic
            loss = F.mse_loss(q_critic_1, q_cur) + F.mse_loss(q_critic_2, q_cur)
            self.critic_1_optim.zero_grad()
            self.critic_2_optim.zero_grad()
            loss.backward()
            self.critic_1_optim.step()
            self.critic_2_optim.step()

            # Update actor
            # if self.step % 2 == 0:
            loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            
            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

            self.step += 1

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self):
        torch.save(self.actor.state_dict(), FILE_NAME)


def evaluate_policy(env, agent, episodes=5):
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

if __name__ == "__main__":
    set_seed(SEED)
    warnings.filterwarnings('ignore')
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    td3 = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    eps = 0.2

    pbar = tqdm(range(TRANSITIONS))
    mean_rewards = 0
    std_rewards = 0
    best_mean_rewards = 0
    print(f"Using device: {DEVICE}")
    
    for i in pbar:
        steps = 0
        
        #Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + eps * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(test_env, td3, 5)
            mean_rewards = np.mean(rewards)
            std_rewards = np.std(rewards)
            
            if mean_rewards > best_mean_rewards:
                best_mean_rewards = mean_rewards
                td3.save()

        pbar.set_postfix_str(f"Reward mean: {mean_rewards}, Reward std: {std_rewards}, Best: {best_mean_rewards}")
