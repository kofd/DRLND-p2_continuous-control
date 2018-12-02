import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_A1 = nn.Linear(128, 128)
        self.fc_A2 = nn.Linear(128, action_size)
        self.fc_A3 = nn.Linear(128, action_size)

    def forward(self, state):
        X = F.relu(self.fc1(state))
        X = F.relu(self.fc2(X))
        A = F.relu(self.fc_A1(X))
        return Normal(self.fc_A2(A), F.elu(self.fc_A3(A)) + 1)


def collect_trajectories(envs, name, policy, tmax=200):
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    envs.reset()

    env_info = envs.reset(train_mode=True)[name]

    for t in range(tmax):
        state = env_info.vector_observations
        dist = policy(torch.tensor(state, dtype=torch.float, device=device))

        action = torch.tanh(dist.sample())
        probs = np.exp(dist.log_prob(action).cpu().detach().numpy())
        action = action.cpu().detach().numpy()
        step = envs.step(action)[name]
        reward, done = step.rewards, step.local_done

        # store the result
        state_list.append(state)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        if any(done):
            break

    return prob_list, state_list, action_list, reward_list


def surrogate(policy, old_probs, states, actions, rewards, discount=0.995,
              epsilon=0.1, beta=0.01):
    rewards = np.array(rewards) * np.power(discount, np.arange(len(rewards))).reshape(len(rewards),1)
    rewards = rewards[::-1].cumsum(axis=0)[::-1]
    rewards = (rewards - np.mean(rewards, axis=1, keepdims=True)) / \
              (np.std(rewards, axis=1, keepdims=True) + 1.e-10)

    actions = torch.tensor(actions, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(-1)
    states = torch.tensor(np.array(states), dtype=torch.float, device=device)

    dist = policy(states)
    new_probs = torch.exp(dist.log_prob(actions))

    # ratio for clipping
    ratio = new_probs / old_probs
    clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(clipped_surrogate + beta * entropy)
