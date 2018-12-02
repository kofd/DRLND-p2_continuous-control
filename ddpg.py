from collections import namedtuple, deque
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Normal(dist.Normal):
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.normal(self.loc.expand(shape), self.scale.expand(shape))


class Actor(nn.Module):
    def __init__(self, feature_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, action_size)
        self.fc_std = nn.Linear(128, action_size)

    def forward(self, state):
        X = F.elu(self.fc1(state))
        X = F.elu(self.fc2(X))
        A = Normal(self.fc_mean(X), torch.exp(self.fc_std(X)))
        return A


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.fc_mean = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)

    def forward(self, state, action):
        X = F.elu(self.fc1(state))
        X = F.elu(self.fc2(torch.cat([X, action], -1)))
        V = Normal(self.fc_mean(X), torch.exp(self.fc_std(X)))
        return V


class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, tau=0.001,
                 buffer_size=10000, batch_size=64):
        self.tau = tau
        self.local_actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.local_critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size

        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(local_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(local_param.data)

    def act(self, state, det=True):
        state = torch.from_numpy(state).float().to(device)

        self.local_actor.eval()
        with torch.no_grad():
            action_dist = self.local_actor.forward(state)
            action_values = action_dist.mean if det else action_dist.sample()
            action_values = torch.tanh(action_values).cpu().data.numpy()
        self.local_actor.train()

        return action_values

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self, gamma, det=True):
        if len(self.replay_buffer) > self.batch_size:
            experiences = self.replay_buffer.sample()
            self.update_local(experiences, gamma, det)
            self.update_target()

    def update_local(self, experiences, gamma, det=True):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions = torch.tanh(self.target_actor.forward(states).mean).detach()
            next_values = self.target_critic.forward(states, next_actions).mean.detach()
            q_target = 0.1 * rewards + ((1. - dones) * gamma * next_values)

        q_local = self.local_critic.forward(states, actions)
        q_local = q_local.mean if det else q_local.sample()
        critic_loss = F.mse_loss(q_local, q_target)#-q_local.log_prob(q_target).mean()

        self.local_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.critic_optimizer.step()

        local_actions = self.local_actor.forward(states)
        local_actions = local_actions.mean if det else local_actions.sample()
        policy_loss = -self.local_critic.forward(states, torch.tanh(local_actions)).mean.mean()

        self.local_actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), 1)
        self.actor_optimizer.step()

    def update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.memory = deque(maxlen=int(buffer_size))
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        for state, action, reward, next_state, done in \
                zip(state, action, reward, next_state, done):
            self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
