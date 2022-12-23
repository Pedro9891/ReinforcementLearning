# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import os
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from DDPG_soft_updates import soft_updates
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class CriticNet(nn.Module):
    def __init__(self, state_dim, n_actions) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(state_dim, 400), nn.ReLU())
        self.model = nn.Sequential(nn.Linear(400+n_actions, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 1))
    def forward(self, state, action):
        encode = self.input_layer(state)
        encode = torch.cat([encode, action], dim=1)
        return self.model(encode)


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int, d):
        self.n_actions = n_actions
        self.state_dim = 8
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, self.n_actions),
            nn.Tanh()
        ).to(device)
        self.t_actor = nn.Sequential(
            nn.Linear(self.state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, self.n_actions),
            nn.Tanh()
        ).to(device)
        self.t_actor.load_state_dict(self.actor.state_dict())
        self.critic = CriticNet(self.state_dim, self.n_actions).to(device)
        self.t_critic = CriticNet(self.state_dim, self.n_actions).to(device)
        self.t_critic.load_state_dict(self.critic.state_dict())
        self.d = d
        self.optim_actor = optim.Adam(self.actor.parameters(), lr= 5e-5)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr= 5e-5)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        return self.actor(state).detach().cpu().numpy()
        pass

    def backward(self, batch, discount_factor, t):
        ''' Performs a backward pass on the network '''
        states, actions, rewards, next_states, dones = batch
        self.optim_critic.zero_grad()
        self.optim_actor.zero_grad()
        dones = torch.tensor(dones).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        states = torch.tensor(np.array(states)).to(device)
        next_states = torch.tensor(np.array(next_states)).to(device)
        batch_size = states.shape[0]
        y =  torch.where(dones, rewards, rewards+discount_factor*self.t_critic.forward(next_states, self.t_actor(next_states)).squeeze())
        loss = nn.functional.mse_loss(self.critic(states,actions).squeeze(), y)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm(self.critic.parameters(), 1)
        self.optim_critic.step()
        if t % self.d == 0:
            loss = - self.critic(states, self.actor(states)).squeeze().mean()
            nn.utils.clip_grad.clip_grad_norm(self.actor.parameters(), 1)
            loss.backward()
            self.optim_actor.step()
            self.t_actor = soft_updates(self.actor, self.t_actor)
            self.t_critic = soft_updates(self.critic, self.t_critic)

    def save(self, save_dir):
        torch.save(self.actor, os.path.join(save_dir, 'neural-network-2-actor.pth'))
        torch.save(self.critic, os.path.join(save_dir, 'neural-network-2-critic.pth'))

        




class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
