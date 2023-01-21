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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class Agent(nn.Module):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.last_action = None
        self.theta_network = MyNetwork(8, n_actions) #number of states  = 8
        self.theta_prime_network = MyNetwork(8, n_actions)
        self.theta_prime_network.load_state_dict(self.theta_network.state_dict())
        self.optimizer = optim.Adam(self.theta_network.parameters(), lr= 0.0001)

    def forward(self, state: np.ndarray, eps_k=0):
        ''' Performs a forward computation '''
        out = self.theta_network(state)

        #take epsilon greedy action
        if np.random.uniform(0, 1) > eps_k:
            action = out.max(1)[1].item()
        else:
            action = np.random.randint(0, self.n_actions)

        return action
        #pass

    def backward(self, batch, discount_factor):
        ''' Performs a backward pass on the network '''
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = batch
        dones = torch.tensor(dones).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        states = torch.tensor(np.array(states)).to(device)
        next_states = torch.tensor(np.array(next_states)).to(device)
        batch_size = states.shape[0]
        Q_prime = torch.max(self.theta_prime_network(next_states), dim=1)[0]
        target = torch.where(dones, rewards, rewards+discount_factor*Q_prime)
        value_func = self.theta_network(states)[range(batch_size), actions]
        loss = nn.functional.mse_loss(value_func, target)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm(self.theta_network.parameters(), 2)
        self.optimizer.step()
    
    def save_theta(self, file_path):
        torch.save(self.theta_network, file_path)


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = torch.tensor(np.random.randint(0, self.n_actions))
        return torch.rand([1, self.n_actions])
        return self.last_action
