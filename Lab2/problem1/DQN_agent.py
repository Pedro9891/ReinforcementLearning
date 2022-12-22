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

        self.input_layer = nn.Linear(input_size, 32)
        self.input_layer_activation = nn.ReLU()

        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):

        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)
        out = self.output_layer(l1)

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
        self.mNetwork = MyNetwork(8, n_actions) #number of states  = 8
        self.tNetwork = MyNetwork(8, n_actions)
        self.tNetwork.load_state_dict(self.mNetwork.state_dict())

        self.optimizer = optim.Adam(self.mNetwork.parameters(), lr= 0.0001)

    def forward(self, state: np.ndarray, eps_k):
        ''' Performs a forward computation '''
        out = self.mNetwork(state)

        #take epsilon greedy action
        if np.random.uniform(0, 1) > eps_k:
            action = out.max(1)[1].item()
        else:
            action = np.random.randint(0, self.n_actions)

        return action
        #pass

    def backward(self, states, actions, target_states,N):
        ''' Performs a backward pass on the network '''
        self.optimizer.zero_grad()
        #states, actions, rewards, next_states, dones = buffer.sample_batch(3)
        Q_values = self.mNetwork(torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32).to(device))
        Q_tensor = torch.zeros(64, requires_grad=False,dtype=torch.float32).to(device)

        #Get Q(s_i, a_i)
        for i in range(N):
            Q_tensor[i] = Q_values[i,actions[i]]

        target_values = torch.tensor(target_states, requires_grad=True, dtype=torch.float32).to(device)


        loss = nn.functional.mse_loss(Q_tensor, target_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mNetwork.parameters(), 1)
        self.optimizer.step()


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
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
