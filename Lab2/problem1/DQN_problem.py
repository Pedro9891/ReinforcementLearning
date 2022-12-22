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
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from DQN_agent import Agent
from collections import deque

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=1000):
        self.buffer = deque(maxlen = maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            print("Buffer too small")
        indices = np.random.choice(len(self.buffer), n, replace=False)

        batch = [self.buffer[i] for i in indices]

        return zip(*batch)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Parameters
N_episodes = 300                             # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = Agent(n_actions).to(device)

# Create buffer
buffer_size = 20000
buffer = ExperienceReplayBuffer(maximum_length = buffer_size)


N = 64                  #Training batch
C = int(buffer_size/N)  #Set target network to main network every C iterations.
gamma = 0.95            #Discount factor
eps_max = 0.99          #epsilon
eps_min = 0.05
### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for nr,i in enumerate(EPISODES):
    # Reset enviroment data and initialize variables
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        eps_k = max(eps_min, eps_max - (eps_max - eps_min)*(nr - 1)/(N_episodes*0.95 - 1))
        
        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor(np.array([state]),
                                    requires_grad=False,
                                    dtype=torch.float32).to(device)

        # Take a random action
        action = agent.forward(state_tensor, eps_k)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        # print(len(env.step(action)))
        next_state, reward, done, _, _ = env.step(action)



        #Store experience tuple in buffer
        exp = (state, action, reward, next_state, done)
        buffer.append(exp)

        if len(buffer) > 10000:
            #Sample a batch from the experience buffer
            states, actions, rewards, next_states, dones = buffer.sample_batch(N)

            #Compute target values
            y = np.zeros(len(states))
            for j in range(len(states)):
                if not dones[j]:
                    next_state_tensor = torch.tensor(np.array([next_states[j]]),
                                                requires_grad=False,
                                                dtype=torch.float32).to(device)
                    out = agent.tNetwork(next_state_tensor)
                    max_aQ = torch.max(out)
                    y[j] = rewards[j] + gamma * max_aQ
                else:
                    y[j] = rewards[j]

            #perform backwards propagation
            agent.backward(states, actions, y, N)

            # Update episode reward
            total_episode_reward += reward

        #set target network to main network
        if t % C == 0:
            agent.tNetwork.load_state_dict(agent.mNetwork.state_dict())

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
