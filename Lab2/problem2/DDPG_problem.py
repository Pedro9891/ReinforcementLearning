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
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, Agent
from collections import deque
import numpy as np
# torch.manual_seed(32)
# np.random.seed(32)
torch.manual_seed(21)
np.random.seed(21)
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float,
                    help='gamma value',
                    default=0.99)
parser.add_argument('--buffer', type=int, 
                    help='buffer length', 
                    default=30000)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 700               # Number of episodes to run for training
discount_factor = args.gamma        # Value of gamma
n_ep_running_average = 50      # Running average of 50 episodes
m = len(env.action_space.high) # dimensionality of the action
buffer_length = args.buffer
buffer =  ExperienceReplayBuffer(buffer_length)
N = 64
d = 2
sigma=0.2
mu=0.15
save_dir = os.path.join(f'./saves2/gamma_{args.gamma}_buffer_{args.buffer}_eps_{N_episodes}_big_net')
if not os.path.exists(save_dir): os.makedirs(save_dir)
plot_every=10
# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
agent = Agent(m, d)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    n = 0
    # if i < 400:
    #     sigma=0.25
    # else:
    #     sigma=0.2
    while not done:
        # Take a random action
        state_tensor = torch.tensor(np.array([state]),
                                    requires_grad=False,
                                    dtype=torch.float32).to(device)
        action = agent.forward(state_tensor).squeeze()
        w = np.random.normal(0, sigma, action.shape)
        n = -mu*n + w
        action += n
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        exp = (state, action, reward, next_state, done)
        buffer.append(exp)
        # Update episode reward
        total_episode_reward += reward
        # Update state for next iteration
        state = next_state
        t+= 1
        if len(buffer) > 500:
            agent.backward(buffer.sample_batch(N), discount_factor, t)

        




    # Append episode reward
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
    if  i > 50 and i % plot_every==0:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax[0].plot([j for j in range(1, i+2)], episode_reward_list, label='Episode reward')
        ax[0].plot([j for j in range(1, i+2)], running_average(
            episode_reward_list, n_ep_running_average), label='Avg. episode reward')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Total reward')
        ax[0].set_title('Total Reward vs Episodes')
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot([j for j in range(1, i+2)], episode_number_of_steps, label='Steps per episode')
        ax[1].plot([j for j in range(1, i+2)], running_average(
            episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Total number of steps')
        ax[1].set_title('Total number of steps vs Episodes')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'fig.png'))
        plt.close()
        agent.save(save_dir)


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
plt.savefig(os.path.join(save_dir, 'fig.png'))
plt.close()
agent.save(save_dir)
