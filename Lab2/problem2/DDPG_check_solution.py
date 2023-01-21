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
from tqdm import trange
from DDPG_agent import RandomAgent
random_agent = True
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load model
try:
    model = torch.load('/home/ali/reinforce/ReinforcementLearning/Lab2/problem2/saves2/gamma_0.98_buffer_30000_eps_800_big_net/neural-network-2-actor.pth', map_location=device)
    print('Network model: {}'.format(model))
except:
    print('File neural-network-2-actor.pth not found!')
    exit(-1)

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()
if random_agent: model = RandomAgent(len(env.action_space.high))

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 125

# Reward
episode_reward_list = []  # Used to store episodes reward
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        action = model.forward(torch.tensor([state]).to(device))
        # print(action)
        # assert False
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

if avg_reward - confidence >= CONFIDENCE_PASS:
    print('Your policy passed the test!')
else:
    print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))


import matplotlib.pyplot as plt

sums = []
t =  0
for r in episode_reward_list:
    t += r
    sums.append(t)
plt.plot(sums)
plt.title(f'Sum of Episodic Reward on {"Random" if random_agent else "trained"} Agent!')
plt.xlabel('Episode')
plt.ylabel('Aggregated reward')
plt.savefig(f'aggr_random_{random_agent}.png')
plt.close()
