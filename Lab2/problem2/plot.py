import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import cm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
actor = torch.load('/home/ali/reinforce/ReinforcementLearning/Lab2/problem2/saves2/gamma_0.98_buffer_30000_eps_800_big_net/neural-network-2-actor.pth')
critic = torch.load('/home/ali/reinforce/ReinforcementLearning/Lab2/problem2/saves2/gamma_0.98_buffer_30000_eps_800_big_net/neural-network-2-critic.pth').to(device)
omega = np.linspace(-np.pi, np.pi, 110)
y = np.linspace(0, 1.5, 50)
omega, y = np.meshgrid(omega, y)
assert omega.shape  == y.shape
old_shape = omega.shape
omega = torch.tensor(omega).reshape((-1))
y =  torch.tensor(y).reshape((-1))
state = torch.zeros((y.shape[0], 8))
state[:, 1] = y
state[:,4] = omega

action = actor(state.to(device))
q = critic(state.to(device), action).squeeze()

print(action.shape, q.shape)



omega = omega.detach().cpu().reshape(old_shape).numpy()
y = y.detach().cpu().reshape(old_shape).numpy()
q = q.detach().cpu().reshape(old_shape).numpy()
action = action.detach().cpu().reshape((old_shape[0], old_shape[1], -1)).numpy()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
surf = ax.plot_surface(omega, y, q, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('omega')
ax.set_ylabel('y')
ax.set_zlabel('Q')

plt.savefig('p2_q.png')

plt.close()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
surf = ax.plot_surface(omega, y, action[:,:,1], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('omega')
ax.set_ylabel('y')
ax.set_zlabel('action_2')

plt.savefig('p2_ac.png')
plt.close()






