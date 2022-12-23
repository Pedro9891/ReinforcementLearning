import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import cm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('/home/ali/reinforce/ReinforcementLearning/Lab2/problem1/saves/model_ep_1000_gam_0.98_buffer_20000_batch_128_step/neural-network-1.pth')

omega = np.linspace(-np.pi, np.pi, 120)
y = np.linspace(0, 1.5, 80)
omega, y = np.meshgrid(omega, y)
assert omega.shape  == y.shape
old_shape = omega.shape
omega = torch.tensor(omega).reshape((-1))
y =  torch.tensor(y).reshape((-1))
state = torch.zeros((y.shape[0], 8))
state[:, 1] = y
state[:,4] = omega
q = model(state.to(device))

q, q_arg_max = torch.max(q, dim=-1)

omega = omega.detach().cpu().reshape(old_shape).numpy()
y = y.detach().cpu().reshape(old_shape).numpy()
q = q.detach().cpu().reshape(old_shape).numpy()
q_arg_max = q_arg_max.detach().cpu().reshape(old_shape).numpy()
print(omega.shape, y.shape, q.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
surf = ax.plot_surface(omega, y, q, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('omega')
ax.set_ylabel('y')
ax.set_zlabel('Q')

plt.savefig('fig_f.png')
plt.close()
omega = omega.reshape((-1))
y = y.reshape((-1))
q_arg_max = q_arg_max.reshape((-1))
plt.scatter(omega[q_arg_max==0], y[q_arg_max==0], c = 'red', label='action 1', s=1)
plt.scatter(omega[q_arg_max==1], y[q_arg_max==1], c = 'blue', label='action 2', s=1)
plt.scatter(omega[q_arg_max==2], y[q_arg_max==2], c = 'yellow', label='action 3', s=1)
plt.scatter(omega[q_arg_max==3], y[q_arg_max==3], c = 'green', label='action 4', s=1)
plt.legend()
plt.savefig('scatter.png')
plt.close()
print(np.unique(q_arg_max))





