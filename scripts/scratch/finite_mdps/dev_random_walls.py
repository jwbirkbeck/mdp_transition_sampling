import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
from src.grid_worlds.simple_grid_v1 import SimpleGridV1

device = torch.device('cpu')
width = 16
height = 16

dists = []
for seed in range(100):
    print(seed)
    env_a = SimpleGridV1(height=height, width=width, seed=0, device=device, render_mode='human')
    env_b = SimpleGridV1(height=height, width=width, seed=seed, device=device, render_mode='human')

    next_states_a, rewards_a = env_a.get_all_transitions(render=False)
    next_states_b, rewards_b = env_b.get_all_transitions(render=False)

    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    dists.append(ot.emd2(a=a, b=b, M=M))

plt.hist(dists)
plt.show()

env_a.render()
env_b.render()

env_a._add_random_walls(n_walls=1)

next_states_a, rewards_a = env_a.get_all_transitions(render=False)
next_states_b, rewards_b = env_b.get_all_transitions(render=False)

samples_a = torch.cat((next_states_a, rewards_a), dim=1)
samples_b = torch.cat((next_states_b, rewards_b), dim=1)
a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
print(ot.emd2(a=a, b=b, M=M))

env_a.close()
