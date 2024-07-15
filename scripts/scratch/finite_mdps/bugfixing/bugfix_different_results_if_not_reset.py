import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.finite_mdps.simple_grid_v1 import SimpleGridV1

width = 20
height = 20
device=torch.device('cpu')

# env_1 = SimpleGridV1(height=width, width=height, seed=0, device=device, render_mode='human')
# for _ in range(3):
#     env_2 = SimpleGridV1(height=width, width=height, seed=3, device=device, render_mode='human')
#     next_states_a, rewards_a = env_1.get_all_transitions(render=False)
#     next_states_b, rewards_b = env_2.get_all_transitions(render=False)
#     samples_a = torch.cat((next_states_a, rewards_a), dim=1)
#     samples_b = torch.cat((next_states_b, rewards_b), dim=1)
#     a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
#     b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
#     M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
#     print(ot.emd2(a=a, b=b, M=M))

env_1 = SimpleGridV1(height=width, width=height, seed=0, device=device, render_mode='human')
env_2 = SimpleGridV1(height=width, width=height, seed=3, device=device, render_mode='human')
next_states_a, rewards_a = env_1.get_all_transitions(render=False)
next_states_b, rewards_b = env_2.get_all_transitions(render=False)
samples_a = torch.cat((next_states_a, rewards_a), dim=1)
samples_b = torch.cat((next_states_b, rewards_b), dim=1)
a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
print(ot.emd2(a=a, b=b, M=M))

env_1 = SimpleGridV1(height=width, width=height, seed=0, device=device, render_mode='human')
next_states_fresh, rewards_fresh = env_1.get_all_transitions(render=False)
next_states_stale, rewards_stale = env_1.get_all_transitions(render=False)

torch.sum(next_states_fresh - next_states_stale, dim=1)
next_states_fresh[0].reshape(12, 12)
next_states_stale[0].reshape(12, 12)

env_1.render()
env_2.render()

#
# env.seed = 0
# env.reset()
# env.render()
# agent.train_agent(train=False, render=True)
# env.get_all_transitions(render=True)
#
# torch.sum(samples_a - samples_b, dim=0)
# samples_a[0, :-1].reshape(20, 20)
#
# env.seed = 0
# state_1, _ = env.reset()
# env.render()
# env.seed = 2
# state_2, _ = env.reset()
# env.render()
# env.seed = 3
# state_2, _ = env.reset()
# env.render()
#
# torch.sum(state_1 - state_2, dim=1)
#
# state_1[2,:]
# state_2[2, :]
