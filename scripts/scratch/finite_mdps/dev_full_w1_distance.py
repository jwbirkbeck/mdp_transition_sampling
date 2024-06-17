import torch
import matplotlib.pyplot as plt
from src.finite_mdps.simple_grid_v1 import SimpleGridV1
import numpy as np
import ot

device = torch.device('cpu')
env_a = SimpleGridV1(height=12, width=12, seed=0, device=device, render_mode='human')
env_b = SimpleGridV1(height=12, width=12, seed=0, device=device, render_mode='human')

next_states_a, rewards_a = env_a.get_all_transitions(render=False)
next_states_b, rewards_b = env_b.get_all_transitions(render=False)


samples_a = torch.cat((next_states_a, rewards_a), dim=1)
samples_b = torch.cat((next_states_b, rewards_b), dim=1)
a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
ot.emd2(a=a, b=b, M=M)

env_a.render()

next_states_a, rewards_a = env_a.get_all_transitions(render=True)
env_a.close()

"""
For a given seed:
    For every possible state space:
        For every possible action:
            Set the MDP by the state and action
            Execute the MDP
            Record the transition

Calculation of tensor size produced over two MDPs:
    n_states * n_actions * 2
    gridsize**2 * 4 * 2
"""
