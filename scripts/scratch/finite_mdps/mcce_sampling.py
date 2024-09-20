import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.grid_worlds.simple_grid_v2 import SimpleGridV2
from src.sampler.samplers import MDPDifferenceSampler

device = torch.device('cpu')
size = 20
env_a = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')
env_b = SimpleGridV2(size=size, seed=2, device=device, render_mode='human')

state_space = env_a.observation_space
action_space = env_a.action_space

sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                               state_space=state_space, action_space=action_space, method='mcce')

sampler.get_difference(n_states=15, n_transitions=5)

next_states_a, rewards_a = env_a.get_all_transitions(render=False)
next_states_b, rewards_b = env_b.get_all_transitions(render=False)
samples_a = torch.cat((next_states_a, rewards_a), dim=1)
samples_b = torch.cat((next_states_b, rewards_b), dim=1)
a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
print(ot.emd2(a=a, b=b, M=M))

next_states_a, rewards_a = env_a.get_all_transitions(render=False)
next_states_b, rewards_b = env_b.get_all_transitions(render=False)
samples_a = torch.cat((next_states_a, rewards_a), dim=1)
samples_b = torch.cat((next_states_b, rewards_b), dim=1)
a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
print(ot.emd2(a=a, b=b, M=M))
