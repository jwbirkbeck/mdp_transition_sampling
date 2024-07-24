import scipy
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot
from src.finite_mdps.simple_grid_v2 import SimpleGridV2
from src.sampler.samplers import MDPDifferenceSampler
import pickle

def get_true_distance(env_a, env_b):
    next_states_a, rewards_a = env_a.get_all_transitions(render=False)
    next_states_b, rewards_b = env_b.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    return ot.emd2(a=a, b=b, M=M)


device = torch.device('cpu')
size = 20
env_a = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')
env_b = SimpleGridV2(size=size, seed=2, device=device, render_mode='human')

state_space = env_a.observation_space
action_space = env_a.action_space

sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                               state_space=state_space, action_space=action_space, method='mcce')

true_distance = []
sampled_distance = []
for seed in range(5000):
    print(seed)
    env_a = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')
    env_b = SimpleGridV2(size=size, seed=seed, device=device, render_mode='human')

    state_space = env_a.observation_space
    action_space = env_a.action_space

    sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                   state_space=state_space, action_space=action_space, method='mcce')

    true_distance.append(get_true_distance(env_a, env_b))

    env_a.reset()
    env_b.reset()
    sampled_distance.append(sampler.get_difference(n_states=15, n_transitions=1))
print("done")

dists = {'true': true_distance, 'sampled': sampled_distance}


with open('simplegrid_dists_from_samples.pkl', 'wb') as file:
    pickle.dump(dists, file)
