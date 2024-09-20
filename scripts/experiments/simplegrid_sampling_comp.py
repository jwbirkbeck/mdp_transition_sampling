import os, sys
import torch
import numpy as np
import pandas as pd
import ot
from src.grid_worlds.simple_grid_v2 import SimpleGridV2
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.filepaths import results_path_iridis, results_path_local


results_dir = os.path.join(results_path_iridis, 'simplegrid_sampling_comp/')

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device('cpu')
size = 20
base_env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')
comp_env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

state_space = base_env.observation_space
action_space = base_env.action_space

mcce_sampler = MDPDifferenceSampler(env_a=base_env, env_b=comp_env, state_space=state_space, action_space=action_space,
                                    method='mcce')
rand_sampler = MDPDifferenceSampler(env_a=base_env, env_b=comp_env, state_space=state_space, action_space=action_space,
                                    method='random')

def get_true_distance(env_a, env_b):
    next_states_a, rewards_a = env_a.get_all_transitions(render=False)
    next_states_b, rewards_b = env_b.get_all_transitions(render=False)
    samples_a = torch.cat((next_states_a, rewards_a), dim=1)
    samples_b = torch.cat((next_states_b, rewards_b), dim=1)
    a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
    b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
    M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
    return ot.emd2(a=a, b=b, M=M)

results = pd.DataFrame()
for seed in range(5000):
    comp_env.seed = seed

    base_env.reset()
    comp_env.reset()

    # full distance
    true_dist = get_true_distance(base_env, comp_env)

    pd_row = pd.DataFrame({'seed': [seed], 'dist': [true_dist], 'type': ['true'], 'n_states': [0]})
    results = pd.concat((results, pd_row))
    # get sampled dists
    for n_states in [1, 5, 10, 15]:
        mcce_dist = mcce_sampler.get_difference(n_states=n_states, n_transitions=1)
        pd_row = pd.DataFrame({'seed': [seed], 'dist': [mcce_dist], 'type': ['mcce'], 'n_states': [n_states]})
        results = pd.concat((results, pd_row))

        rand_dist = rand_sampler.get_difference(n_states=n_states, n_transitions=1)
        pd_row = pd.DataFrame({'seed': [seed], 'dist': [rand_dist], 'type': ['rand'], 'n_states': [n_states]})
        results = pd.concat((results, pd_row))

    results.to_csv(os.path.join(results_dir, f'results_{run_index}.csv'), index=False)



