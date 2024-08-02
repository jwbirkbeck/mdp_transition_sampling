import scipy.stats
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from src.finite_mdps.simple_grid_v2 import SimpleGridV2
from src.utils.filepaths import project_path_local

device = torch.device('cpu')
size = 20
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'simplegrid_reward_shaped_sampling.pkl'), 'rb') as file:
    results_dict = pickle.load(file)
    true_rs = np.array(results_dict['true'])
    rs = np.array(results_dict['sampled'])
with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'simplegrid_random_sampling.pkl'), 'rb') as file:
    results_dict = pickle.load(file)
    true_r = np.array(results_dict['true'])
    random = np.array(results_dict['sampled'])
with open('../../local_experiments/simplegrid_w1_vs_returns.pkl', 'rb') as file:
    results_dict = pickle.load(file)
    eval_dists = results_dict['dists']
    evals = results_dict['evals']



def get_optimal_action(env, minimize=False):
    agent_pos = env._agent_pos
    goal_pos = env._goal_pos
    opt_actions = []
    if not minimize:
        if agent_pos[0] < goal_pos[0]:
            opt_actions.append(1)
        if agent_pos[0] > goal_pos[0]:
            opt_actions.append(3)
        if agent_pos[1] < goal_pos[1]:
            opt_actions.append(2)
        if agent_pos[1] > goal_pos[1]:
            opt_actions.append(0)
        if torch.all(agent_pos == goal_pos):
            # check if we're near an edge. if we are, move into that edge:
            if agent_pos[0] == 1:
                opt_actions.append(3)
            if agent_pos[0] == env.size-2:
                opt_actions.append(1)
            if agent_pos[1] == env.size-2:
                opt_actions.append(2)
    else:
        if agent_pos[0] > goal_pos[0]:
            opt_actions.append(1)
        if agent_pos[0] < goal_pos[0]:
            opt_actions.append(3)
        if agent_pos[1] > goal_pos[1]:
            opt_actions.append(2)
        if agent_pos[1] < goal_pos[1]:
            opt_actions.append(0)
    if len(opt_actions) == 0:
        opt_actions = [0, 1, 2, 3]
    return np.random.choice(opt_actions)


def get_optimal_return(env, minimize=False):
    ep_reward = 0
    env.reset()
    truncated = terminated = False
    while not (truncated or terminated):
        observation, reward, terminated, truncated, info = env.step(get_optimal_action(env, minimize=minimize))
        ep_reward += reward.item()
        optimal_return = ep_reward
    return optimal_return


def get_max_min_return(env):
    min_return = get_optimal_return(env=env, minimize=True)
    max_return = get_optimal_return(env=env, minimize=False)
    return min_return, max_return


min_rs = []
max_rs = []
for seed in range(5000):
    print(seed)
    env.seed = seed
    min_r, max_r = get_max_min_return(env=env)
    min_rs.append(min_r)
    max_rs.append(max_r)

soprs = [(max_r - r) / (max_r - min_r) for r, max_r, min_r in zip(evals, max_rs, min_rs)]
# plotdata = pd.DataFrame({'true_rs': true_rs, 'rs': rs, 'true_r': true_r, 'r': random, 'evals': evals, 'sopr': soprs})
# plotdata.min_r[plotdata.evals < plotdata.min_r] = plotdata.evals[plotdata.evals < plotdata.min_r]
# plotdata.max_r[plotdata.evals > plotdata.max_r] = plotdata.evals[plotdata.evals > plotdata.max_r]
# plotdata.sopr = (plotdata.max_r - plotdata.evals) / (plotdata.max_r - plotdata.min_r)


import scipy
scipy.stats.linregress([rs, soprs])
scipy.stats.linregress([random, soprs])

scipy.stats.pearsonr(rs, soprs)
scipy.stats.pearsonr(random, soprs)

scipy.stats.pearsonr(rs, random)
