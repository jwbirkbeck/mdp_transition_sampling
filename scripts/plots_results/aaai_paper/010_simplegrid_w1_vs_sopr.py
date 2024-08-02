import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.finite_mdps.simple_grid_v2 import SimpleGridV2
import pickle

device = torch.device('cpu')
size = 20
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

with open('../../local_experiments/simplegrid_w1_vs_returns.pkl', 'rb') as file:
    results_dict = pickle.load(file)
    dists = results_dict['dists']
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
plotdata = pd.DataFrame({'dists': dists, 'evals': evals, 'sopr': soprs, 'min_r': min_rs, 'max_r': max_rs})
plotdata.loc[plotdata.evals < plotdata.min_r, 'min_r'] = plotdata.loc[plotdata.evals < plotdata.min_r, 'evals']
plotdata.loc[plotdata.evals > plotdata.max_r, 'max_r'] = plotdata.loc[plotdata.evals > plotdata.max_r, 'evals']
plotdata.sopr = (plotdata.max_r - plotdata.evals) / (plotdata.max_r - plotdata.min_r)

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.dists > bin_low, plotdata.dists <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=0.125)
plt.xlabel("W1 distance from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SimpleGrid: SOPR vs W1 MDP distance")
plt.xticks(ticks = np.arange(0, 12.5, 1))
plt.tight_layout()
plt.savefig("010_simplegrid_w1_vs_sopr.png", dpi=300)
plt.show()
