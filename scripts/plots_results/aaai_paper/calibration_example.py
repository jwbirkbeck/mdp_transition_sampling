import torch
import scipy
import numpy as np
import pandas as pd
import pickle
import sys, os, glob
import matplotlib.pyplot as plt
from src.finite_mdps.simple_grid_v2 import SimpleGridV2


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


device = torch.device('cpu')
sg_size = 20
sg_env = SimpleGridV2(size=sg_size, seed=0, device=device, render_mode='human')

# # # # #
# SimpleGrid example
# # # # #
with open('../../local_experiments/simplegrid_w1_vs_returns.pkl', 'rb') as file:
    results_dict = pickle.load(file)
    sg_w1 = results_dict['dists']
    sg_eval = results_dict['evals']

sg_min_rs = []
sg_max_rs = []
for seed in range(5000):
    print(seed)
    sg_env.seed = seed
    min_r, max_r = get_max_min_return(env=sg_env)
    sg_min_rs.append(min_r)
    sg_max_rs.append(max_r)

sg_plotdata = pd.DataFrame({'w1': sg_w1, 'evals': sg_eval, 'min_r': sg_min_rs, 'max_r': sg_max_rs})
sg_plotdata.loc[sg_plotdata.evals < sg_plotdata.min_r, 'min_r'] = sg_plotdata.evals[sg_plotdata.evals < sg_plotdata.min_r]
sg_plotdata.loc[sg_plotdata.evals > sg_plotdata.max_r, 'max_r'] = sg_plotdata.evals[sg_plotdata.evals > sg_plotdata.max_r]
sg_plotdata['sopr'] = (sg_plotdata.max_r - sg_plotdata.evals) / (sg_plotdata.max_r - sg_plotdata.min_r)


def fit_spline(x, y, bins, k=1):
    spline_x = []
    spline_y = []
    for ind in range(len(bins) - 1):
        bin_low = bins[ind]
        bin_high = bins[ind + 1]
        this_y = y[np.logical_and(x >= bin_low, x < bin_high)]
        spline_x.append(bin_low + (bin_high - bin_low) / 2)
        spline_y.append(np.median(this_y))
    t, c, k = scipy.interpolate.splrep(spline_x, spline_y, k=k)
    spline = scipy.interpolate.BSpline(t, c, k)
    return spline

sg_spline = fit_spline(x=sg_plotdata.w1, y=sg_plotdata.sopr, bins=list(range(13)), k=1)

sg_spline_xs = np.linspace(0, 12, 100)
sg_spline_ys = sg_spline(sg_spline_xs)
sg_plotdata['adj_dists'] = sg_spline(sg_plotdata.w1)

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = sg_plotdata[np.logical_and(sg_plotdata.w1 > bin_low, sg_plotdata.w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=0.125)
# plt.xlabel("W1 distance from base MDP")
# plt.ylabel("SOPR (lower is better)")
# plt.title("SimpleGrid: SOPR vs W1 MDP distance")
plt.xticks(ticks = np.arange(0, 12.5, 1))
plt.tight_layout()
plt.plot(sg_spline_xs, sg_spline_ys, color='black')
plt.show()
bins = np.arange(0, 1.05, 0.1)
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = sg_plotdata[np.logical_and(sg_plotdata.adj_dists > bin_low, sg_plotdata.adj_dists <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.05, bw_method=0.125)
plt.xlabel("Calibrated W1 distance from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SimpleGrid: SOPR vs Calibrated W1 distance")
plt.xticks(ticks = np.arange(0, 1.05, 0.1))
plt.tight_layout()
plt.show()

# # # # #
# MetaWorld example
# # # # #

mw_w1_dists = pd.read_csv('/opt/project/scripts/plots_results/cont_nonstat_w1_vs_returns/w1_dists_new.csv')

results_dir = "/opt/project/results/cont_nonstat_w1_vs_returns/"
all_filenames = glob.glob(os.path.join(results_dir, "eval_[0-9]*.csv"))
mw_agent_results = pd.DataFrame()
tmp_ind = 0
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = tmp_ind
    tmp_ind += 1
    mw_agent_results = pd.concat((mw_agent_results, tmp))

ret_max = mw_agent_results.rename(columns={'ep_reward': 'ret_max'}).groupby(['task', 'test_ind']).ret_max.max()
ret_min = mw_agent_results.rename(columns={'ep_reward': 'ret_min'}).groupby(['task', 'test_ind']).ret_min.min()

mw_agent_results = mw_agent_results.merge(ret_max, how='left', on=['task', 'test_ind'])
mw_agent_results = mw_agent_results.merge(ret_min, how='left', on=['task', 'test_ind'])
mw_agent_results['sopr'] = (mw_agent_results.ret_max - mw_agent_results.ep_reward) / (mw_agent_results.ret_max - mw_agent_results.ret_min)

mw_median_w1_dists = mw_w1_dists.drop('rep', axis=1).groupby(['task', 'test_ind']).median()
mw_agent_results = mw_agent_results.merge(mw_median_w1_dists, how='inner', on=['task', 'test_ind'])

mw_agent_results = mw_agent_results[mw_agent_results.w1 < 1.2]
_, bins = pd.qcut(mw_agent_results.w1, q=4, retbins=True)
mw_spline = fit_spline(x=mw_agent_results.w1, y=mw_agent_results.sopr, bins=bins, k=1)

mw_spline_xs = np.linspace(0, max(mw_agent_results.w1), 100)
mw_spline_ys = mw_spline(mw_spline_xs)
mw_agent_results['adj_w1'] = mw_spline(mw_agent_results.w1)

bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = mw_agent_results[np.logical_and(mw_agent_results.w1 > bin_low, mw_agent_results.w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.15, bw_method=2e-2)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.xlabel("W1 distance from from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SOPR against Wasserstein MDP distance")
plt.tight_layout()
plt.plot(mw_spline_xs, mw_spline_ys, color='black')
plt.show()

_, bins = pd.qcut(mw_agent_results.adj_w1, q=8, retbins=True)
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = mw_agent_results[np.logical_and(mw_agent_results.adj_w1 > bin_low, mw_agent_results.adj_w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.05, bw_method=0.125)
plt.xlabel("W1 distance from from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SOPR against Wasserstein MDP distance")
plt.xticks(ticks = np.arange(0, 1.05, 0.1))
plt.tight_layout()
plt.show()

plt.plot(sg_spline_xs, sg_spline_ys, color='black')
plt.plot(mw_spline_xs, mw_spline_ys, color='blue')
plt.show()

plt.scatter(sg_plotdata.adj_dists, sg_plotdata.sopr, color='black', s=3, alpha = 0.1)
plt.scatter(mw_agent_results.adj_w1, mw_agent_results.sopr, color='blue', s=3, alpha = 0.1)
plt.show()