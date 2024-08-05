import torch
import scipy
import numpy as np
import pandas as pd
import pickle
import sys, os, glob, re
import matplotlib.pyplot as plt
from src.utils.filepaths import *
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


def fit_spline(x, y, bins, k=2, s=0.005, use_median=True):
    spline_x = []
    spline_y = []
    for ind in range(len(bins) - 1):
        bin_low = bins[ind]
        bin_high = bins[ind + 1]
        this_y = y[np.logical_and(x >= bin_low, x < bin_high)]
        spline_x.append(bin_low + (bin_high - bin_low) / 2)
        if use_median:
            spline_y.append(np.median(this_y))
        else:
            spline_y.append(np.mean(this_y))

    t, c, k = scipy.interpolate.splrep(spline_x, spline_y, k=k, s=s)
    spline = scipy.interpolate.BSpline(t, c, k)
    return spline


device = torch.device('cpu')
sg_size = 20
sg_env = SimpleGridV2(size=sg_size, seed=0, device=device, render_mode='human')

# # # # #
# SimpleGrid example
# # # # #
with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'simplegrid_w1_vs_returns.pkl'), 'rb') as file:
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

# sg_raw_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_, sg_raw_bins = pd.qcut(sg_plotdata.w1, q=16, retbins=True)
sg_spline = fit_spline(x=sg_plotdata.w1, y=sg_plotdata.sopr, bins=sg_raw_bins, k=2, s=0.05)
sg_spline_xs = np.linspace(0, 12, 100)
sg_spline_ys = sg_spline(sg_spline_xs)
sg_plotdata['adj_dists'] = sg_spline(sg_plotdata.w1)

bin_vols = []
plt_color = 'C0'
for ind in range(len(sg_raw_bins) - 1):
    bin_low = sg_raw_bins[ind]
    bin_high = sg_raw_bins[ind + 1]
    this_boxplot_data = sg_plotdata[np.logical_and(sg_plotdata.w1 > bin_low, sg_plotdata.w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt_dict = plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.5, bw_method=0.125)
        plt_dict['bodies'][0].set_facecolor(plt_color)
        plt_dict['bodies'][0].set_alpha(0.25)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in plt_dict:
                vp = plt_dict[partname]
                vp.set_edgecolor(plt_color)
plt.xticks(ticks = np.arange(0, 12.5, 1))
plt.yticks(ticks = np.arange(0, 1.05, 0.1))
plt.xlabel('W1-MDP distance from base MDP')
plt.ylabel('SOPR (lower is better)')
plt.title('W1-SOPR relationship, SimpleGrid')
plt.plot(sg_spline_xs, sg_spline_ys, color=plt_color, alpha=0.4, label='Calibration spline', linestyle='dotted')
plt.legend()
plt.tight_layout()
# plt.savefig('021_sg_calibration_curve.png', dpi=300)
plt.show()

sg_adj_bins = np.arange(0, 1.05, 0.1)
bin_vols = []
plt_color = 'C0'
for ind in range(len(sg_adj_bins) - 1):
    bin_low = sg_adj_bins[ind]
    bin_high = sg_adj_bins[ind + 1]
    this_boxplot_data = sg_plotdata[np.logical_and(sg_plotdata.adj_dists > bin_low, sg_plotdata.adj_dists <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt_dict = plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.05, bw_method=0.125)
        plt_dict['bodies'][0].set_facecolor(plt_color)
        plt_dict['bodies'][0].set_alpha(0.25)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in plt_dict:
                vp = plt_dict[partname]
                vp.set_edgecolor(plt_color)
plt.xlabel("Calibrated CHIRP")
plt.ylabel("SOPR (lower is better)")
plt.title("Calibrated CHIRP vs SOPR, SimpleGrid")
plt.xticks(ticks = np.arange(0, 1.05, 0.1))
plt.yticks(ticks = np.arange(0, 1.05, 0.1))
plt.tight_layout()
# plt.savefig('021_sg_calibrated_chirp.png', dpi=300)
plt.show()

# # # # #
# MetaWorld example
# # # # #

with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'mw_w1_dists.pkl'), 'rb') as file:
    mw_w1_dists = pickle.load(file)
mw_w1_dists = mw_w1_dists.rename(columns={'ns_test_ind': 'test_ind'})

mw_agent_eval_dir = os.path.join(results_path_local, 'mw_agent_perf')
all_filenames = glob.glob(os.path.join(mw_agent_eval_dir, "eval_[0-9]*.csv"))
mw_agent_evals = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = re.search(pattern="[0-9]+(?=\.csv)", string=filename)[0]
    mw_agent_evals = pd.concat((mw_agent_evals, tmp))
mw_agent_evals = mw_agent_evals.rename(columns={'ep_reward': 'reward'})

ret_max = mw_agent_evals.rename(columns={'reward': 'ret_max'}).groupby(['task', 'run']).ret_max.max()
ret_min = mw_agent_evals.rename(columns={'reward': 'ret_min'}).groupby(['task', 'run']).ret_min.min()
mw_agent_evals = mw_agent_evals.merge(ret_max, how='left', on=['task', 'run'])
mw_agent_evals = mw_agent_evals.merge(ret_min, how='left', on=['task', 'run'])
mw_agent_evals['sopr'] = (mw_agent_evals.ret_max - mw_agent_evals.reward) / (mw_agent_evals.ret_max - mw_agent_evals.ret_min)
mw_median_w1_dists = mw_w1_dists.drop('rep', axis=1).groupby(['task', 'test_ind']).median()
mw_agent_evals = mw_agent_evals.merge(mw_median_w1_dists, how='left', on=['task', 'test_ind'])

_, mw_raw_bins = pd.qcut(mw_agent_evals.w1, q=16, retbins=True)
mw_spline = fit_spline(x=mw_agent_evals.w1, y=mw_agent_evals.sopr, bins=mw_raw_bins, k=1, s=0.000001)
mw_spline_xs = np.linspace(0, max(mw_agent_evals.w1), 100)
mw_spline_ys = mw_spline(mw_spline_xs)
mw_agent_evals['adj_w1'] = mw_spline(mw_agent_evals.w1)

bin_vols = []
plt_color = 'C1'
for ind in range(len(mw_raw_bins) - 1):
    bin_low = mw_raw_bins[ind]
    bin_high = mw_raw_bins[ind + 1]
    this_boxplot_data = mw_agent_evals[np.logical_and(mw_agent_evals.w1 > bin_low, mw_agent_evals.w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt_dict = plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.075, bw_method=2e-2)
        plt_dict['bodies'][0].set_facecolor(plt_color)
        plt_dict['bodies'][0].set_alpha(0.4)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in plt_dict:
                vp = plt_dict[partname]
                vp.set_edgecolor(plt_color)
plt.title('W1-SOPR relationship, MetaWorld')
plt.xlabel('W1-MDP distance from base MDP')
plt.ylabel("SOPR (lower is better)")
plt.plot(mw_spline_xs, mw_spline_ys, color='C3', alpha=0.4, label='Calibration spline', linestyle='dotted')
plt.legend()
plt.tight_layout()
# plt.savefig('021_mw_calibration_curve.png', dpi=300)
plt.show()

mw_adj_bins = np.arange(0.0, 1.05, 0.1)
bin_vols = []
for ind in range(len(mw_adj_bins) - 1):
    bin_low = mw_adj_bins[ind]
    bin_high = mw_adj_bins[ind + 1]
    this_boxplot_data = mw_agent_evals[np.logical_and(mw_agent_evals.adj_w1 > bin_low, mw_agent_evals.adj_w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt_dict = plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.1, bw_method=2e-2)
        plt_dict['bodies'][0].set_facecolor(plt_color)
        plt_dict['bodies'][0].set_alpha(0.4)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            if partname in plt_dict:
                vp = plt_dict[partname]
                vp.set_edgecolor(plt_color)
plt.xlabel("Calibrated CHIRP distance from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("Calibrated CHIRP vs SOPR, MetaWorld")
plt.xticks(ticks = np.arange(0, 1.05, 0.1))
plt.yticks(ticks = np.arange(0, 1.05, 0.1))
plt.tight_layout()
# plt.savefig('021_mw_calibrated_chirp.png', dpi=300)
plt.show()

plt.plot(sg_spline_xs, sg_spline_ys, color='C0', alpha=0.8, label='SimpleGrid', linestyle='dotted')
plt.plot(mw_spline_xs, mw_spline_ys, color='C3', alpha=0.8, label='MetaWorld', linestyle='dotted')
plt.xlabel("Raw W1-MDP distance from base MDP")
plt.ylabel("Calibrated CHIRP")
plt.title("Calibration curves, SimpleGrid and MetaWorld")
plt.xticks(ticks=np.arange(0, 12.1, 1))
plt.yticks(ticks=np.arange(0, 1.05, 0.1))
plt.legend()
plt.tight_layout()
# plt.savefig('021_calibration_curves.png', dpi=300)
plt.show()

