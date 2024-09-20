import ot
import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from src.grid_worlds.simple_grid_v2 import SimpleGridV2
from src.utils.filepaths import *
import matplotlib.pyplot as plt
import scipy


n_agents = 10
size = 20
# goal_x_bounds = [1, size-1]
# goal_y_bounds = [int(3 * size / 4), size - 1]
#
# agent_x_bounds = [1, size-1]
# agent_y_bounds = [int(size / 4), size - 1]
#
# count = 0
# for g_x in range(goal_x_bounds[0], goal_x_bounds[1]):
#     for g_y in range(goal_y_bounds[0], goal_y_bounds[1]):
#         for a_x in range(agent_x_bounds[0], agent_x_bounds[1]):
#             for a_y in range(agent_y_bounds[0], agent_y_bounds[1]):
#                 count += 1
# print('done')
# # 20 x 20 = 400 different MDP variants

def calc_soprs(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device('cpu')
    size = 20

    env_a = SimpleGridV2(size=size, seed=seed, device=device, render_mode='human')
    env_b = SimpleGridV2(size=size, seed=seed, device=device, render_mode='human')

    soprs = []
    w1s = []
    for rep in range(750):
        env_b.seed = np.random.randint(low=0, high=int(1e6), size=1).item()

        env_a.reset()
        env_b.reset()

        opt_b = env_b.get_analytical_return(minimize=False)
        min_b = env_b.get_analytical_return(minimize=True)
        a_in_b = env_b.get_analytical_return(alt_goal_pos=env_a._goal_pos, minimize=False)
        sopr = (opt_b - a_in_b) / (opt_b - min_b)
        soprs.append(sopr)

        next_states_a, rewards_a = env_a.get_all_transitions(render=False)
        next_states_b, rewards_b = env_b.get_all_transitions(render=False)
        samples_a = torch.cat((next_states_a, rewards_a), dim=1)
        samples_b = torch.cat((next_states_b, rewards_b), dim=1)
        a = np.ones(shape=samples_a.shape[0]) / samples_a.shape[0]
        b = np.ones(shape=samples_b.shape[0]) / samples_b.shape[0]
        M = ot.dist(samples_a.numpy(), samples_b.numpy(), metric='euclidean', p=1)
        w1s.append(ot.emd2(a=a, b=b, M=M))
    return w1s, soprs

n_jobs = 14
w1s = []
soprs = []
with mp.Pool(n_jobs) as pool:
    output = pool.map(calc_soprs, list(range(n_jobs)))
    for job in output:
        w1s += job[0]
        soprs += job[1]
print('done')

sg_chirp_sopr = {'w1s': w1s, 'soprs': soprs}

with open(os.path.join(project_path_local, 'pickles', 'sg_chirp_sopr.pkl'), 'wb') as file:
    pickle.dump(sg_chirp_sopr, file)

plotdata = pd.DataFrame(sg_chirp_sopr)

_, bins = pd.qcut(plotdata.w1s, q=16, retbins=True)
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.w1s > bin_low, plotdata.w1s <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.soprs, positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=0.125)
plt.xlabel("W1 distance from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SimpleGrid: SOPR vs W1-MDP distance")
plt.xticks(ticks = np.arange(0, 26, 2))
plt.tight_layout()
# plt.savefig("010_simplegrid_w1_vs_sopr.png", dpi=300)
plt.show()

def lin_func(var, coeff):
    return var * coeff

linreg =  scipy.stats.linregress(plotdata.w1s.values, plotdata.soprs.values)
plt.scatter(plotdata.w1s.values, plotdata.soprs.values, s=3, alpha=0.25)
plt.plot(plotdata.w1s.values, linreg.intercept + linreg.slope*plotdata.w1s.values, 'r', label='fitted line')
plt.show()

model_popt, model_pcov = scipy.optimize.curve_fit(f=lin_func, xdata=plotdata.w1s.values, ydata=plotdata.soprs.values)
plt.scatter(plotdata.w1s.values, plotdata.soprs.values, s=3)
# plt.plot(plotdata.w1s.values, model_popt[1] + model_popt[0]*plotdata.w1s.values, 'r', label='fitted line')
plt.plot(plotdata.w1s.values, model_popt[0]*plotdata.w1s.values, 'r', label='fitted line')
plt.show()

coeffs = np.polyfit(plotdata.w1s.values, plotdata.soprs.values, deg=3)
plt.scatter(plotdata.w1s.values, plotdata.soprs.values, s=3, alpha=0.2)
plt.plot(plotdata.w1s.values[np.argsort(plotdata.w1s.values)],
         coeffs[0]*plotdata.w1s.values[np.argsort(plotdata.w1s.values)]**3 +
         coeffs[1]*plotdata.w1s.values[np.argsort(plotdata.w1s.values)]**2 +
         coeffs[2]*plotdata.w1s.values[np.argsort(plotdata.w1s.values)]**1 +
         coeffs[3]*plotdata.w1s.values[np.argsort(plotdata.w1s.values)]**0, 'r', label='fitted line')
plt.show()


corr = scipy.stats.pearsonr(plotdata.w1s.values, plotdata.soprs.values)
corr[1]

scipy.stats.spearmanr(plotdata.w1s.values, plotdata.soprs.values)
