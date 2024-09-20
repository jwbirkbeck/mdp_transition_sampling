import scipy.stats
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from src.grid_worlds.simple_grid_v2 import SimpleGridV2
from src.utils.filepaths import project_path_local

device = torch.device('cpu')
size = 20
env = SimpleGridV2(size=size, seed=0, device=device, render_mode='human')

with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'simplegrid_reward_shaped_sampling_3.pkl'), 'rb') as file:
    results_dict = pickle.load(file)
    true_rs = np.array(results_dict['true'])
    rs = np.array(results_dict['sampled'])
with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'simplegrid_random_sampling_3.pkl'), 'rb') as file:
    results_dict = pickle.load(file)
    true_r = np.array(results_dict['true'])
    random = np.array(results_dict['sampled'])

# # # # #
# Overall statistics:
# # # # #

'{:.2e}'.format(np.mean(true_rs - rs))
'{:.2e}'.format(np.mean(true_r - random))

'{:.2e}'.format(np.std(true_rs - rs))
'{:.2e}'.format(np.std(true_r - random))

import scipy
scipy.stats.ttest_rel(true_rs - rs, true_r - random)

plt.scatter(true_rs, true_rs - rs, s=3, alpha=0.25)
coef = np.polyfit(true_rs, true_rs - rs, 1)
linear_reg = np.poly1d(coef)
plt.plot(true_rs, linear_reg(true_rs), color='black', label=f'OLS: $y = {coef[0]:.5f}x {coef[1]:+.5f}$', alpha=0.5)
plt.xlabel("True W1 distance")
plt.xticks(np.arange(0, 12.1, 1))
plt.ylabel("Estimator error")
plt.ylim(-0.0035, 0.0035)
plt.title("Estimation errors, reward-shaped sampling")
plt.legend()
plt.tight_layout()
# plt.savefig('simplegrid_reward_shaped_sampling.png', dpi=300)
plt.show()

plt.scatter(true_r, true_r - random, s=3, alpha=0.25)
coef = np.polyfit(true_r, true_r - random, 1)
linear_reg = np.poly1d(coef)
plt.plot(true_r, linear_reg(true_r), color='black', label=f'OLS: $y = {coef[0]:.5f}x {coef[1]:+.5f}$', alpha=0.5)
plt.xlabel("True W1 distance")
plt.xticks(np.arange(0, 12.1, 1))
plt.ylabel("Estimator error")
plt.ylim(-0.0035, 0.0035)
plt.title("Estimation errors, random sampling")
plt.legend()
plt.tight_layout()
# plt.savefig('simplegrid_random_sampling.png', dpi=300)
plt.show()

scipy.stats.linregress(true_rs, true_rs - rs)
scipy.stats.linregress(true_r, true_r - random)

# # # # #
# Binned plot:
# # # # #
plotdata = pd.DataFrame({'true_rs': true_rs, 'rs': rs, 'true_r': true_r, 'rand': random})
_, bins = pd.qcut(plotdata.true_rs, q=10, retbins=True)

fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
bin_vols = []
coef = np.polyfit(true_rs, true_rs - rs, 1)
linear_reg = np.poly1d(coef)
plt.plot(true_rs, linear_reg(true_rs), color='black', label=f'OLS: $y = {coef[0]:.5f}x {coef[1]:+.5f}$', alpha = 0.35)
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.true_rs > bin_low, plotdata.true_rs <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot((this_boxplot_data.true_rs - this_boxplot_data.rs), positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=0.125)
plt.xlabel("True W1 distance from base MDP")
plt.ylabel("Estimation error")
plt.title("SimpleGrid: Estimation error for W1-MDP distance")
plt.xticks(ticks = np.arange(0, 12.5, 1))
plt.tight_layout()
# plt.savefig("simplegrid_sampled_dist_bias.png", dpi=300)
plt.show()
