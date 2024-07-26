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
    true = np.array(results_dict['true'])
    reward_shaped = np.array(results_dict['sampled'])
with open(os.path.join(project_path_local, 'scripts', 'local_experiments', 'simplegrid_random_sampling.pkl'), 'rb') as file:
    results_dict = pickle.load(file)
    true2 = np.array(results_dict['true'])
    random = np.array(results_dict['sampled'])

# # # # #
# Overall statistics:
# # # # #
np.mean(true - reward_shaped)
np.mean(true - random)

np.std(true - reward_shaped)
np.std(true - random)

plt.scatter(true, true - reward_shaped, s=3, alpha=0.25)
coef = np.polyfit(true,true - reward_shaped,1)
linear_reg = np.poly1d(coef)
plt.plot(true, linear_reg(true), color='black', label=f'$y = {coef[0]:.5f}x {coef[1]:+.5f}$')
plt.xlabel("True W1 distance")
plt.xticks(np.arange(0, 12.1, 1))
plt.ylabel("Raw rstimation error")
plt.title("Estimation errors by distance for reward-shaped sampling")
plt.legend()
plt.tight_layout()
# plt.savefig('sampling_bias_by_distance.png', dpi=300)
plt.show()

plt.scatter(true, true - random, s=3, alpha=0.25)
coef = np.polyfit(true, true - random,1)
linear_reg = np.poly1d(coef)
plt.plot(true, linear_reg(true), color='black', label=f'$y = {coef[0]:.5f}x {coef[1]:+.5f}$')
plt.xlabel("True W1 distance")
plt.xticks(np.arange(0, 12.1, 1))
plt.ylabel("Raw estimation error")
plt.title("Estimation errors by distance for reward-shaped sampling")
plt.legend()
plt.tight_layout()
# plt.savefig('sampling_bias_by_distance.png', dpi=300)
plt.show()

scipy.stats.linregress(true, true-reward_shaped)
scipy.stats.linregress(true, true-random)

# # # # #
# Binned plot:
# # # # #
plotdata = pd.DataFrame({'true': true, 'reward_shaped': reward_shaped})
_, bins = pd.qcut(plotdata.true, q=10, retbins=True)

fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
# plt.plot(x[order], y_sm[order], color='black', label='LOWESS', alpha=0.2)
# plt.fill_between(x[order], y_sm[order] - 1.96*y_std[order], y_sm[order] + 1.96*y_std[order], alpha=0.15, label='LOWESS uncertainty')
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.true > bin_low, plotdata.true <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot((this_boxplot_data.reward_shaped - this_boxplot_data.true), positions=[position], showmedians=True, showextrema=False, widths=0.75, bw_method=0.125)
plt.xlabel("True W1 distance from base MDP")
plt.ylabel("Estimation error")
plt.title("SimpleGrid: Estimation error for W1-MDP distance")
plt.xticks(ticks = np.arange(0, 12.5, 1))
plt.tight_layout()
plt.savefig("simplegrid_sampled_dist_bias.png", dpi=300)
plt.show()
