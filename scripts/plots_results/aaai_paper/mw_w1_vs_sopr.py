import glob, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool_10
import pickle

# w1_dists = pd.read_csv('/opt/project/scripts/plots_results/cont_nonstat_w1_vs_returns/w1_dists_new.csv')
# results_dir = "/opt/project/results/cont_nonstat_w1_vs_returns/"

with open('/opt/project/scripts/local_experiments/mw_w1_dists.pkl', 'rb') as file:
    mw_w1_dists = pickle.load(file)
mw_w1_dists = mw_w1_dists.rename(columns={'ns_test_ind': 'test_ind'})

results_dir = "/opt/project/results/mw_w1_vs_sopr/"

all_filenames = glob.glob(os.path.join(results_dir, "eval_[0-9]*.csv"))
agent_results = pd.DataFrame()
tmp_ind = 0
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = tmp_ind
    tmp_ind += 1
    agent_results = pd.concat((agent_results, tmp))

# ret_max = agent_results.groupby(['task'])['reward'].max()
# ret_min = agent_results.groupby(['task'])['reward'].min()

ret_max = agent_results.rename(columns={'reward': 'ret_max'}).groupby(['task', 'test_ind']).ret_max.max()
ret_min = agent_results.rename(columns={'reward': 'ret_min'}).groupby(['task', 'test_ind']).ret_min.min()

agent_results = agent_results.merge(ret_max, how='left', on=['task', 'test_ind'])
agent_results = agent_results.merge(ret_min, how='left', on=['task', 'test_ind'])
agent_results['sopr'] = (agent_results.ret_max - agent_results.reward) / (agent_results.ret_max - agent_results.ret_min)

median_w1_dists = mw_w1_dists.drop('rep', axis=1).groupby(['task', 'test_ind']).median()

agent_results = agent_results.merge(median_w1_dists, how='left', on=['task', 'test_ind'])

# # # # # # # # # #
# Non-stationarity vs w1
# # # # # # # # # #
fig, ax = plt.subplots()
for ind in agent_results.test_ind.unique():
    plotdata = agent_results[agent_results.test_ind==ind]
    err = np.percentile(plotdata.w1, [25, 75])
    plt.errorbar(x=ind, y=plotdata.w1.median(), yerr=err[1] - err[0], capsize=2)
ax.set_xticks(ticks=np.arange(0, 100001, 10000), labels=np.arange(0, 100001, 10000))
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.show()

# bins = [0.03, 0.15, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]

plotdata = agent_results # [agent_results.w1 < 1.2]
_, bins = pd.qcut(plotdata.w1, q=8, retbins=True)
fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
# plt.plot(x[order], y_sm[order], color='black', label='LOWESS', alpha=0.2)
# plt.fill_between(x[order], y_sm[order] - 1.96*y_std[order], y_sm[order] + 1.96*y_std[order], alpha=0.15, label='LOWESS uncertainty')
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.w1 > bin_low, plotdata.w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.reward, positions=[position], showmedians=True, showextrema=False, widths=0.15, bw_method=2e-2)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.xlabel("W1 distance from from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SOPR against Wasserstein MDP distance")
# plt.xticks(ticks = np.arange(0, 2.1, 2/10), rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
# plt.savefig("mw_w1_vs_sopr_5_mdps.png", dpi=300)
plt.show()


