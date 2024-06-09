import glob, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool, task_pool_10


"""
Load the saved w1_distances
Load the agent performances for each of the five tasks
Translate agent performance into SOPR
Merge them together and plot w1 vs SOPR 
"""

w1_dists = pd.read_csv('/opt/project/scripts/plots_results/cont_nonstat_w1_vs_returns/w1_dists_new.csv')

results_dir = "/opt/project/results/cont_nonstat_w1_vs_returns/"
all_filenames = glob.glob(os.path.join(results_dir, "eval_[0-9]*.csv"))
agent_results = pd.DataFrame()
tmp_ind = 0
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = tmp_ind
    tmp_ind += 1
    agent_results = pd.concat((agent_results, tmp))

ret_max = agent_results.rename(columns={'ep_reward': 'ret_max'}).groupby(['task', 'test_ind']).ret_max.max()
ret_min = agent_results.rename(columns={'ep_reward': 'ret_min'}).groupby(['task', 'test_ind']).ret_min.min()

agent_results = agent_results.merge(ret_max, how='left', on=['task', 'test_ind'])
agent_results = agent_results.merge(ret_min, how='left', on=['task', 'test_ind'])
agent_results['sopr'] = (agent_results.ret_max - agent_results.ep_reward) / (agent_results.ret_max - agent_results.ret_min)

median_w1_dists = w1_dists.drop('rep', axis=1).groupby(['task', 'test_ind']).median()

agent_results = agent_results.merge(median_w1_dists, how='left', on=['task', 'test_ind'])

# # # # # # # # # #
# Non-stationarity vs w1
# # # # # # # # # #
fig, ax = plt.subplots()
for ind in w1_dists.test_ind.unique():
    plotdata = w1_dists[w1_dists.test_ind==ind]
    err = np.percentile(plotdata.w1, [25, 75])
    plt.errorbar(x=ind, y=plotdata.w1.median(), yerr=err[1] - err[0], capsize=2)
ax.set_xticks(ticks=np.arange(0, 100001, 10000), labels=np.arange(0, 100001, 10000))
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.show()

# # # # # # # # # #
# sopr vs w1
# # # # # # # # # #
plotdata = agent_results[agent_results.w1 < 2.0]
_, bins = pd.qcut(plotdata.w1, q=10, retbins=True)
fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.w1 > bin_low, plotdata.w1 <= bin_high)]
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.05, bw_method=2e-2)
plt.show()

plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.xlabel("W1 distance from training MDP")
plt.ylabel("Episode Total Rewards")
plt.title("Agent Returns vs W1 to new MDP, five-task aggregation")
plt.tight_layout()
# plt.savefig("w1_vs_performance_violinplot_5_tasks.png")
plt.show()