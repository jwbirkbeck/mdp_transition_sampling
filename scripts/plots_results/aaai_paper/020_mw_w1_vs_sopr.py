import glob, sys, os, re
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from src.utils.consts import task_pool_10
from src.utils.filepaths import *
import pickle

with open('/opt/project/scripts/local_experiments/mw_w1_dists.pkl', 'rb') as file:
    mw_w1_dists = pickle.load(file)
mw_w1_dists = mw_w1_dists.rename(columns={'ns_test_ind': 'test_ind'})

agent_eval_dir = os.path.join(results_path_local, 'mw_agent_perf')
all_filenames = glob.glob(os.path.join(agent_eval_dir, "eval_[0-9]*.csv"))
agent_evals = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = re.search(pattern="[0-9]+(?=\.csv)", string=filename)[0]
    agent_evals = pd.concat((agent_evals, tmp))

agent_evals = agent_evals.rename(columns={'ep_reward': 'reward'})

ret_max = agent_evals.rename(columns={'reward': 'ret_max'}).groupby(['task', 'run']).ret_max.max()
ret_min = agent_evals.rename(columns={'reward': 'ret_min'}).groupby(['task', 'run']).ret_min.min()

agent_evals = agent_evals.merge(ret_max, how='left', on=['task', 'run'])
agent_evals = agent_evals.merge(ret_min, how='left', on=['task', 'run'])

agent_evals['sopr'] = (agent_evals.ret_max - agent_evals.reward) / (agent_evals.ret_max - agent_evals.ret_min)

median_w1_dists = mw_w1_dists.drop('rep', axis=1).groupby(['task', 'test_ind']).median()
agent_evals = agent_evals.merge(median_w1_dists, how='left', on=['task', 'test_ind'])

plotdata = agent_evals
_, bins = pd.qcut(plotdata.w1, q=16, retbins=True)
bin_vols = []
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata[np.logical_and(plotdata.w1 > bin_low, plotdata.w1 <= bin_high)]
    bin_vols.append(this_boxplot_data.shape[0])
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.075, bw_method=2e-2)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.xlabel("W1 distance from base MDP")
plt.ylabel("SOPR (lower is better)")
plt.title("SOPR against W1-MDP distance")
plt.xticks(ticks = np.arange(0, 1.21, 0.1), rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.savefig("mw_w1_vs_sopr.png", dpi=300)
plt.show()

# # w1_dists = pd.read_csv('/opt/project/scripts/plots_results/cont_nonstat_w1_vs_returns/w1_dists_new.csv')
# # results_dir = "/opt/project/results/cont_nonstat_w1_vs_returns/"
#
# with open('/opt/project/scripts/local_experiments/mw_w1_dists.pkl', 'rb') as file:
#     mw_w1_dists = pickle.load(file)
# mw_w1_dists = mw_w1_dists.rename(columns={'ns_test_ind': 'test_ind'})
#
# with open('/opt/project/scripts/local_experiments/mw_w1_dists_random.pkl', 'rb') as file:
#     mw_w1_dists_random = pickle.load(file)
# mw_w1_dists_random = mw_w1_dists_random.rename(columns={'ns_test_ind': 'test_ind'})
#
# # 20 agents were trained for each task - some did not converge toward the optimal policy.
# # To avoid this tainting the result, we select the best agent of the 20 by highest mean performance in the base MDP:
# training_results_dir = "/opt/project/results/archive/c3_003a_train_agents/"
# all_filenames = glob.glob(os.path.join(training_results_dir, "eval_[0-9]*.csv"))
# agent_training = pd.DataFrame()
# tmp_ind = 0
# for filename in all_filenames:
#     tmp = pd.read_csv(filename)
#     tmp['run'] = tmp_ind
#     tmp_ind += 1
#     agent_training = pd.concat((agent_training, tmp))
#
# task_agent_dict = {}
# for task_ind, task in enumerate(agent_training.task.unique().tolist()):
#     best_agent_ind = agent_training.query(f"task=='{task}'").groupby(['run'])['rewards'].rolling(window=50).mean().groupby('run').tail(1).argmax()
#     task_agent_dict[task] = best_agent_ind
#
#
# eval_results_dir = "/opt/project/results/mw_w1_vs_sopr/"
# all_filenames = glob.glob(os.path.join(eval_results_dir, "eval_[0-9]*.csv"))
# eval_results = pd.DataFrame()
# tmp_ind = 0
# for filename in all_filenames:
#     tmp = pd.read_csv(filename)
#     tmp['run'] = tmp_ind
#     tmp_ind += 1
#     eval_results = pd.concat((eval_results, tmp))
#
# eval_results_filtered = pd.DataFrame()
# for key in task_agent_dict:
#     task = key
#     agent_index = task_agent_dict[key]
#     best_results_for_task = eval_results.query(f"task=='{task}' and run=={agent_index}")
#     best_results_for_task = best_results_for_task.sort_values(by=['test_ind', 'eval_ind'])
#     best_results_for_task = best_results_for_task.drop(labels=['eval_ind', 'run'], axis=1)
#     eval_results_filtered = pd.concat((eval_results_filtered, best_results_for_task))
#
# ret_max = eval_results_filtered.rename(columns={'reward': 'ret_max'}).groupby(['task', 'test_ind']).ret_max.max()
# ret_min = eval_results_filtered.rename(columns={'reward': 'ret_min'}).groupby(['task', 'test_ind']).ret_min.min()
#
# eval_results_filtered = eval_results_filtered.merge(ret_max, how='left', on=['task', 'test_ind'])
# eval_results_filtered = eval_results_filtered.merge(ret_min, how='left', on=['task', 'test_ind'])
# eval_results_filtered['sopr'] = (eval_results_filtered.ret_max - eval_results_filtered.reward) / (eval_results_filtered.ret_max - eval_results_filtered.ret_min)
# eval_results_filtered['sopr'] = (5000 - eval_results_filtered.reward) / 5000
#
# median_w1_dists = mw_w1_dists.drop('rep', axis=1).groupby(['task', 'test_ind']).median()
# # median_w1_dists_random = mw_w1_dists_random.drop('rep', axis=1).groupby(['task', 'test_ind']).median()
# # median_w1_dists_random = median_w1_dists_random.rename(columns={'w1': 'w1_random'})
# eval_results_filtered = eval_results_filtered.merge(median_w1_dists, how='left', on=['task', 'test_ind'])
# # eval_results_filtered = eval_results_filtered.merge(median_w1_dists_random, how='left', on=['task', 'test_ind'])
# #
# # scipy.stats.pearsonr(eval_results_filtered.w1, eval_results_filtered.sopr)
# # scipy.stats.pearsonr(eval_results_filtered.w1_random, eval_results_filtered.sopr)
# #
# # scipy.stats.linregress(eval_results_filtered.w1, eval_results_filtered.sopr)
# # scipy.stats.linregress(eval_results_filtered.w1_random, eval_results_filtered.sopr)
# #
# # scipy.spatial.distance.correlation(eval_results_filtered.w1, eval_results_filtered.sopr)
# # scipy.spatial.distance.correlation(eval_results_filtered.w1_random, eval_results_filtered.sopr)
#
#
# # # # # # # # # # #
# # Non-stationarity vs w1
# # # # # # # # # # #
# fig, ax = plt.subplots()
# for ind in eval_results_filtered.test_ind.unique():
#     plotdata = eval_results_filtered[eval_results_filtered.test_ind==ind]
#     err = np.percentile(plotdata.w1, [25, 75])
#     plt.errorbar(x=ind, y=plotdata.w1.median(), yerr=err[1] - err[0], capsize=2)
# ax.set_xticks(ticks=np.arange(0, 100001, 10000), labels=np.arange(0, 100001, 10000))
# plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
# plt.tight_layout()
# plt.show()
#
# # bins = [0.03, 0.15, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
#
# plotdata = eval_results_filtered#.query("task=='reach-v2'") # [agent_results.w1 < 1.2]
# _, bins = pd.qcut(plotdata.w1, q=10, retbins=True)
# bins = [0] + list(bins)
# bin_vols = []
# for ind in range(len(bins) - 1):
#     bin_low = bins[ind]
#     bin_high = bins[ind + 1]
#     this_boxplot_data = plotdata[np.logical_and(plotdata.w1 > bin_low, plotdata.w1 <= bin_high)]
#     bin_vols.append(this_boxplot_data.shape[0])
#     position = bin_low + (bin_high - bin_low) / 2
#     if this_boxplot_data.shape[0] > 0:
#         plt.violinplot(this_boxplot_data.sopr, positions=[position], showmedians=True, showextrema=False, widths=0.1, bw_method=2e-2)
# plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
# plt.xlabel("W1 distance from base MDP")
# plt.ylabel("SOPR (lower is better)")
# plt.title("SOPR against Wasserstein MDP distance")
# plt.xticks(ticks = np.arange(0, 1.21, 0.1), rotation=-45, ha='left', rotation_mode='anchor')
# plt.tight_layout()
# plt.savefig("mw_w1_vs_sopr_5_mdps.png", dpi=300)
# plt.show()
#
# eval_results_filtered.groupby('test_ind')['reward'].mean().plot()
# plt.show()
#
# eval_results_filtered.query("run == 0 and task=='reach-v2").groupby('task')['reward'].max()
