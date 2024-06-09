import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import glob
from scipy.stats import pearsonr
from src.metaworld.wrapper import MetaWorldWrapper
from src.metaworld.nonstationarity_distribution import MWNSDistribution
from src.sampler.samplers import MDPDifferenceSampler

# # # # # # # # # #
# Calculate w1 dists between all tasks and non-stationarities
# # # # # # # # # #
task_selection = ['handle-press-side-v2',
                  'handle-press-v2',
                  'plate-slide-back-v2',
                  'reach-v2',
                  'reach-wall-v2']

nonstat_sequence_length = 100_000
low_inds = np.arange(0, nonstat_sequence_length / 10, nonstat_sequence_length / 100, dtype=int)
med_inds = np.arange(nonstat_sequence_length / 10, nonstat_sequence_length/2, nonstat_sequence_length / 20, dtype=int)
high_inds = np.arange(nonstat_sequence_length / 2, nonstat_sequence_length+1, nonstat_sequence_length / 4, dtype=int)

nonstat_eval_inds = np.concatenate((low_inds, med_inds, high_inds))
nonstat_eval_reps = 10

# env_a = MetaWorldWrapper(task_name=task_selection[0])
# env_a.change_task(task_name=task_selection[0], task_number=0)
# env_a.reset()
#
# env_b = MetaWorldWrapper(task_name=task_selection[0])
# env_b.change_task(task_name=task_selection[0], task_number=0)
# env_b.reset()
#
# state_space = env_a.observation_space
# action_space = env_a.action_space
#
# sampler = MDPDifferenceSampler(environment_a=env_a,
#                                environment_b=env_b,
#                                state_space=state_space,
#                                action_space=action_space)
#
# sampler.get_difference(n_states=50, n_transitions=5)

w1_dists = pd.DataFrame()
for task in task_selection:
    print(task)
    env_a = MetaWorldWrapper(task_name=task)
    env_a.change_task(task_name=task, task_number=0)
    env_a.reset()

    env_b = MetaWorldWrapper(task_name=task)
    env_b.change_task(task_name=task, task_number=0)
    env_b.reset()

    state_space = env_a.observation_space
    action_space = env_a.action_space

    sampler = MDPDifferenceSampler(environment_a=env_a,
                                   environment_b=env_b,
                                   state_space=state_space,
                                   action_space=action_space)

    ns_dist = MWNSDistribution(seed=0,
                               state_space=state_space,
                               action_space=action_space,
                               current_task=env_a.task_name)
    ns_dist.task_dist.set_prob_task_change(probability=0.0)
    ns_dist.maint_dist.set_prob_maintenance(probability=0.0)
    ns_dist.generate_sequence(sequence_length=nonstat_sequence_length + 1)
    ns_dist.freeze()
    env_b.ns_dist = ns_dist

    for test_ind in nonstat_eval_inds:
        print(test_ind)
        ns_dist.set_sequence_ind(ind=test_ind)
        for _ in range(nonstat_eval_reps):
            dist = sampler.get_difference(n_states=50, n_transitions=5)
            print(dist)
            pd_row = pd.DataFrame({'task': [task], 'test_ind': [test_ind],
                                   'w1': [dist]})
            w1_dists = pd.concat((w1_dists, pd_row))
    w1_dists.to_csv("w1_dists.csv", index=False)
print("done")

# # # # # # # # # #
# Load IRIDIS runs of agent performance in each nonstat
# # # # # # # # # #

w1_dists = pd.read_csv("w1_dists.csv")
w1_dists['test_ind'] = w1_dists['ns_seq_ind']

results_dir = "/opt/project/results/cont_nonstat_w1_vs_returns/"
all_filenames = glob.glob(results_dir + "train_[0-9]*.csv")
train_results = pd.DataFrame()
tmp_ind = 0
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = tmp_ind
    tmp_ind += 1
    train_results = pd.concat((train_results, tmp))

all_filenames = glob.glob(results_dir + "eval_[0-9]*.csv")
eval_results = pd.DataFrame()
tmp_ind = 0
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = tmp_ind
    tmp_ind += 1
    eval_results = pd.concat((eval_results, tmp))

fig, ax = plt.subplots(figsize=(9 / 1.2, 6 / 1.2))
for run in train_results.run.unique():
    plotdata = train_results[train_results.run == run]['rewards'].values
    plt.plot(plotdata, label=run)
    plt.title(run)
plt.show()

good_runs = list(train_results.query('episode == 749 and rewards > 4000')['run'].unique())

train_results2 = train_results.query(f'run in {good_runs}')
eval_results2 = eval_results.query(f'run in {good_runs}')

w1_dists2 = w1_dists

ax = w1_dists2.groupby('test_ind').boxplot(column=['w1'], subplots=False)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
ax.set_xticks(ticks=range(1, len(w1_dists.test_ind.unique()) + 1), labels=w1_dists.test_ind.unique())
plt.xlabel("Non-stationarity sequence Index")
plt.ylabel("Measured Wasserstein distance")
plt.ylim(0, 5)
plt.tight_layout()
# plt.savefig('w1_vs_test_ind.png')
plt.show()

plt.scatter(w1_dists.test_ind.unique(), w1_dists.groupby('test_ind')['w1'].median())
plt.xlabel("Non-stationarity sequence Index")
plt.ylabel("Median measured Wasserstein distance")
plt.show()

fig, ax = plt.subplots()
for ind in w1_dists.test_ind.unique():
    plotdata = w1_dists[w1_dists.test_ind==ind]
    err = np.percentile(plotdata.w1, [25, 75])
    plt.errorbar(x=ind, y=plotdata.w1.median(), yerr=err[1] - err[0], capsize=2)
    # plt.boxplot(plotdata.w1, positions=[ind], widths=[10000])
ax.set_xticks(ticks=np.arange(0, 100001, 10000), labels=np.arange(0, 100001, 10000))
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.tight_layout()
plt.show()

plotdata = pd.merge(eval_results, w1_dists, on=['task', 'test_ind'])
plotdata2 = plotdata[plotdata.w1 < 2.5]
_, bins = pd.qcut(plotdata2.w1, q=10, retbins=True)
fig, ax = plt.subplots(figsize=(8 / 1.2, 5 / 1.2))
for ind in range(len(bins) - 1):
    bin_low = bins[ind]
    bin_high = bins[ind + 1]
    this_boxplot_data = plotdata2[np.logical_and(plotdata2.w1 > bin_low, plotdata2.w1 <= bin_high)]
    position = bin_low + (bin_high - bin_low) / 2
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.ep_reward, positions=[position], showmedians=True,
                       showextrema=False, widths=0.1, bw_method=2e-2)

plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.xlabel("W1 distance from training MDP")
plt.ylabel("Episode Total Rewards")
plt.title("Agent Returns vs W1 to new MDP, five-task aggregation")
plt.tight_layout()
# plt.savefig("w1_vs_performance_violinplot_5_tasks.png")
plt.show()

for task in task_selection:
    plotdata = pd.merge(eval_results, w1_dists, on=['task', 'test_ind'])
    plotdata = plotdata[plotdata.task == task]
    plotdata2 = plotdata[plotdata.w1 < 2.5]
    _, bins = pd.qcut(plotdata2.w1, q=10, retbins=True)
    fig, ax = plt.subplots(figsize=(8/1.2, 5/1.2))
    for ind in range(len(bins) - 1):
        bin_low = bins[ind]
        bin_high = bins[ind+1]
        this_boxplot_data = plotdata2[np.logical_and(plotdata2.w1 > bin_low, plotdata2.w1 <= bin_high)]
        position = bin_low + (bin_high - bin_low) / 2
        if this_boxplot_data.shape[0] > 0:
            plt.violinplot(this_boxplot_data.ep_reward, positions=[position], showmedians=True,
                           showextrema=False, widths=0.1, bw_method=2e-2)

    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    plt.xlabel("W1 distance from training MDP")
    plt.ylabel("Episode Total Rewards")
    plt.title("Agent Returns vs W1 to new MDP, " + str(task))
    plt.tight_layout()
    plt.savefig("w1_vs_performance_violinplot_" + str(task) + ".png")
    plt.show()

pearsonr(plotdata.w1, plotdata.ep_reward)

plotdata = eval_results
ax = plotdata.reset_index().groupby('test_ind').boxplot(column=['ep_reward'], subplots=False)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
ax.set_xticks(ticks=range(1, len(plotdata.test_ind.unique()) + 1), labels=plotdata.test_ind.unique())
# plt.xlabel("Non-stationarity sequence Index")
# plt.ylabel("Measured Wasserstein distance")
plt.tight_layout()
# plt.savefig('w1_vs_test_ind.png')
plt.show()

plotdata = pd.merge(eval_results, w1_dists, on=['task', 'test_ind'])
plotdata2 = plotdata[plotdata.w1 < 2.5]
for task in task_selection:
    ax = plotdata2.query("task=='" + task + "'").reset_index().groupby('w1').boxplot(column=['ep_reward'], subplots=False)
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    ax.set_xticks(ticks=range(1, len(plotdata.test_ind.unique()) + 1), labels=plotdata.test_ind.unique())
    # plt.xlabel("Non-stationarity sequence Index")
    # plt.ylabel("Measured Wasserstein distance")
    plt.title(task)
    plt.tight_layout()
    # plt.savefig('w1_vs_test_ind.png')
    plt.show()

fig, ax = plt.subplots(figsize=(9/1.2, 5/1.2))
bins = list(eval_results2.test_ind.unique())
plotdata = eval_results2
for ind, bin in enumerate(bins):
    this_boxplot_data = plotdata.query('test_ind == ' + str(bin))
    if this_boxplot_data.shape[0] > 0:
        plt.violinplot(this_boxplot_data.ep_reward, positions=[ind], widths=1,
                       showmedians=1, bw_method=5e-2, showextrema=False)
ax.set_xticks(ticks=range(0, len(plotdata.test_ind.unique())), labels=plotdata.test_ind.unique())
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.ylabel("Episode Reward")
plt.tight_layout()
plt.show()