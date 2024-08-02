import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool_10
import scipy


results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'random_vs_reward_based_sampling/')
all_filenames = glob.glob(results_path + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    results = pd.concat((results, tmp))
results = results.reset_index()

results = results[results.env_a.isin(task_pool_10)]
results = results[results.env_b.isin(task_pool_10)]

results[['n_states', 'n_transitions']].drop_duplicates()

results = results.query("n_states == 50 and n_transitions == 5")
# scipy.stats.pearsonr(results.rew_dist, soprs)
# scipy.stats.pearsonr(random, soprs)

task_sel = results.env_a.unique()
results = results[results.env_a.isin(task_sel)]
results = results[results.env_b.isin(task_sel)]
results = results.reset_index()
task_sel = [task_sel[0]]

plt.rcParams.update({'figure.autolayout': True})
for task in [task_sel[2]]:
    plotdata = results[results.env_a == task]
    if len(plotdata) != 0:
        ax = plotdata.groupby('env_b').boxplot(column=['rand_dist'], subplots=False, figsize=(9/1.4, 6/1.4))
        env_b_list = results.env_b.unique().tolist()
        ax.set_xticks(ticks=range(1, len(env_b_list) + 1), labels=env_b_list)
        ax.get_xticklabels()[env_b_list.index(task)].set_color('red')
        plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
        plt.ylim(0, 7.0)
        plt.ylabel("Approx. W1 distance")
        plt.title("W1 distance approximations, smallest samples")
        plt.tight_layout()
        # plt.savefig('mdp_distances_smallest_sample_reach.png', dpi=200)
        plt.show()

        ax = plotdata.groupby('env_b').boxplot(column=['rew_dist'], subplots=False, figsize=(9/1.4, 6/1.4))
        env_b_list = results.env_b.unique().tolist()
        ax.set_xticks(ticks=range(1, len(env_b_list) + 1), labels=env_b_list)
        ax.get_xticklabels()[env_b_list.index(task)].set_color('red')
        plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
        plt.ylim(0, 7.0)
        plt.ylabel("Approx. W1 distance")
        plt.title("W1 distance approximations, smallest samples")
        plt.tight_layout()
        # plt.savefig('mdp_distances_smallest_sample_reach.png', dpi=200)
        plt.show()

fig, ax = plt.subplots(figsize=(9/1.4, 6/1.4))
for ind, dist in enumerate(['rand_dist', 'rew_dist']):
    plotdata = results.groupby(['env_a', 'env_b', 'n_states', 'n_transitions'])
    plt.violinplot(plotdata[dist].std(), positions=[ind], bw_method=0.4, showmedians=True)
ax.set_xticks(ticks=[0, 1], labels=['Random', 'Proposed'])
plt.xlabel('Sampling scheme')
plt.ylabel("Standard deviation of estimate")
plt.title("Standard deviation of estimation (lower is better)")
plt.tight_layout()
plt.savefig("std_of_schemes.png", dpi=200)
plt.show()

results.groupby(['env_a', 'env_b', 'n_states', 'n_transitions'])[['rand_dist', 'rew_dist']].std().boxplot()
plt.show()

results.groupby(['env_a', 'env_b', 'n_states', 'n_transitions'])[['rand_dist', 'rew_dist']].std().mean()
results.groupby(['env_a', 'env_b', 'n_states', 'n_transitions'])[['rand_dist', 'rew_dist']].std().std()
