import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.utils.consts import task_pool_10

results_dir = "/opt/project/results/simple_w1_vs_returns_test/"
all_filenames = glob.glob(results_dir + "train_[0-9]*.csv")
train_results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    train_results = pd.concat((train_results, tmp))

all_filenames = glob.glob(results_dir + "test_[0-9]*.csv")
test_results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    test_results = pd.concat((test_results, tmp))

del train_results['Unnamed: 0']
del test_results['Unnamed: 0']

task_pool = train_results.env.unique()

results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
w_dists = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['index'] = None
    w_dists = pd.concat((w_dists, tmp))
w_dists.reset_index()
w_dists = w_dists.query('n_states == 25 and n_transitions == 10')
w_dists = w_dists[w_dists.env_a.isin(task_pool)]
w_dists = w_dists[w_dists.env_b.isin(task_pool)]


for ind in range(5):
    fig, ax = plt.subplots()
    q = "env_a == '" + task_pool[ind] + "' and episodes >= 950"
    plotdata = test_results.query(q)
    plotdata['sopr'] = (max(plotdata['rewards']) - plotdata['rewards']) / (max(plotdata['rewards']) - min(plotdata['rewards']))
    for ind2 in range(5):
        plt.violinplot(plotdata['sopr'][plotdata.env_b==task_pool[ind2]], positions=[ind2+1], showmedians=True, bw_method=0.3)
    # ax = plotdata.drop(['env_a', 'episodes', 'rewards'], axis=1).reset_index().groupby('env_b').boxplot(column=['sopr'], subplots=False, figsize=(9 / 1.2, 6 / 1.2))
    ax.set_xticks(ticks=range(1, 6), labels=task_pool.tolist())
    ax.get_xticklabels()[ind].set_color('red')
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    plt.ylabel("SOPR")
    plt.xlabel("Task")
    plt.tight_layout()
    # plt.savefig('w1_vs_task_' + str(ind) + '_perf.png')
    plt.show()

for ind in range(5):
    ax = w_dists.query("env_a == '" + task_pool[ind] + "'").groupby('env_b').boxplot(column=['dist'], subplots=False)
    ax.set_xticks(ticks=range(1, len(task_pool) + 1), labels=task_pool)
    ax.get_xticklabels()[ind].set_color('red')
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    plt.ylabel("Wasserstein Distance")
    plt.xlabel("Task")
    plt.tight_layout()
    # plt.savefig('w1_vs_task_' + str(ind) + '_w1.png')
    plt.show()

ind = 4
q = "env_a == '" + task_pool[ind] + "' and episodes >= 950"
A = test_results.query(q).drop(['env_a', 'episodes'], axis=1).groupby('env_b')['rewards'].median()
B = w_dists.query("env_a == '" + task_pool[ind] + "'").groupby('env_b')['dist'].median()
plt.scatter(B, A)
plt.xlabel("W1 dist")
plt.ylabel("Rewards")
plt.show()
pearsonr(A, B)

As = []
Bs = []
for ind in range(5):
    q = "env_a == '" + task_pool[ind] + "' and episodes >= 950"
    A = test_results.query(q).drop(['env_a', 'episodes'], axis=1).groupby('env_b')['rewards'].median()
    B = w_dists.query("env_a == '" + task_pool[ind] + "'").groupby('env_b')['dist'].median()
    As.append(A)
    Bs.append(B)
plt.scatter(Bs, As)
plt.xlabel('Wasserstein distance')
plt.ylabel('Episode reward')
plt.tight_layout()
plt.savefig('w1_vs_performance_tasks.png')
plt.show()

# scatterplot of mean w_dist for pair against all returns
x_axis = w_dists.groupby(['env_a', 'env_b'])['dist'].mean().to_frame()
plotdata = x_axis.merge(test_results.query('episodes >= 975'), on=['env_a', 'env_b'])

test_results.query("env_a=='reach-v2' and env_b=='reach-v2' and episodes==975")
w_dists.query("env_a=='reach-v2' and env_b=='reach-v2'")

from itertools import product
scatterdata = pd.DataFrame()
for env_a, env_b in list(product(task_pool, task_pool)):
    tmp_w1 = w_dists.query("env_a=='" + env_a + "' and env_b =='" + env_b + "'")['dist'].values
    tmp_rewards = test_results.query("episodes==975 and env_a=='" + env_a + "' and env_b =='" + env_b + "'")['rewards'].values
    tmp_pd = pd.DataFrame({'env_a': [env_a] * tmp_w1.shape[0],
                           'env_b': [env_b] * tmp_w1.shape[0],
                           'w1': tmp_w1,
                           'rewards': tmp_rewards})
    scatterdata = pd.concat((scatterdata, tmp_pd))

plt.scatter(scatterdata.w1, scatterdata.rewards)
plt.show()

pearsonr(scatterdata.w1, scatterdata.rewards)

# TODO:
#   Train an agent until succeeds at reach-v2
#   Benchmark it
#   Benchmark it on handle-press-v2
#   Visually display the two behaviours
#   Investigate w1 distance between reach-v2 and handle-press-v2
#   split w1 distance by reward and by state and see if those results are more informative about returns drop
# TODO FOR LEARNING:
#   investigate wasserstein distances for policy selection
#   multiple worldmodels, one per policy, then select policy based on which worldmodel is most accurate for recent history
# TODO:
#   manually train an agent until success in a completely stationary task
#   test agent against all random variants of said task
#   measure w1 dist between all random variants of said task