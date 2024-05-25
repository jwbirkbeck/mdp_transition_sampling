import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
What am I trying to show: 
    That the policy selection approach can avoid forgetting better than a SAC agent
    
    This would be shown as increased rewards over a long period 
    
Load medoid results
plot sac average vs ps3, ps6 averages
"""

results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'medoid_policy_selection/')
all_filenames = glob.glob(results_path + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run_index'] = re.search('[0-9]+', filename).group(0)
    results = pd.concat((results, tmp))
results = results.rename(columns={'p10': 'ps10'})

results2 = pd.melt(results, id_vars=['episode', 'run_index'], value_vars=['sac', 'ps3', 'ps6', 'ps10', 'ps3r', 'ps6r'], var_name='method', value_name='reward')
results3 = results2.groupby(['episode', 'method']).mean('reward')
for m in results2['method'].unique():
    plt.plot(results3.query("method=='" + m + "'")['reward'].values)
plt.show()

plotdata = results2.loc[(results2.run_index=='2') & (results2.method=='sac')].filter(items=['episode', 'reward'])
plt.plot(plotdata.episode, plotdata.reward)
plt.title("SAC episode rewards, run 2 of 20")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.tight_layout()
plt.savefig('training_example.png')
plt.show()

sums = results2.groupby(['run_index', 'method'])['reward'].sum()
sums = sums.reset_index()
max_sum = max(sums.reward)

for _ in range(1):
    plot_order = ['sac', 'ps3r', 'ps6r', 'ps3', 'ps6', 'ps10']
    plot_tick_labels = ['SAC', '3 pol: random', '6 pol: random', "3 pol: K-m", "6 pol: K-m", "10 pol: K-m"]
    fig, ax = plt.subplots()
    plt.hlines(y=sums[sums.method == 'sac'].reward.median() / max_sum, xmin=0, xmax=5, linestyles='dashed', colors='lightgrey')
    for ind, m in enumerate(plot_order):
        plotdata = sums[sums.method == m].reward
        # plt.violinplot(plotdata, positions=[ind], showmeans=True, widths=0.5, bw_method=0.2)
        plt.violinplot(plotdata, positions=[ind], showmedians=True, widths=0.5, bw_method=0.2)
    ax.set_xticks(ticks=range(len(plot_order)), labels=plot_tick_labels)
    # ax.set_yticks(ticks=[i / 10 for i in range(0, 11)], labels=[i / 10 for i in range(0, 11)])
    plt.xticks(rotation=-20, ha='left', rotation_mode='anchor')
    plt.xlabel('Method')
    plt.ylabel('Normalized reward')
    plt.title('Normalized episodic training rewards, 20 runs')
    plt.tight_layout()
    plt.savefig('normalised_median_rewards.png')
    plt.show()

means = results2.groupby(['run_index', 'method'])['reward'].mean()
means = means.reset_index()
max_mean = max(means.reward)

for _ in range(1):
    plot_order = ['sac', 'ps10', 'ps3r', 'ps6r', 'ps3', 'ps6']
    plot_tick_labels = ['SAC', '10 SACs', '3 pol: random', '6 pol: random', "3 pol: K-m", "6 pol: K-m"]
    fig, ax = plt.subplots()
    plt.hlines(y=means[means.method == 'sac'].reward.median(), xmin=0, xmax=5, linestyles='dashed', colors='lightgrey')
    for ind, m in enumerate(plot_order):
        plotdata = means[means.method == m].reward
        plt.violinplot(plotdata, positions=[ind], showmedians=True, widths=0.5, bw_method=0.2)
    # plot a horizontal line for sac performance:
    ax.set_xticks(ticks=range(len(plot_order)), labels=plot_tick_labels)
    plt.xticks(rotation=-20, ha='left', rotation_mode='anchor')
    plt.xlabel('Method')
    plt.ylabel('Episode reward')
    plt.title('Median episodic rewards per method, 20 runs, w/ aggregate median ')
    plt.tight_layout()
    plt.savefig('median_rewards.png')
    plt.show()
