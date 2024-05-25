import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "/opt/project/results/"
m_results_path = os.path.join(results_dir, 'medoid_policy_selection/')
all_filenames = glob.glob(m_results_path + "results_[0-9]*.csv")
m_results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run_index'] = re.search('[0-9]+', filename).group(0)
    m_results = pd.concat((m_results, tmp))
m_results = m_results.rename(columns={'p10': 'ps10'})

b_results_path = os.path.join(results_dir, 'bandit_policy_selection/')
all_filenames = glob.glob(b_results_path + "results_[0-9]*.csv")
b_results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run_index'] = re.search('[0-9]+', filename).group(0)
    b_results = pd.concat((b_results, tmp))

m_results2 = pd.melt(m_results, id_vars=['episode', 'run_index'], value_vars=['sac', 'ps3', 'ps6', 'ps10', 'ps3r', 'ps6r'], var_name='method', value_name='reward')
b_results2 = pd.melt(b_results, id_vars=['episode', 'run_index'], value_vars=['bps3', 'bps6', 'bps10'], var_name='method', value_name='reward')
results2 = pd.concat((m_results2, b_results2), axis=0)
results3 = results2.groupby(['episode', 'method']).mean('reward')


means = results2.groupby(['run_index', 'method'])['reward'].mean()
means = means.reset_index()
max_mean = max(means.reward)

# Two plots:



for _ in range(1):
    plot_order = ['sac', 'ps10', 'bps10', 'ps3r', 'bps3', 'ps3', 'ps6r', 'bps6', 'ps6']
    plot_tick_labels = plot_order
    fig, ax = plt.subplots()
    # plt.hlines(y=means[means.method == 'sac'].reward.median() / max_sum, xmin=0, xmax=5, linestyles='dashed', colors='lightgrey')
    for ind, m in enumerate(plot_order):
        plotdata = means[means.method == m].reward
        plt.violinplot(plotdata, positions=[ind], showmedians=True, widths=0.5, bw_method=0.2)
    ax.set_xticks(ticks=range(len(plot_order)), labels=plot_tick_labels)
    # ax.set_yticks(ticks=[i / 10 for i in range(0, 11)], labels=[i / 10 for i in range(0, 11)])
    # plt.xticks(rotation=-20, ha='left', rotation_mode='anchor')
    # plt.xlabel('Method')
    # plt.ylabel('Normalized reward')
    # plt.title('Normalized episodic training rewards, 20 runs')
    plt.tight_layout()
    # plt.savefig('normalised_median_rewards.png')
    plt.show()
