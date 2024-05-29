import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool, task_pool_10

"""
Using task_transition_sampling run to plot precision vs sample sizes
"""

results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'task_transition_sampling/')
all_filenames = glob.glob(results_path + "results_[0-9]*.csv")
all_filenames += glob.glob(results_path + "results_large_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    results = pd.concat((results, tmp))
results = results.reset_index()

results = results[results.env_a.isin(task_pool_10)]
results = results[results.env_b.isin(task_pool_10)]

queries = []
queries.append('n_states==5 and n_transitions==1')
queries.append('n_states==5 and n_transitions==5')
queries.append('n_states==5 and n_transitions==10')
queries.append('n_states==15 and n_transitions==1')
queries.append('n_states==15 and n_transitions==5')
queries.append('n_states==15 and n_transitions==10')
queries.append('n_states==25 and n_transitions==1')
queries.append('n_states==25 and n_transitions==5')
queries.append('n_states==25 and n_transitions==10')
queries.append('n_states==150 and n_transitions==1')
queries.append('n_states==150 and n_transitions==5')
queries.append('n_states==150 and n_transitions==10')

tmp = results[['n_states', 'n_transitions']].drop_duplicates()

sample_sizes = tmp.n_states * (2 * tmp.n_transitions).values

sample_sizes = ['5\n1', '5\n5', '5\n10', '15\n1', '15\n5', '15\n10', '25\n1', '25\n5', '25\n10', '150\n1', '150\n5', '150\n10']

fig, ax = plt.subplots(figsize=(9/1.4, 6/1.4))
for ind, q in enumerate(queries):
    plotdata = results.query(q)[['n_states', 'n_transitions', 'env_a', 'env_b', 'dist']]
    plotdata2 = plotdata.groupby(['env_a', 'env_b']).dist.std()
    # plt.violinplot(plotdata2, positions=[ind], showmedians=True, showextrema=False, widths=0.5)
    plt.boxplot(plotdata2, positions=[ind], widths=0.5, showcaps=True)
plt.xlabel('Sample size')
plt.title("Standard deviation of W1 estimate, aggregated over 10 MDPs")
plt.ylabel('Standard deviation')
ax.set_xticks(ticks=range(12), labels=sample_sizes)
plt.xlabel("States sampled\nTransitions sampled per MDP")
plt.tight_layout()
plt.savefig('std_by_sample_size.png', dpi=200)
plt.show()

queries = []
queries.append('n_states==5 and n_transitions==1')
queries.append('n_states==150 and n_transitions==10')

# tasks_to_plot = [task_pool_10[0]]
tasks_to_plot = ['reach-v2']
for base_ind, task in enumerate(tasks_to_plot):
    for q in [queries[0]]:
        plt.rcParams.update({'figure.autolayout': True})
        plotdata = results[results.env_a == task].query(q).groupby('env_b')
        if len(plotdata) != 0:
            ax = plotdata.boxplot(column=['dist'], subplots=False, figsize=(9/1.4, 6/1.4))
            env_b_list = results.env_b.unique().tolist()
            ax.set_xticks(ticks=range(1, len(env_b_list) + 1), labels=env_b_list)
            ax.get_xticklabels()[env_b_list.index(task)].set_color('red')
            plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
            plt.ylim(0, 7.0)
            plt.ylabel("Approx. W1 distance")
            plt.title("W1 distance approximations, smallest samples")
            plt.tight_layout()
            plt.savefig('mdp_distances_smallest_sample_reach.png', dpi=200)
            plt.show()



results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'sample_size_vs_w1_precision/')
all_filenames = glob.glob(results_path + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    results = pd.concat((results, tmp))
results = results.reset_index()

results.columns

results[['n_states', 'n_transitions']].drop_duplicates()
results.groupby(['env_a', 'env_b', 'n_states', 'n_transitions']).count()

queries = []
queries.append('n_states==50 and n_transitions==1')

tasks_to_plot = task_pool_10
for base_ind, task in enumerate(tasks_to_plot):
    for q in queries:
        plt.rcParams.update({'figure.autolayout': True})
        plotdata = results[results.env_a == task].query(q).groupby('env_b')
        if len(plotdata) != 0:
            ax = plotdata.boxplot(column=['dist'], subplots=False, figsize=(9/1.4, 6/1.4))
            env_b_list = results.env_b.unique().tolist()
            ax.set_xticks(ticks=range(1, len(env_b_list) + 1), labels=env_b_list)
            ax.get_xticklabels()[env_b_list.index(task)].set_color('red')
            plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
            plt.ylim(0, 10.0)
            plt.ylabel("Approx. W1 distance")
            plt.title("W1 distance approximations, smallest samples")
            plt.tight_layout()
            # plt.savefig('mdp_distances_smallest_sample_reach.png', dpi=200)
            plt.show()


