import glob
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool, task_pool_10

"""
Using task_transition_sampling run to plot precision vs sample sizes
"""

results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    results = pd.concat((results, tmp))
results.reset_index()

results = results[results.env_a.isin(task_pool_10)]
results = results[results.env_b.isin(task_pool_10)]

query1 = 'n_states==5 and n_transitions==1'
query2 = 'n_states==15 and n_transitions==1'
query3 = 'n_states==25 and n_transitions==1'
query4 = 'n_states==5 and n_transitions==5'
query5 = 'n_states==15 and n_transitions==5'
query6 = 'n_states==25 and n_transitions==5'
query7 = 'n_states==5 and n_transitions==10'
query8 = 'n_states==15 and n_transitions==10'
query9 = 'n_states==25 and n_transitions==10'
queries = [query1, query2, query3, query4, query5, query6, query7, query8, query9]



tasks_to_plot = [task_pool_10[0]]
for base_ind, task in enumerate(tasks_to_plot):
    for q in queries:
        plt.rcParams.update({'figure.autolayout': True})
        plotdata = results[results.env_a == task].query(q).groupby('env_b')
        ax = plotdata.boxplot(column=['dist'], subplots=False, figsize=(9/1.2, 6/1.2))
        env_b_list = results.env_b.unique().tolist()
        ax.set_xticks(ticks=range(1, len(env_b_list) + 1), labels=env_b_list)
        ax.get_xticklabels()[base_ind].set_color('red')
        plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
        plt.ylabel("W1 distance")
        plt.title("Wasserstein distance between tasks")
        plt.tight_layout()
        plt.show()
