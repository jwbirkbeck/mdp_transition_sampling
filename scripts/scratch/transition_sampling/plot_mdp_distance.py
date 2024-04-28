import matplotlib.pyplot as plt
import pandas as pd
import glob
from src.utils.consts import task_pool

results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    results = pd.concat((results, tmp))

results.reset_index

query1 = 'n_states==5 and n_transitions==1'
query2 = 'n_states==15 and n_transitions==1'
query3 = 'n_states==25 and n_transitions==1'
query4 = 'n_states==5 and n_transitions==5'
query5 = 'n_states==15 and n_transitions==5'
query6 = 'n_states==25 and n_transitions==5'
query7 = 'n_states==5 and n_transitions==10'
query8 = 'n_states==15 and n_transitions==10'
query9 = 'n_states==25 and n_transitions==10'

# exp_task_pool = [task_pool[i] for i in range(len(task_pool)) if i not in [12, 16]]
# trained_agents_pool = [exp_task_pool[i] for i in range(len(exp_task_pool)) if i in [6, 7, 8, 9, 10, 11, 13, 14]]

task_selection = ['handle-press-side-v2', 'handle-press-v2', 'plate-slide-back-v2', 'reach-v2', 'reach-wall-v2']
# trained_agents_pool = task_pool
trained_agents_pool = task_selection

results2 = results[results.env_a.isin(trained_agents_pool)]
results2 = results2[results2.env_b.isin(trained_agents_pool)]

for base_ind in range(len(trained_agents_pool)):
    base_env = trained_agents_pool[base_ind]
    query_pool = [[query1, query2, query3, query4, query5, query6, query7, query8, query9][-1]]
    for q in query_pool:
        plt.rcParams.update({'figure.autolayout': True})
        plotdata = results2[results2.env_a == base_env].query(q).groupby('env_b')
        ax = plotdata.boxplot(column=['dist'], subplots=False, figsize=(9/1.2, 6/1.2))
        ax.set_xticks(ticks=range(1, len(trained_agents_pool) + 1), labels=trained_agents_pool)
        ax.get_xticklabels()[base_ind].set_color('red')
        plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
        plt.ylabel("W1 distance")
        plt.title("Wasserstein distance between tasks")
        plt.ylim(0, 6)
        plt.tight_layout()
        # plt.savefig('transition_sampling_' + base_env + '_25_10.png')
        plt.show()

