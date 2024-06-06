

import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool, task_pool_10

"""
Using task_transition_sampling run to plot precision vs sample sizes
"""

results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'sample_size_vs_w1_precision/')
all_filenames = glob.glob(results_path + "results_[0-9]*.csv")
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

pd_sample_sizes = results[['n_states', 'n_transitions']].drop_duplicates()

queries = []
xaxis_labels = []
for ns, nt in zip(pd_sample_sizes.n_states, pd_sample_sizes.n_transitions):
    queries.append(f'n_states=={ns} and n_transitions=={nt}')
    xaxis_labels.append(f'{ns}\n{nt}')

fig, ax = plt.subplots(figsize=(9/1.4, 6/1.4))
for ind, q in enumerate(queries):
    plotdata = results.query(q)[['n_states', 'n_transitions', 'env_a', 'env_b', 'dist']]
    plotdata2 = plotdata.groupby(['env_a', 'env_b']).dist.std()
    # plt.violinplot(plotdata2, positions=[ind], showmedians=True, showextrema=False, widths=0.5)
    plt.boxplot(plotdata2, positions=[ind], widths=0.5, showcaps=True)
plt.xlabel('Sample size')
plt.title("Standard deviation of W1 estimate, aggregated over 10 MDPs")
plt.ylabel('Standard deviation')
ax.set_xticks(ticks=range(len(queries)), labels=xaxis_labels)
plt.xlabel("States sampled\nTransitions sampled per MDP")
plt.tight_layout()
# plt.savefig('std_by_sample_size.png', dpi=200)
plt.show()

