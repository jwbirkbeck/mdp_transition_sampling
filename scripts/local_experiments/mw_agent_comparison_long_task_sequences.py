import pandas as pd
import numpy as np
import itertools
import pickle
import os
from src.metaworld.nonstationarity_distribution import TaskNSDist
from src.utils.consts import task_pool_10
from src.utils.filepaths import results_path_local

p_task_change = 0.1
max_episodes = 50000 # will not be reached

run_config_list = list(itertools.product(*[list(range(10)), list(range(5)), list(range(12))]))
run_configs = pd.DataFrame()
for run_config in run_config_list:
    task_sequence_index = run_config[0]
    agent_index = run_config[1]
    rep_index = run_config[2]
    config_row = pd.DataFrame({'agent_index': [agent_index],
                               'task_sequence_index': [task_sequence_index],
                               'rep_index': [rep_index]})
    run_configs = pd.concat((run_configs, config_row), ignore_index=True)

with open(os.path.join(results_path_local, 'agent_comparison_long', 'run_configs.pkl'), 'wb') as file:
    pickle.dump(run_configs, file)

task_sequences = {'seed': [], 'sequence': []  }
for seed in range(10):
    rng = np.random.default_rng(seed)
    task_dist = TaskNSDist(rng=rng, current_task=task_pool_10[0], task_pool=task_pool_10)
    task_dist.set_prob_task_change(probability=p_task_change)
    task_dist.generate_sequence(sequence_length=max_episodes)
    task_sequence = list(task_dist.seq)
    task_sequences['seed'].append(seed)
    task_sequences['sequence'].append(task_sequence)

with open(os.path.join(results_path_local, 'agent_comparison_long', 'task_sequences.pkl'), 'wb') as file:
    pickle.dump(task_sequences, file)
