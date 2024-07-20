import sys, os, time
import torch
import numpy as np
import pandas as pd
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool_10, bounded_state_space

results_dir = "~/mdp_transition_sampling/results/"
# results_dir = "/opt/project/results/"
results_path = os.path.join(results_dir, 'sample_size_vs_w1_precision/')

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device('cpu')
                                                            # total sample sizes:
sampling_params = [{'n_states': 5,  'n_transitions': 1 },   # 10
                   {'n_states': 15, 'n_transitions': 1 },   # 30
                   {'n_states': 15, 'n_transitions': 5 },   # 150
                   {'n_states': 25, 'n_transitions': 1 },   # 50
                   {'n_states': 25, 'n_transitions': 5 },   # 250
                   {'n_states': 25, 'n_transitions': 10},   # 500
                   {'n_states': 50, 'n_transitions': 1 },   # 100
                   {'n_states': 50, 'n_transitions': 5 },   # 500
                   {'n_states': 50, 'n_transitions': 10}]   # 1000

results = pd.DataFrame()
for params in sampling_params:
    for task_a in task_pool_10:
        for task_b in task_pool_10:
                env_a = MetaWorldWrapper(task_name=task_a, render_mode=None)
                env_b = MetaWorldWrapper(task_name=task_b, render_mode=None)
                env_a.change_task(task_name=task_a, task_number=0)
                env_b.change_task(task_name=task_b, task_number=0)
                env_a.reset()
                env_b.reset()

                state_bounds = bounded_state_space
                action_bounds = env_a.action_space

                sampler = MDPDifferenceSampler(env_a=env_a,
                                               env_b=env_b,
                                               state_space=state_bounds,
                                               action_space=action_bounds)
                start_time = time.time()
                dist = sampler.get_difference(n_states=params['n_states'],
                                              n_transitions=params['n_transitions'])
                end_time = time.time()
                runtime = round(end_time - start_time, 4)
                pd_row = pd.DataFrame({'env_a': [task_a],
                                       'env_b': [task_b],
                                       'dist': [dist],
                                       'n_states': params['n_states'],
                                       'n_transitions': params['n_transitions'],
                                       'runtime': [runtime]})
                results = pd.concat((results, pd_row))
                csv_name = "results_" + str(run_index) + ".csv"
                results.to_csv(os.path.join(results_path, csv_name), index=False)
