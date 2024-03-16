import sys
import os
import pandas as pd
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, bounded_state_space


results_path = os.path.expanduser('~/mdp_transition_sampling/results/task_transition_sampling')
os.makedirs(results_path) if not os.path.isdir(results_path) else None

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
base_index = int(sys.argv[1])

# Temporary exclusion of tasks requiring additional reward calculation mechanisms:
exp_task_pool = [task_pool[ind] for ind in range(len(task_pool)) if ind not in (4, 10, 11, 12, 16, 17)]
base_task = exp_task_pool[base_index]
reps = 5

sampling_params = [{'n_states': 5, 'n_transitions': 1},
                   {'n_states': 5, 'n_transitions': 5},
                   {'n_states': 5, 'n_transitions': 10},
                   {'n_states': 15, 'n_transitions': 1},
                   {'n_states': 15, 'n_transitions': 5},
                   {'n_states': 15, 'n_transitions': 10},
                   {'n_states': 25, 'n_transitions': 1},
                   {'n_states': 25, 'n_transitions': 5},
                   {'n_states': 25, 'n_transitions': 10}]

results = pd.DataFrame()
for param_set in sampling_params:
    for rep in range(reps):
        for ind in range(len(exp_task_pool)):
            env_a = MetaWorldWrapper(task_name=base_task, render_mode=None)
            env_b = MetaWorldWrapper(task_name=exp_task_pool[ind], render_mode=None)

            env_a.reset()
            env_b.reset()

            state_bounds = bounded_state_space
            action_bounds = env_a.action_space

            sampler = MDPDifferenceSampler(environment_a=env_a, environment_b=env_b,
                                           state_space=state_bounds, action_space=action_bounds)
            dist = sampler.get_difference(n_states=param_set['n_states'], n_transitions=param_set['n_transitions'])
            pd_row = pd.DataFrame({'env_a': [base_task],
                                   'env_b': [exp_task_pool[ind]],
                                   'dist': [dist],
                                   'n_states': param_set['n_states'],
                                   'n_transitions': param_set['n_transitions']})
            results = pd.concat((results, pd_row))
            # Save after every update for loss avoidance:
            csv_name = "results_" + str(base_index) + ".csv"
            results.to_csv(os.path.join(results_path, csv_name), index=False)
            env_a.new_task()
            env_b.new_task()
