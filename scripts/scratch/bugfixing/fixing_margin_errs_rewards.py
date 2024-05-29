import pandas as pd
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, task_pool_10, bounded_state_space

# new_task_pool = [task_pool[ind] for ind in range(len(task_pool)) if ind not in (4, 10, 11, 12, 16)]
# err_task_pool = [task_pool[ind] for ind in range(len(task_pool)) if ind in (4, 10, 11, 12, 16)]
# exp_task_pool = err_task_pool
# task = err_task_pool[0]

reps = 10
results = pd.DataFrame()
task = 'handle-press-v2'
exp_task_pool = task_pool_10
for rep in range(reps):
    for ind in range(len(exp_task_pool)):
        percent_prog = round(100 * (((rep * len(exp_task_pool)) + ind) / (reps * len(exp_task_pool))), 1)
        print(str(percent_prog) + "%")
        env_a = MetaWorldWrapper(task_name=task, render_mode=None)
        env_b = MetaWorldWrapper(task_name=exp_task_pool[ind], render_mode=None)

        env_a.reset()
        env_b.reset()

        state_bounds = bounded_state_space
        action_bounds = env_a.action_space

        sampler = MDPDifferenceSampler(environment_a=env_a, environment_b=env_b,
                                       state_space=state_bounds, action_space=action_bounds)
        dist = sampler.get_difference(n_states=25, n_transitions=5)
        pd_row = pd.DataFrame({'env_a': [task],
                               'env_b': [exp_task_pool[ind]],
                               'dist': [dist]})
        results = pd.concat((results, pd_row))
print("done")

import matplotlib.pyplot as plt
results.groupby('env_b').boxplot(column=['dist'], subplots=False)
plt.show()

