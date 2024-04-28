import pandas as pd
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, bounded_state_space

exp_task_pool = [task_pool[i] for i in range(len(task_pool)) if i not in [12, 16]]
exp_task_pool = [exp_task_pool[i] for i in range(len(exp_task_pool)) if i in [6, 7, 8, 9, 10, 11, 13, 14]]

base_ind = 2
reps = 10
results = pd.DataFrame()
# exp_task_pool = task_pool
for rep in range(reps):
    for ind in range(len(exp_task_pool)):
        percent_prog = round(100 * (((rep * len(exp_task_pool)) + ind) / (reps * len(exp_task_pool))), 1)
        print(str(percent_prog) + "%")
        env_a = MetaWorldWrapper(task_name=exp_task_pool[base_ind], render_mode=None)
        env_b = MetaWorldWrapper(task_name=exp_task_pool[ind], render_mode=None)

        env_a.reset()
        env_b.reset()

        state_bounds = bounded_state_space
        action_bounds = env_a.action_space

        sampler = MDPDifferenceSampler(environment_a=env_a, environment_b=env_b,
                                       state_space=state_bounds, action_space=action_bounds)
        pd_row = pd.DataFrame({'env_a': [exp_task_pool[base_ind]],
                               'env_b': [exp_task_pool[ind]],
                               'dist': [sampler.get_difference(n_states=25, n_transitions=10)]})
        results = pd.concat((results, pd_row))
print("done")

plotdata = results.groupby('env_b')
ax = plotdata.boxplot(column=['dist'], subplots=False, figsize=(9/1.2, 6/1.2))
ax.set_xticks(ticks=range(1, len(exp_task_pool) + 1), labels=exp_task_pool)
ax.get_xticklabels()[base_ind].set_color('red')
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.ylabel("W1 distance")
plt.title("Wasserstein distance between tasks")
plt.ylim(0, 6)
plt.tight_layout()
plt.show()
