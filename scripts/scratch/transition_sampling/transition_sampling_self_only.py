import pandas as pd
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, bounded_state_space

reps = 10
results = pd.DataFrame()
exp_task_pool = task_pool
for rep in range(reps):
    for ind in range(len(exp_task_pool)):
        percent_prog = round(100 * (((rep * len(exp_task_pool)) + ind) / (reps * len(exp_task_pool))), 1)
        print(str(percent_prog) + "%")
        env_a = MetaWorldWrapper(task_name=exp_task_pool[ind], render_mode=None)
        env_b = MetaWorldWrapper(task_name=exp_task_pool[ind], render_mode=None)

        env_a.reset()
        env_b.reset()

        state_bounds = bounded_state_space
        action_bounds = env_a.action_space

        sampler = MDPDifferenceSampler(env_a=env_a, env_b=env_b,
                                       state_space=state_bounds, action_space=action_bounds)
        pd_row = pd.DataFrame({'env_a': [exp_task_pool[ind]],
                               'env_b': [exp_task_pool[ind]],
                               'dist': [sampler.get_difference(n_states=25, n_transitions=5)]})
        results = pd.concat((results, pd_row))
print("done")

results_self_self = results

import pickle
with open("/opt/project/self_self.pickle", "wb") as file:
    pickle.dump(results_self_self, file)

with open("/opt/project/self_self.pickle", "rb") as file:
    results_self_self = pickle.load(file)

plotdata = results_self_self.groupby('env_b')
ax = plotdata.boxplot(column=['dist'], subplots=False, figsize=(9 / 1.2, 6 / 1.2))
ax.set_xticks(ticks=range(1, len(task_pool) + 1), labels=task_pool)
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.ylabel("W1 distance")
plt.title("Wasserstein distance between tasks")
plt.ylim(0, 6)
plt.tight_layout()
plt.show()

# I should test:
# 0 vs 1, 2, 14, 15, 16
# specific task choice vs specific task choice - maybe this is causing the issues with dial-turn and soccer?
