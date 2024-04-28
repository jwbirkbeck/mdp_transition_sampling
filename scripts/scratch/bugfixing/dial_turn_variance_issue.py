import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, bounded_state_space

task = task_pool[4]
env_a = MetaWorldWrapper(task_name=task, render_mode=None)
env_b = MetaWorldWrapper(task_name=task, render_mode=None)

env_a.reset()
env_b.reset()

action_bounds = env_a.action_space

dists = []
sampler = MDPDifferenceSampler(environment_a=env_a, environment_b=env_b,
                               state_space=bounded_state_space, action_space=action_bounds)

for _ in range(10):
    dists.append(sampler.get_difference(n_states=25, n_transitions=5))
print("done")

plt.boxplot(dists)
plt.ylim(0, 6)
plt.show()
# Run the MDPDifferenceSampler and see if the distances are fixed:

