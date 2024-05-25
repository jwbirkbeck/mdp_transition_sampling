import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.policy_selector.simple_policy_selector import SimplePolicySelector
from src.metaworld.wrapper import MetaWorldWrapper

task_pool = ['coffee-button-v2', 'dial-turn-v2', 'door-unlock-v2',
             'handle-press-side-v2', 'handle-press-v2',
             'plate-slide-back-v2', 'plate-slide-v2', 'push-back-v2',
             'reach-v2', 'reach-wall-v2']

device = torch.device('cpu')
env = MetaWorldWrapper(task_name=task_pool[0])
policy_selector = SimplePolicySelector(env=env, method='bandit', device=device, task_names=task_pool, n_policies=3)

p_task_change = 0.5
rewards = []
for _ in range(250):
    print(_)
    if np.random.uniform() < p_task_change:
        new_task_name = np.random.choice(task_pool)
        env.change_task(task_name=new_task_name)
    ep_rew, _ = policy_selector.play_episode()
    rewards.append(ep_rew)
print("done")

plt.plot(rewards)
plt.show()

policy_selector.task_policy_mapping