import pandas as pd
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, bounded_state_space

results = pd.DataFrame()
for task in task_pool:
    env = MetaWorldWrapper(task_name=task, render_mode=None)
    env.reset()
    rewards = []
    manual_rewards = []
    errors = []
    for _ in range(2500):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        manual_reward = env.compute_reward_wrap(states=[state], actions=[action])[0]
        rewards.append(reward)
        manual_rewards.append(manual_reward)
        errors.append(abs(reward - manual_reward))
        if done or truncated:
            env.reset()
    pd_row = pd.DataFrame({'task': [task] * len(errors), 'errors': errors})
    results = pd.concat((results, pd_row))
print("done")

ax = results.groupby('task').boxplot(column=['errors'], subplots=False, figsize=(9/1.2, 6/1.2))
ax.set_xticks(ticks=range(1, len(task_pool)+1), labels=task_pool)
plt.rcParams.update({'figure.autolayout': True})
plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
plt.ylabel("Absolute error")
plt.title("Reward Reconstruction Error by Task")
plt.tight_layout()
plt.savefig('/opt/project/plots/transition_sampling/reward_reconstruction_error.png')
plt.show()
