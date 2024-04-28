import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src.utils.consts import task_pool
from src.metaworld.wrapper import MetaWorldWrapper


def calc_regret_ratio(task_a, task_b, n_trials, discount=0.99):
    agent_dir = "/opt/project/results/task_regret_ratios/"
    env_b = MetaWorldWrapper(task_name=task_b)

    opt_a = SAC.load(agent_dir + "opt___" + task_a + ".sb3")
    opt_b = SAC.load(agent_dir + "opt___" + task_b + ".sb3")
    min_b = SAC.load(agent_dir + "min___" + task_b + ".sb3")

    returns_a_in_b = []
    returns_opt_b = []
    returns_min_b = []

    opt_a.set_env(env=env_b)
    for _ in range(n_trials):
        env_b.change_task()
        obs, info = env_b.reset()
        returns = 0.0
        steps = -1.0
        terminated = truncated = False
        while not terminated or truncated:
            action, _states = opt_a.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_b.step(action)
            steps += 1
            returns += (discount ** steps) * reward
            if terminated or truncated:
                returns_a_in_b.append(returns)
    print("agent A's performance in environment B recorded")

    opt_b.set_env(env=env_b)
    for _ in range(n_trials):
        env_b.change_task()
        obs, info = env_b.reset()
        returns = 0.0
        steps = -1.0
        terminated = truncated = False
        while not terminated or truncated:
            action, _states = opt_b.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_b.step(action)
            steps += 1
            returns += (discount ** steps) * reward
            if terminated or truncated:
                returns_opt_b.append(returns)
    print("optimal B's performance recorded")

    min_b.set_env(env=env_b)
    for _ in range(n_trials):
        env_b.change_task()
        obs, info = env_b.reset()
        returns = 0.0
        steps = -1.0
        terminated = truncated = False
        while not terminated or truncated:
            action, _states = min_b.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_b.step(action)
            steps += 1
            returns += (discount ** steps) * reward
            if terminated or truncated:
                returns_min_b.append(returns)
    print("minimal B's performance recorded")
    print("done")
    return returns_a_in_b, returns_opt_b, returns_min_b



exp_task_pool = [task_pool[i] for i in range(len(task_pool)) if i not in [12, 16]]
trained_agents_pool = [exp_task_pool[i] for i in range(len(exp_task_pool)) if i in [6, 7, 8, 9, 10, 11, 13, 14]]

n_trials = 5
results = pd.DataFrame()
for base_task in trained_agents_pool:
    for task in trained_agents_pool:
        rew_a_in_b, rew_opt_b, rew_min_b = (
            calc_regret_ratio(task_a=base_task, task_b=task, n_trials=n_trials, discount=1.0))
        pd_row = pd.DataFrame({'env_a': [base_task] * n_trials,
                               'env_b': [task] * n_trials,
                               'rew_a_in_b': rew_a_in_b,
                               'rew_opt_b': rew_opt_b,
                               'rew_min_b': rew_min_b})
        results = pd.concat((results, pd_row))
print('done')

for base_ind in range(len(trained_agents_pool)):
    # base_ind = 0
    base_task = trained_agents_pool[base_ind]
    plotdata = pd.DataFrame()
    for comp_task in trained_agents_pool:
        filtered_results = results.query("env_a == '" + base_task + "' and env_b == '" + comp_task + "'")
        rew_a_in_b = filtered_results.rew_a_in_b
        rew_opt_b = filtered_results.rew_opt_b
        rew_min_b = filtered_results.rew_min_b
        pd_row = pd.DataFrame({'env_a': [base_task] * n_trials,
                               'env_b': [comp_task] * n_trials,
                               'ratio': (rew_opt_b - rew_a_in_b) / (rew_opt_b - rew_min_b)})
        # max_a_in_b = filtered_results.rew_a_in_b.max()
        # max_b = filtered_results.rew_opt_b.max()
        # min_b = filtered_results.rew_min_b.min()
        # pd_row = pd.DataFrame({'env_a': [base_task],
        #                        'env_b': [comp_task],
        #                        'ratio': (max_b - max_a_in_b) / (max_b - min_b)})
        plotdata = pd.concat((plotdata, pd_row))

    plt.rcParams.update({'figure.autolayout': True})
    ax = plotdata.groupby('env_b').boxplot(column=['ratio'], subplots=False, figsize=(9 / 1.2, 6 / 1.2))
    ax.set_xticks(ticks=range(1, len(trained_agents_pool) + 1), labels=trained_agents_pool)
    ax.get_xticklabels()[base_ind].set_color('red')
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    plt.ylabel("Regret ratio (higher is more regret)")
    plt.title("Regret ratios between tasks")
    plt.ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.savefig('regret_ratios_' + base_task + ".png")
    plt.show()

