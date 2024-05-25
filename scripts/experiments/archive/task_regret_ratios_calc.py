import sys, os
import pandas as pd
from stable_baselines3 import SAC
from src.utils.consts import task_pool
from src.metaworld.wrapper import MetaWorldWrapper

results_path = os.path.expanduser('~/mdp_transition_sampling/results/task_transition_sampling')
os.makedirs(results_path) if not os.path.isdir(results_path) else None

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
base_index = int(sys.argv[1])

base_task = task_pool[base_index]
trials = 100


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
    return returns_a_in_b, returns_opt_b, returns_min_b


results = pd.DataFrame()
for task in task_pool:
    returns_a_in_b, returns_opt_b, returns_min_b = (
        calc_regret_ratio(task_a=base_task, task_b=task, n_trials=100, discount=1.0))
    pd_row = pd.DataFrame({'env_a': base_task,
                           'env_b': task,
                           'returns_a_in_b': returns_a_in_b,
                           'returns_opt_b': returns_opt_b,
                           'returns_min_b': returns_min_b})
    results = pd.concat((results, pd_row))
