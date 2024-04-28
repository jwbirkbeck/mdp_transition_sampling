import copy
import pandas as pd
import numpy as np
import torch
import gymnasium as gym
import time as timer
import os
import pickle
from src.utils.consts import task_pool
from src.metaworld.wrapper import MetaWorldWrapper
from src.lpg_ftw.mjrl.utils.train_agent import train_agent
from src.lpg_ftw.mjrl.policies.gaussian_mlp_lpg_ftw import MLPLPGFTW
from src.lpg_ftw.mjrl.algos.npg_cg_ftw import NPGFTW
from src.lpg_ftw.mjrl.baselines.mlp_baseline import MLPBaseline
from src.lpg_ftw.mjrl.utils.gym_env import GymEnv, EnvSpec
from src.lpg_ftw.mjrl.samplers.base_sampler import do_rollout
import src.lpg_ftw.mjrl.utils.process_samples as process_samples

env_dict = {'button-press-topdown-v2': 'lpg_ftw_wrappers:WrappedButtonPressTopdown',
             'button-press-v2': 'lpg_ftw_wrappers:WrappedButtonPress',
             'button-press-wall-v2': 'lpg_ftw_wrappers:WrappedButtonPressWall',
             'coffee-button-v2': 'lpg_ftw_wrappers:WrappedCoffeeButton',
             'coffee-push-v2': 'lpg_ftw_wrappers:WrappedCoffeePush',
             'dial-turn-v2': 'lpg_ftw_wrappers:WrappedDialTurn',
             'door-close-v2': 'lpg_ftw_wrappers:WrappedDoorClose',
             'door-unlock-v2': 'lpg_ftw_wrappers:WrappedDoorUnlock',
             'handle-press-side-v2': 'lpg_ftw_wrappers:WrappedHandlePressSide',
             'handle-press-v2': 'lpg_ftw_wrappers:WrappedHandlePress',
             'peg-insert-side-v2': 'lpg_ftw_wrappers:WrappedPegInsertSide',
             'plate-slide-back-v2': 'lpg_ftw_wrappers:WrappedPlateSlideBack',
             'plate-slide-v2': 'lpg_ftw_wrappers:WrappedPlateSlide',
             'push-back-v2': 'lpg_ftw_wrappers:WrappedPushBack',
             'reach-v2': 'lpg_ftw_wrappers:WrappedReach',
             'reach-wall-v2': 'lpg_ftw_wrappers:WrappedReachWall',
             'soccer-v2': 'lpg_ftw_wrappers:WrappedSoccer'}


e_unshuffled = {}
for task_id, (env_id, entry_point) in enumerate(env_dict.items()):
    # kwargs = {'obs_type': 'plain'}
    kwargs = {}
    # if env_id == 'reach-v2':
    #     kwargs['tasks'] = None
    # elif env_id == 'push-v1':
    #     kwargs['task_type'] = 'push'
    # elif env_id == 'pick-place-v1':
    #     kwargs['task_type'] = 'pick_place'
    gym.envs.register(
        id=env_id,
        entry_point='src.metaworld.' + entry_point,
        max_episode_steps=None,
        disable_env_checker=True,
        order_enforce=False,
        kwargs=kwargs
    )
    e_unshuffled[task_id] = GymEnv(env_id)


# TODO:
#  re-test train_agent loop works with above steps taken
#  Ensure that errors do not occur in above process once task is changed, as it does in the manual version further below

num_tasks = len(task_pool)
SEED = 50
num_cpu = 16

np.random.seed(SEED)
torch.manual_seed(SEED)

# # # # #
#
# # # # #

e = {}
baseline = {}
task_order = np.random.permutation(num_tasks)
for task_id in range(num_tasks):
    e[task_id] = e_unshuffled[task_order[task_id]]
    baseline[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=1, learn_rate=1e-3, use_gpu=False)

policy = MLPLPGFTW(e[0].spec, hidden_sizes=(32, 32), k=1, max_k=3)
agent = NPGFTW(e, policy, baseline, normalized_step_size=0.01, save_logs=True, new_col_mode='max_k')


job_name = 'deleteme_testing_lpgftw'
niter = 10
gamma = 0.995
gae_lambda = 0.97
num_cpu = 15
sample_mode = 'trajectories'
num_traj = 15

task_id = 1
env_id = task_pool[task_id]
agent.set_task(task_id=task_id)

best_policy = copy.copy(agent.policy)
best_perf = -1e8
train_curve = best_perf * np.ones(niter)
mean_pol_perf = 0.0
for i in range(niter):
    if train_curve[i - 1] > best_perf:
        best_policy = copy.copy(agent.policy)
        best_perf = train_curve[i - 1]
    args = dict(N=num_traj, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu, env_name=env_id)
    stats = agent.train_step(**args)
    train_curve[i] = stats[0]
print("done")

train_curve
train_curve_0 = train_curve
train_curve_1 = train_curve

train_curve_0 = np.append(train_curve_0, train_curve)
train_curve_1 = np.append(train_curve_1, train_curve)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('module://backend_interagg')
plt.plot(train_curve_0, label='task 0')
plt.plot(train_curve_1, label='task 1')
plt.legend()
plt.show()

# # # # # # # # # # #
# # # NOT WORKING WHEN MULTIPLE TASK IDs USED
# # # # # # # # # # #
#
# num_tasks = len(task_pool)
# all_horizon = 500
# SEED = 50
# num_cpu = 16
#
# baselines = {}
# envs = []
# for env_id in task_pool:
#     envs.append(MetaWorldWrapper(task_name=env_id, render_mode=None))
#     envs[-1].horizon = all_horizon
#     envs[-1].env_id = env_id
#
# all_obs_dim = envs[0].observation_space.shape[0]
# all_action_dim = envs[0].action_space.shape[0]
# all_env_spec = EnvSpec(obs_dim=all_obs_dim, act_dim=all_action_dim, horizon=all_horizon, num_agents=1)
#
# for task_id in range(num_tasks):
#     baselines[task_id] = MLPBaseline(all_env_spec, reg_coef=1e-3, batch_size=64, epochs=1, learn_rate=1e-3, use_gpu=False)
#
# policy = MLPLPGFTW(all_env_spec, hidden_sizes=(32, 32), k=1, max_k=3, seed=SEED)
# agent = NPGFTW(envs, policy, baselines, normalized_step_size=0.01, seed=SEED, save_logs=True, new_col_mode='max_k')
#
# job_name = 'testing'
#
# task_id = 0
# env_id = envs[task_id].task_name
# print(env_id)
# agent.set_task(task_id=task_id)
#
# gamma = 0.995
# gae_lambda = 0.97
# n_episodes = 5
# n_training_rounds = 5
# episode_timesteps = 500
# stats = pd.DataFrame()
# for _ in range(n_training_rounds):
#     paths = do_rollout(N=n_episodes, policy=policy, T=episode_timesteps, env=envs[task_id], env_name=env_id, pegasus_seed=None)
#     process_samples.compute_returns(paths=paths, gamma=gamma)
#     process_samples.compute_advantages(paths=paths, baseline=agent.baseline, gamma=gamma, gae_lambda=gae_lambda)
#     eval_statistics = agent.train_from_paths(paths=paths, task_id=task_id)
#     eval_statistics.append(n_episodes)
#     # eval_statistics format: [mean_return, std_return, min_return, max_return, n_episodes]
#     print(eval_statistics)
#     pd_row = pd.DataFrame({'mean_return': [eval_statistics[0]],
#               'std_return': [eval_statistics[1]],
#               'min_return': [eval_statistics[2]],
#               'max_return': [eval_statistics[3]],
#               'N': [eval_statistics[4]]})
#     stats = pd.concat((stats, pd_row))
#     agent.add_approximate_cost(N=n_episodes, task_id=task_id, env_name=env_id, env=envs[task_id], num_cpu=num_cpu)
# print("done")

# # # # # #
# # OLD: working code for single task example
# # # # # #
#
# task_id = 0
# env_id = 'reach-v2'
# SEED = 0
# horizon_int = 500
# env = MetaWorldWrapper(task_name=env_id, render_mode=None)
# env_spec = EnvSpec(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], horizon=horizon_int,
#                    num_agents=1)
# envs = [env]
# num_cpu = 1
#
# env.horizon = horizon_int
#
# baseline_mtl = {}
# baseline_mtl[task_id] = MLPBaseline(env_spec, reg_coef=1e-3, batch_size=64, epochs=1, learn_rate=1e-3, use_gpu=False)
#
# policy = MLPLPGFTW(env_spec, hidden_sizes=(32, 32), k=1, max_k=3, seed=SEED)
# agent = NPGFTW(envs, policy, baseline_mtl, normalized_step_size=0.01, seed=SEED, save_logs=True, new_col_mode='max_k')
# agent.set_task(task_id=task_id)
#
# gamma = 0.995
# gae_lambda = 0.98
# N = 10
# T = 500
# stats = pd.DataFrame()
# for _ in range(10):
#     paths = do_rollout(N=N, policy=policy, T=T, env=env, env_name=env_id, pegasus_seed=None)
#     # compute returns
#     process_samples.compute_returns(paths=paths, gamma=gamma)
#     # compute advantages
#     process_samples.compute_advantages(paths=paths, baseline=agent.baseline, gamma=gamma, gae_lambda=gae_lambda)
#     # train from paths
#     eval_statistics = agent.train_from_paths(paths=paths, task_id=task_id)
#     eval_statistics.append(N)
#     # [mean_return, std_return, min_return, max_return, N]
#     print(eval_statistics)
#     pd_row = pd.DataFrame({'mean_return': [eval_statistics[0]],
#               'std_return': [eval_statistics[1]],
#               'min_return': [eval_statistics[2]],
#               'max_return': [eval_statistics[3]],
#               'N': [eval_statistics[4]]})
#     stats = pd.concat((stats, pd_row))
#     agent.add_approximate_cost(N=N, task_id=task_id, env_name=env_id, env=env, num_cpu=num_cpu)
# print("done")
#
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('module://backend_interagg')
#
# plt.errorbar(range(len(stats.mean_return.values)), stats.mean_return.values, stats.std_return.values)
# plt.show()
