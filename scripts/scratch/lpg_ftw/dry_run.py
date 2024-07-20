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


num_tasks = len(task_pool)
SEED = 50
num_cpu = 12

np.random.seed(SEED)
torch.manual_seed(SEED)

e = {}
baseline = {}
task_order = np.random.permutation(num_tasks)
for task_id in range(num_tasks):
    e[task_id] = e_unshuffled[task_order[task_id]]
    baseline[task_id] = MLPBaseline(e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=1, learn_rate=1e-3, use_gpu=False)

policy = MLPLPGFTW(e[0].spec, hidden_sizes=(32, 32), k=1, max_k=3)
agent = NPGFTW(e, policy, baseline, normalized_step_size=0.01, save_logs=True, new_col_mode='max_k')


job_name = 'deleteme_testing_lpgftw'
niter = 150
gamma = 0.995
gae_lambda = 0.97
num_cpu = 5
sample_mode = 'trajectories'
num_traj = 5

task_id = 0 # 0 # 14
env_id = task_pool[task_id]
agent.set_task(task_id=task_id)

best_policy = copy.copy(agent.policy)
best_perf = -1e8
train_curve = best_perf * np.ones(niter)
for i in range(niter):
    if train_curve[i - 1] > best_perf:
        best_policy = copy.copy(agent.policy)
        best_perf = train_curve[i - 1]
    args = dict(N=num_traj, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda, num_cpu=num_cpu, env_name=env_id, T=500)
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


train_curve_0 = np.append(train_curve_0, train_curve)

