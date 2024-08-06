import torch
import pickle
import os, sys
import numpy as np
import pandas as pd
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10
from src.utils.filepaths import *

# Lifetime Policy Reuse, CHIRP policy Reuse, and Bandit CHIRP Policy Reuse
from src.policy_selector.simple_policy_selector import SimplePolicySelector

# Lifelong learning with modulating masks
from src.metaworld.mask_lrl_wrappers import MaskMetaworldWrapper
from src.mask_lrl.network.network_bodies import DummyBody_CL
from src.mask_lrl.network.network_bodies import FCBody_SS
from src.mask_lrl.network.network_heads import GaussianActorCriticNet_SS
from src.mask_lrl.agent.PPO_agent import LLAgent
from src.mask_lrl.utils.config import Config
from src.mask_lrl.utils.normalizer import RunningStatsNormalizer
from src.mask_lrl.utils.normalizer import RewardRunningStatsNormalizer
from src.mask_lrl.utils.trainer_ll import run_iterations_w_oracle

# Lifelong policy gradient learning for faster training without forgetting
import gymnasium as gym
from src.lpg_ftw.mjrl.policies.gaussian_mlp_lpg_ftw import MLPLPGFTW
from src.lpg_ftw.mjrl.algos.npg_cg_ftw import NPGFTW
from src.lpg_ftw.mjrl.baselines.mlp_baseline import MLPBaseline
from src.lpg_ftw.mjrl.utils.gym_env import GymEnv


assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
run_index = int(sys.argv[1])

device = torch.device("cpu")
results_dir = os.path.join(results_path_iridis, 'agent_comparison_long_small')

with open(os.path.join(results_dir, 'task_sequences.pkl'), 'rb') as file:
    task_sequences = pickle.load(file)
    task_sequence = task_sequences['sequence'][0]

lpr_cluster_size = 6
cpr_cluster_size = 8
bandit_cpr_cluster_size = 6
mask_cluster_size = 10 # mask_lrl has no equivalent parameter to cluster size - therefore this value is irrelevant
lpg_cluster_size = 2

lpr_mode = cpr_mode = bandit_cpr_mode = mask_lrl_mode = lpg_ftw_mode = False
if run_index % 5 == 0:
    lpr_mode = True
    cluster_size = lpr_cluster_size
    agent_name = 'LPR'
elif run_index % 5 == 1:
    cpr_mode = True
    cluster_size = cpr_cluster_size
    agent_name = 'CPR'
elif run_index % 5 == 2:
    bandit_cpr_mode = True
    cluster_size = bandit_cpr_cluster_size
    agent_name = 'Bandit CPR'
elif run_index % 5 == 3:
    mask_lrl_mode = True
    cluster_size = mask_cluster_size
    agent_name = 'Mask-LRL'
elif run_index % 5 == 4:
    lpg_ftw_mode = True
    cluster_size = lpg_cluster_size
    agent_name = 'LPG-FTW'

base_env = MetaWorldWrapper(task_name=task_pool_10[task_sequence[0]], render_mode=None)  # tasks randomly change during experiment

# # # # #
# mask-lrl setup
# # # # #

if mask_lrl_mode:
    mask_task_fn = MaskMetaworldWrapper
    mask_task_fn.state_dim = mask_task_fn().observation_space.shape[0]
    mask_task_fn.action_dim = mask_task_fn().action_space.shape[0]

    mask_config = Config()
    mask_config.task_fn = MaskMetaworldWrapper
    mask_config.eval_task_fn = MaskMetaworldWrapper
    mask_config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
    mask_config.log_dir = None
    mask_config.task_ids = list(range(len(task_pool_10)))
    mask_config.lr = 5e-4
    mask_config.cl_preservation = 'supermask'
    mask_config.state_normalizer = RunningStatsNormalizer()
    mask_config.reward_normalizer = RewardRunningStatsNormalizer()
    mask_config.discount = 0.99
    mask_config.use_gae = True
    mask_config.gae_tau = 0.97
    mask_config.entropy_weight = 5e-3
    mask_config.rollout_length = 500  # 512 * 10 # (i.e., 512 * 2.5, if num_workers is set to 4)
    mask_config.optimization_epochs = 16
    mask_config.num_mini_batches = 15  # with rollout of 5120, 160 mini_batch gives 32 samples per batch
    mask_config.ppo_ratio_clip = 0.2
    mask_config.iteration_log_interval = 1
    mask_config.gradient_clip = 5
    mask_config.max_steps = 500  # 10_240_000 # taken from the default args
    mask_config.evaluation_episodes = 10
    mask_config.logger = None  # get_logger(log_dir=config.log_dir, file_name='train-log')
    mask_config.cl_requires_task_label = True

    mask_state_dim = mask_task_fn.state_dim
    mask_action_dim = mask_task_fn.action_dim
    mask_label_dim = 0
    mask_num_tasks = len(task_pool_10)
    mask_new_task_mask = 'random'  # taken from paper_requirements.txt in author's project code for Continual World

    mask_config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_SS(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh, discrete_mask=False,
                             num_tasks=mask_num_tasks, new_task_mask=mask_new_task_mask),
        critic_body=FCBody_SS(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh, discrete_mask=False,
                              num_tasks=mask_num_tasks, new_task_mask=mask_new_task_mask),
        num_tasks=mask_num_tasks, new_task_mask=mask_new_task_mask)

# # # # #
# lpg-ftw setup
# # # # #
if lpg_ftw_mode:
    lpg_env_dict = {'coffee-button-v2': 'lpg_ftw_wrappers:WrappedCoffeeButton',
                    'dial-turn-v2': 'lpg_ftw_wrappers:WrappedDialTurn',
                    'door-unlock-v2': 'lpg_ftw_wrappers:WrappedDoorUnlock',
                    'handle-press-side-v2': 'lpg_ftw_wrappers:WrappedHandlePressSide',
                    'handle-press-v2': 'lpg_ftw_wrappers:WrappedHandlePress',
                    'plate-slide-back-v2': 'lpg_ftw_wrappers:WrappedPlateSlideBack',
                    'plate-slide-v2': 'lpg_ftw_wrappers:WrappedPlateSlide',
                    'push-back-v2': 'lpg_ftw_wrappers:WrappedPushBack',
                    'reach-v2': 'lpg_ftw_wrappers:WrappedReach',
                    'reach-wall-v2': 'lpg_ftw_wrappers:WrappedReachWall'}

    lpg_e_unshuffled = {}
    for task_id, (env_id, entry_point) in enumerate(lpg_env_dict.items()):
        kwargs = {}
        gym.envs.register(
            id=env_id,
            entry_point='src.metaworld.' + entry_point,
            max_episode_steps=None,
            disable_env_checker=True,
            order_enforce=False,
            kwargs=kwargs
        )
        lpg_e_unshuffled[task_id] = GymEnv(env_id)

    lpg_num_tasks = len(task_pool_10)
    lpg_num_cpu = 1

    lpg_e = {}
    lpg_baseline = {}
    lpg_task_order = np.arange(lpg_num_tasks)
    for task_id in range(lpg_num_tasks):
        lpg_e[task_id] = lpg_e_unshuffled[lpg_task_order[task_id]]
        lpg_baseline[task_id] = MLPBaseline(lpg_e[task_id].spec, reg_coef=1e-3, batch_size=64, epochs=1, learn_rate=1e-3,
                                            use_gpu=False)

# # # # #
# agent initialisations
# # # # #

if lpr_mode:
    agent = SimplePolicySelector(env=base_env,
                                 method='bandit',
                                 device=device,
                                 task_names=task_pool_10,
                                 n_policies=lpr_cluster_size)

elif cpr_mode:
    with open(os.path.join(results_dir, f'cluster_info_{cpr_cluster_size}.pkl'), 'rb') as file:
        cpr_cluster_info = pickle.load(file)

    agent = SimplePolicySelector(env=base_env,
                                 method='precomputed',
                                 device=device,
                                 task_policy_mapping=cpr_cluster_info['mapping'])

elif bandit_cpr_mode:
    with open(os.path.join(results_dir, f'cluster_info_{bandit_cpr_cluster_size}.pkl'), 'rb') as file:
        bandit_cpr_cluster_info = pickle.load(file)

    agent = SimplePolicySelector(env=base_env,
                              method='bandit',
                              device=device,
                              task_names=task_pool_10,
                              n_policies=cluster_size)

    for ind, row in enumerate(agent.task_policy_mapping):
        row[bandit_cpr_cluster_info['clusters'][ind]] = 1.0

elif mask_lrl_mode:
    agent = LLAgent(mask_config)
    mask_config.agent_name = agent.__class__.__name__
    mask_tasks = agent.config.cl_tasks_info
    mask_config.cl_num_learn_blocks = 1

elif lpg_ftw_mode:
    lpg_policy = MLPLPGFTW(lpg_e[0].spec, hidden_sizes=(32, 32), k=1, max_k=lpg_cluster_size)
    agent = NPGFTW(lpg_e,
                   lpg_policy,
                   lpg_baseline,
                   normalized_step_size=0.01,
                   save_logs=False,
                   new_col_mode='max_k')
    agent.set_task(task_id=0)

# # # # #
# experiment
# # # # #

pd_results = pd.DataFrame()
for episode_ind, task_ind in enumerate(task_sequence):
    # # # #
    # set task
    # # # #
    curr_task_name = task_pool_10[task_ind]
    base_env.change_task(task_name=curr_task_name)

    if lpr_mode or cpr_mode or bandit_cpr_mode:
        ep_rew, _ = agent.play_episode()
    elif mask_lrl_mode:
        _, ep_rew = run_iterations_w_oracle(agent, [agent.config.cl_tasks_info[task_ind]])
        ep_rew = ep_rew[0]
    elif lpg_ftw_mode:
        lpg_args = dict(N=1, sample_mode='trajectories', gamma=0.995, gae_lambda=0.97, num_cpu=1, env_name=curr_task_name)
        lpg_stats = agent.train_step(**lpg_args)
        ep_rew = lpg_stats[0]

    pd_row = pd.DataFrame({'agent': [agent_name],
                           'episode': [episode_ind],
                           'task_ind': [task_ind],
                           'cluster_size': [cluster_size],
                           'reward': [ep_rew]})
    pd_results = pd.concat((pd_results, pd_row))
    pd_results.to_csv(os.path.join(results_dir, f'results_{run_index}.csv'), index=False)
