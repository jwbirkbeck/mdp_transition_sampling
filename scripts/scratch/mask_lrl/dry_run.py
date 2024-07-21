
import torch

from src.metaworld.mask_lrl_wrappers import MaskMetaworldWrapper
from src.utils.consts import task_pool
from src.mask_lrl.network.network_bodies import DummyBody_CL
from src.mask_lrl.network.network_bodies import FCBody_SS
from src.mask_lrl.network.network_heads import GaussianActorCriticNet_SS
from src.mask_lrl.agent.PPO_agent import LLAgent
from src.mask_lrl.utils.config import Config
from src.mask_lrl.utils.normalizer import RunningStatsNormalizer
from src.mask_lrl.utils.normalizer import RewardRunningStatsNormalizer
from src.mask_lrl.utils.trainer_ll import run_iterations_w_oracle


task_fn = MaskMetaworldWrapper
task_fn.state_dim = task_fn().observation_space.shape[0]
task_fn.action_dim = task_fn().action_space.shape[0]

config = Config()
config.task_fn = MaskMetaworldWrapper
config.eval_task_fn = MaskMetaworldWrapper
config.optimizer_fn = lambda params, lr: torch.optim.Adam(params, lr=lr)
config.log_dir = '/opt/project/mask_lrl/'
config.task_ids = list(range(len(task_pool)))
config.lr = 5e-4
config.cl_preservation = 'supermask'
config.state_normalizer = RunningStatsNormalizer()
config.reward_normalizer = RewardRunningStatsNormalizer()
config.discount = 0.99
config.use_gae = True
config.gae_tau = 0.97
config.entropy_weight = 5e-3
config.rollout_length = 500 # 512 * 10 # (i.e., 512 * 2.5, if num_workers is set to 4)
config.optimization_epochs = 16
config.num_mini_batches = 15 # with rollout of 5120, 160 mini_batch gives 32 samples per batch
config.ppo_ratio_clip = 0.2
config.iteration_log_interval = 1
config.gradient_clip = 5
config.max_steps = 500 # 10_240_000 # taken from the default args
config.evaluation_episodes = 10
config.logger = None # get_logger(log_dir=config.log_dir, file_name='train-log')
config.cl_requires_task_label = True

state_dim = task_fn.state_dim
action_dim = task_fn.action_dim
label_dim = 0
num_tasks = len(task_pool)
new_task_mask = 'random'  # taken from paper_requirements.txt in author's project code

config.network_fn = lambda state_dim, action_dim, label_dim: GaussianActorCriticNet_SS(
        state_dim, action_dim, label_dim,
        phi_body=DummyBody_CL(state_dim, task_label_dim=label_dim),
        actor_body=FCBody_SS(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh, \
            discrete_mask=False, num_tasks=num_tasks, new_task_mask=new_task_mask),
        critic_body=FCBody_SS(state_dim + label_dim, hidden_units=(128, 128), gate=torch.tanh, \
            discrete_mask=False, num_tasks=num_tasks, new_task_mask=new_task_mask),
        num_tasks=num_tasks, new_task_mask=new_task_mask)

agent = LLAgent(config)
config.agent_name = agent.__class__.__name__
tasks = agent.config.cl_tasks_info
config.cl_num_learn_blocks = 1

config.max_steps = 500 * 1 # 10_240_000 # taken from the default args
# steps, rewards = run_iterations_w_oracle(agent, tasks[0:1])
# print("done")

all_rew = []
for _ in range(300):
    print(_)
    steps, rewards = run_iterations_w_oracle(agent, tasks[14:15])
    all_rew.append(rewards[0])
print("done")

import matplotlib.pyplot as plt
plt.plot(all_rew)
plt.show()
