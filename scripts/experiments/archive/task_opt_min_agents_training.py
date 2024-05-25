import sys
import os
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

results_path = os.path.expanduser('~/mdp_transition_sampling/results/task_regret_ratios/')
os.makedirs(results_path) if not os.path.isdir(results_path) else None

# Filename and sole argument representing the index to be the base of comparison
assert len(sys.argv) == 2
assert sys.argv[1].isdigit(), 'python file index argument must be integer value'
base_index = int(sys.argv[1])

# Temporary exclusion of tasks requiring additional reward calculation mechanisms:
this_task = task_pool[base_index]
n_episodes = 4000

opt_env = MetaWorldWrapper(task_name=this_task)
opt_agent_str = results_path + 'opt___' + this_task
opt_agent = SAC("MlpPolicy", opt_env, verbose=1)
opt_logger = configure(opt_agent_str + '.log', ['csv'])
opt_agent.set_logger(opt_logger)
opt_agent.learn(total_timesteps=500*n_episodes, log_interval=1)
opt_agent.save(path=opt_agent_str + ".sb3")

min_env = MetaWorldWrapper(task_name=this_task)
min_env.negate_rewards = True
min_agent_str = results_path + 'min___' + this_task
min_agent = SAC("MlpPolicy", min_env, verbose=1)
min_logger = configure(min_agent_str + '.log', ['csv'])
min_agent.set_logger(min_logger)
min_agent.learn(total_timesteps=500*n_episodes, log_interval=1)
min_agent.save(path=min_agent_str + ".sb3")
