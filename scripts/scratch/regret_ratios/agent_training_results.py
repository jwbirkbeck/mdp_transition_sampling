import sys, os
from src.utils.consts import task_pool
from stable_baselines3 import SAC
# import glob
import pandas as pd
import matplotlib.pyplot as plt

# TODO: temporary exclusion of agents not trained:
exp_task_pool = [task_pool[i] for i in range(len(task_pool)) if i not in [12, 16]]
for task in exp_task_pool:
    opt_agent_log = pd.read_csv("/opt/project/results/task_regret_ratios/opt___" + task + ".log/progress.csv")
    plt.plot(opt_agent_log['rollout/ep_rew_mean'], label=task)
plt.legend()
plt.tight_layout()
plt.show()

ind = 5

ind += 1
task = exp_task_pool[ind]
opt_agent_log = pd.read_csv("/opt/project/results/task_regret_ratios/opt___" + task + ".log/progress.csv")
plt.plot(opt_agent_log['rollout/ep_rew_mean'], label=task)
plt.tight_layout()
plt.show()

# good inds = [6, 7, 8, 9, 10, 11, 13, 14]



task = task_pool[14]

opt_agent = SAC.load("/opt/project/results/task_regret_ratios/opt___" + task + ".sb3")
opt_agent_log = pd.read_csv("/opt/project/results/task_regret_ratios/opt___" + task + ".log/progress.csv")

min_agent = SAC.load("/opt/project/results/task_regret_ratios/min___" + task + ".sb3")
min_agent_log = pd.read_csv("/opt/project/results/task_regret_ratios/min___" + task + ".log/progress.csv")

plt.plot(opt_agent_log['rollout/ep_rew_mean'])
plt.plot(min_agent_log['rollout/ep_rew_mean'])
plt.show()



