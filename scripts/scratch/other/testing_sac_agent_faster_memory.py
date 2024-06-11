import torch
import numpy as np
from src.soft_actor_critic.sac_agent import SACAgent
from src.soft_actor_critic.sac_agent_v2 import SACAgentV2
from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool_10
import matplotlib.pyplot as plt
from time import time

env = MetaWorldWrapper(task_name='coffee-button-v2')
device = torch.device('cpu')

agent = SACAgent(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                 batch_size=500, memory_length=1e6, device=device, polyak=0.995)

agent_v2 = SACAgentV2(environment=env, hidden_layer_sizes=(128, 256), alpha=1e-4, gamma=0.99,
                 batch_size=500, memory_length=1e6, device=device, polyak=0.995)

rewards = []
rewards_v2 = []
times = []
times_v2 = []
for rep in range(100):
    start_time = time()
    rew, _ = agent.train_agent()
    end_time = time()
    rewards.append(rew)
    times.append(end_time - start_time)
    start_time = time()
    rew, _ = agent_v2.train_agent()
    end_time = time()
    rewards_v2.append(rew)
    times_v2.append(end_time - start_time)
    print(f"{rep}: diff is {times[-1] - times_v2[-1]}")
print("done")

plt.plot(rewards, label=['old'])
plt.plot(rewards_v2, label=['new'])
plt.legend()
plt.tight_layout()
plt.show()

diffs = [a - b for a,b in zip(times, times_v2)]

plt.plot(diffs)
plt.xlabel("Episode")
plt.ylabel("Time / s")
plt.title("Seconds lost to torch.cat")
plt.tight_layout()
plt.show()

plt.plot(np.cumsum(times) / 60, label='old')
plt.plot(np.cumsum(times_v2) / 60, label='old')
plt.xlabel("Episode")
plt.ylabel("Time (s)")
plt.title("")
plt.tight_layout()
plt.show()

plt.plot(np.cumsum(diffs))
plt.xlabel("Episode")
plt.ylabel("Time (s)")
plt.title("Seconds lost to torch.cat at episode N")
plt.tight_layout()
plt.show()

plt.plot(100 * np.cumsum(diffs) / np.cumsum(times_v2))
plt.xlabel("Episode")
plt.ylabel("Percent")
plt.title("Cumulative percent faster at episode N")
plt.tight_layout()
plt.show()

x = np.arange(1e-6, len(diffs), 1)
x[x <= 0] = 1e-6
y = np.array(diffs)
y[y <= 0] = 1e-6



# Estimate time taken with linear model for old and new methods
from scipy.optimize import curve_fit
def linear_model(x, m, c):
    return m * x + c

x = np.arange(0, len(times), 1)
y = np.array(times)
params_old, cov_old = curve_fit(linear_model, x, y)
y = times_v2
params_new, cov_new = curve_fit(linear_model, x, y)

# Manual intervention: assume time per ep is constant, not descreasing!
params_new[0] = 0

# Estimate time taken for some length of episodes
episode_numbers = np.arange(0, 400, 1)
estimated_times_old = linear_model(episode_numbers, *params_old)
estimated_times_new = linear_model(episode_numbers, *params_new)

plt.plot(np.cumsum(times) / 60, label='old')
plt.plot(np.cumsum(times_v2) / 60, label='new')
plt.ylabel("Time / mins")
plt.xlabel("Episode")
plt.legend()
plt.tight_layout()
plt.show()


plt.plot(times, label='old')
plt.plot(times_v2, label='new')
plt.ylabel("Time / secs")
plt.xlabel("Episode")
plt.legend()
plt.tight_layout()
plt.show()

def plot_times_for_run_length(n_eps):
    episode_numbers = np.arange(0, n_eps, 1)
    estimated_times_old = linear_model(episode_numbers, *params_old)
    estimated_times_new = linear_model(episode_numbers, *params_new)

    plt.plot(np.cumsum(estimated_times_old) / 60 **2, label='old')
    plt.plot(np.cumsum(estimated_times_new) / 60 **2, label='new')
    plt.ylabel("Runtime (hrs)")
    plt.xlabel("Episode")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_times_for_run_length(n_eps=2000)


def n_eps_for_max_runtime_hrs(max_runtime=48):
    n_eps_old = 0
    n_eps_new = 0
    eps_step = 1
    total_time_old = 0
    total_time_new = 0
    while total_time_old / 60**2 < max_runtime:
        n_eps_old += eps_step
        episode_numbers = np.arange(0, n_eps_old, 1)
        estimated_times_old = linear_model(episode_numbers, *params_old)
        total_time_old = np.sum(estimated_times_old)

    while total_time_new / 60**2 < max_runtime:
        n_eps_new += eps_step
        episode_numbers = np.arange(0, n_eps_new, 1)
        estimated_times_new = linear_model(episode_numbers, *params_new)
        total_time_new = np.sum(estimated_times_new)
    return n_eps_old, n_eps_new

n_eps_for_max_runtime_hrs(24)
