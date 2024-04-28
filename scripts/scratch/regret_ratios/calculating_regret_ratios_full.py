import sys, os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.consts import task_pool
np.set_printoptions(suppress=False)

results_dir = "/opt/project/results/task_regret_ratios_full/"

all_filenames = glob.glob(results_dir + "train_results_[0-9]*.csv")
training = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['task_index'] = re.search('[0-9]+', filename).group(0)
    training = pd.concat((training, tmp))

all_filenames = glob.glob(results_dir + "eval_results_[0-9]*.csv")
testing = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['index'] = None
    testing = pd.concat((testing, tmp))

results_dir = "/opt/project/results/task_transition_sampling/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
w_dists = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['index'] = None
    w_dists = pd.concat((w_dists, tmp))

training.reset_index()
testing.reset_index()
w_dists.reset_index()

#  If training agent's performance at an episode % 50 == 0 is high, then look at the regret ratios from
#  the eval at that stage

# for ind in range(len(task_pool)):
#     q = "task_index == '" + str(ind) + "' and agent == 'opt_agent'"
#     plt.plot(training.query(q).reset_index().rewards)
#     plt.show()

# Poor training results:
# 4, 10, 13, 16

comp_inds = [i for i in range(len(task_pool)) if i not in [4, 10, 13, 16]]
comp_tasks = [task_pool[i] for i in comp_inds]

w_dists = w_dists.query('n_states == 25 and n_transitions == 10')
w_dists = w_dists[w_dists.env_a.isin(comp_tasks)]
w_dists = w_dists[w_dists.env_b.isin(comp_tasks)]

def calc_regret_ratio(base_ind, comp_ind, max_mins=False):
    opt_a_in_b_eps = testing.query("env_a == '" + comp_tasks[base_ind] +
                                   "' and env_b == '" + comp_tasks[comp_ind] +
                                   "'  and agent == 'opt_agent'").train_episodes
    opt_b_eps = testing.query("env_a == '" + comp_tasks[comp_ind] +
                              "' and env_b == '" + comp_tasks[comp_ind] +
                              "'  and agent == 'opt_agent'").train_episodes
    min_b_eps = testing.query("env_a == '" + comp_tasks[comp_ind] +
                              "' and env_b == '" + comp_tasks[comp_ind] +
                              "'  and agent == 'min_agent'").train_episodes

    eps_to_use = np.array(pd.merge(pd.merge(opt_a_in_b_eps, opt_b_eps), min_b_eps).train_episodes.unique())

    testing2 = testing[testing.train_episodes.isin(eps_to_use)]

    opt_a_in_b = np.array(testing2.query("env_a == '" + comp_tasks[base_ind] +
                                         "' and env_b == '" + comp_tasks[comp_ind] +
                                         "'  and agent == 'opt_agent'").rewards)
    opt_b = np.array(testing2.query("env_a == '" + comp_tasks[comp_ind] +
                                    "' and env_b == '" + comp_tasks[comp_ind] +
                                    "'  and agent == 'opt_agent'").rewards)
    min_b = np.array(testing2.query("env_a == '" + comp_tasks[comp_ind] +
                                    "' and env_b == '" + comp_tasks[comp_ind] +
                                    "'  and agent == 'min_agent'").rewards)

    if not max_mins:
        numerator = opt_b - opt_a_in_b
        denominator = opt_b - min_b
    else:
        numerator = max(opt_b) - max(opt_a_in_b)
        denominator = max(opt_b) - min(min_b)
    return numerator, denominator

def get_rewards(base_ind, comp_ind):
    opt_a_in_b_eps = testing.query("env_a == '" + comp_tasks[base_ind] +
                                   "' and env_b == '" + comp_tasks[comp_ind] +
                                   "'  and agent == 'opt_agent'").train_episodes
    opt_b_eps = testing.query("env_a == '" + comp_tasks[comp_ind] +
                              "' and env_b == '" + comp_tasks[comp_ind] +
                              "'  and agent == 'opt_agent'").train_episodes
    min_b_eps = testing.query("env_a == '" + comp_tasks[comp_ind] +
                              "' and env_b == '" + comp_tasks[comp_ind] +
                              "'  and agent == 'min_agent'").train_episodes

    eps_to_use = np.array(pd.merge(pd.merge(opt_a_in_b_eps, opt_b_eps), min_b_eps).train_episodes.unique())

    testing2 = testing[testing.train_episodes.isin(eps_to_use)]

    opt_a_in_b = np.array(testing2.query("env_a == '" + comp_tasks[base_ind] +
                                         "' and env_b == '" + comp_tasks[comp_ind] +
                                         "'  and agent == 'opt_agent'").rewards)
    opt_b = np.array(testing2.query("env_a == '" + comp_tasks[comp_ind] +
                                    "' and env_b == '" + comp_tasks[comp_ind] +
                                    "'  and agent == 'opt_agent'").rewards)
    min_b = np.array(testing2.query("env_a == '" + comp_tasks[comp_ind] +
                                    "' and env_b == '" + comp_tasks[comp_ind] +
                                    "'  and agent == 'min_agent'").rewards)

    return opt_a_in_b, opt_b, min_b

base_ind = 0
reward_reductions = []
reward_scales = []
base_task = comp_tasks[base_ind]
for ind in range(len(comp_tasks)):
    nume, deno = calc_regret_ratio(base_ind, ind, max_mins=False)
    reward_reductions.append(nume)
    reward_scales.append(deno)

mean_rewards = []
mean_reward_reduction = []
mean_reward_scales = []
median_rewards = []
median_reward_reduction = []
median_reward_scales = []
max_minned_numerator = []
max_minned_denominator = []
for b_ind in range(len(comp_tasks)):
    for ind in range(len(comp_tasks)):
        opt_a_in_b, opt_b, min_b = get_rewards(base_ind=b_ind, comp_ind=ind)
        mean_rewards.append(np.mean(opt_a_in_b))
        mean_reward_reduction.append(np.mean(opt_b - opt_a_in_b))
        mean_reward_scales.append(np.mean(opt_b - min_b))
        median_rewards.append(np.median(opt_a_in_b))
        median_reward_reduction.append(np.median(opt_b - opt_a_in_b))
        median_reward_scales.append(np.median(opt_b - min_b))
        max_minned_numerator.append(np.mean(np.max(opt_b) - opt_a_in_b))
        max_minned_denominator.append(np.max(opt_b) - np.min(min_b))


mean_rewards = np.array(mean_rewards)
mean_reward_reduction = np.array(mean_reward_reduction)
mean_reward_scales = np.array(mean_reward_scales)
median_rewards = np.array(median_rewards)
median_reward_reduction = np.array(median_reward_reduction)
median_reward_scales = np.array(median_reward_scales)
max_minned_numerator = np.array(max_minned_numerator)
max_minned_denominator = np.array(max_minned_denominator)

mean_w_dists = []
median_w_dists = []
for base_task in comp_tasks:
    mean_w_dists = mean_w_dists + list(w_dists[w_dists.env_a == base_task].groupby('env_b').dist.mean().values)
    median_w_dists = median_w_dists + list(w_dists[w_dists.env_a == base_task].groupby('env_b').dist.median())
mean_w_dists = np.array(mean_w_dists)
median_w_dists = np.array(median_w_dists)

# Corelation between:
#   w dists and regret ratios
#   w dists and max_minned regret ratios
#   w dists and reward reductions
#   w dists and rewards

from scipy.stats import pearsonr

plt.scatter(mean_w_dists, mean_rewards)
plt.xlabel("Mean Wasserstein distance")
plt.ylabel("Mean Reward")
plt.title("Mean reward, Agent A in MDP B")
plt.tight_layout()
plt.savefig('mean_rewards.png')
plt.show()
print(pearsonr(mean_w_dists, mean_rewards))

plt.scatter(mean_w_dists, mean_reward_reduction)
plt.xlabel("Mean Wasserstein distance")
plt.ylabel("Mean Regret")
plt.title("Mean reward reduction, Agent A in MDP B")
plt.tight_layout()
plt.savefig('mean_regret.png')
plt.show()
print(pearsonr(mean_w_dists, mean_reward_reduction))

plt.scatter(mean_w_dists[mean_reward_reduction > 0], np.log(mean_reward_reduction[mean_reward_reduction > 0]))
plt.xlabel("Mean Wasserstein distance")
plt.ylabel("Log Mean Regret")
plt.title("Log Mean Regret, Agent A in MDP B")
plt.tight_layout()
plt.savefig("log_mean_regret.png")
plt.show()
print(pearsonr(mean_w_dists[mean_reward_reduction > 0], np.log(mean_reward_reduction[mean_reward_reduction > 0])))

plt.scatter(mean_w_dists, mean_reward_reduction / mean_reward_scales)
plt.xlabel("Mean Wasserstein distance")
plt.ylabel("Mean Scaled Regret")
plt.title("Mean Scaled Regret, Agent A in MDP B")
plt.tight_layout()
plt.savefig('mean_scaled_regret.png')
plt.show()
print(pearsonr(mean_w_dists, mean_reward_reduction / mean_reward_scales))

plt.scatter(mean_w_dists, max_minned_numerator)
plt.xlabel("Mean Wasserstein distance")
plt.ylabel('Regret')
plt.title("Alternative Regret: Max/Min for MDP B estimates")
plt.tight_layout()
plt.savefig('mean_max_minned_regret.png')
plt.show()
print(pearsonr(mean_w_dists, max_minned_numerator))



plt.scatter(mean_w_dists, max_minned_numerator / max_minned_denominator)
plt.xlabel("Mean Wasserstein distance")
plt.ylabel('Mean Scaled regret')
plt.title("Alternative Scaled Regret: Max/Min for MDP B estimates")
plt.tight_layout()
plt.savefig('mean_max_minned_scaled_regret.png')
plt.show()
print(pearsonr(mean_w_dists, max_minned_numerator / max_minned_denominator))













plt.scatter(median_w_dists, median_rewards)
plt.xlabel("Wasserstein distance")
plt.ylabel("Episode reward")
plt.title("Median reward, Agent A in MDP B")
plt.tight_layout()
plt.show()
print(pearsonr(median_w_dists, median_rewards))

plt.scatter(median_w_dists, median_reward_reduction)
plt.xlabel("Wasserstein distance")
plt.ylabel("Episode reward")
plt.title("Median reward reduction, Agent A in MDP B")
plt.tight_layout()
plt.show()
print(pearsonr(median_w_dists, median_reward_reduction))

# plt.scatter(mean_w_dists[median_reward_reduction > 0], np.log(median_reward_reduction[median_reward_reduction > 0]))
# plt.xlabel("Wasserstein distance")
# plt.ylabel("Log Episode reward")
# plt.title("Log Median reward reduction, Agent A in MDP B")
# plt.tight_layout()
# plt.show()
# print(pearsonr(mean_w_dists[median_reward_reduction > 0], np.log(median_reward_reduction[median_reward_reduction > 0])))

plt.scatter(median_w_dists, median_reward_reduction / median_reward_scales)
plt.xlabel("Wasserstein distance")
plt.ylabel("Median SOPR")
plt.title("Median Scaled Optimal Policy Regret, Agent A in MDP B")
plt.tight_layout()
plt.show()
print(pearsonr(median_w_dists, median_reward_reduction / median_reward_scales))

