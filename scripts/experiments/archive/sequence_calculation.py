import numpy as np
import pandas as pd

from src.metaworld.nonstationarity_distribution import MWNSDistribution
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool
from copy import copy
import matplotlib.pyplot as plt


# med_ns = MWNSDistribution(seed, state_space=state_space, action_space=action_space, current_task=init_task)
# high_ns = MWNSDistribution(seed, state_space=state_space, action_space=action_space, current_task=init_task)


def get_mdp_change_indices(dist):
    s_biases =[]
    s_errors = []
    a_biases = []
    a_errors = []
    tasks = []
    for ind in range(dist.sequence_length -1):
        s_bias_same = (dist.sensor_dist.bias_seq[ind, :] == dist.sensor_dist.bias_seq[ind + 1, :])
        s_error_same = (dist.sensor_dist.error_param_seq[ind, :] == dist.sensor_dist.error_param_seq[ind + 1, :])
        a_bias_same = (dist.actuator_dist.bias_seq[ind, :] == dist.actuator_dist.bias_seq[ind + 1, :])
        a_error_same = (dist.actuator_dist.error_param_seq[ind, :] == dist.actuator_dist.error_param_seq[ind + 1, :])
        task_same = (dist.task_dist.seq[ind] == dist.task_dist.seq[ind + 1])
        s_biases.append(np.all(s_bias_same))
        s_errors.append(np.all(s_error_same))
        a_biases.append(np.all(a_bias_same))
        a_errors.append(np.all(a_error_same))
        tasks.append(task_same)
    s_biases = np.array(s_biases)
    s_errors = np.array(s_errors)
    a_biases = np.array(a_biases)
    a_errors = np.array(a_errors)
    tasks = np.array(tasks)
    mdp_changes_bool = (s_biases * s_errors * a_biases * a_errors * tasks)
    indices = np.where(np.logical_not(mdp_changes_bool))[0]
    del s_biases, s_errors, a_biases, a_errors, tasks
    return indices


def calc_w1_of_nonstat_dist(dist, n_states, n_transitions, n_reps):
    mdp_change_indices = get_mdp_change_indices(dist=dist)
    results = pd.DataFrame()
    for ind in range(len(mdp_change_indices)):
        ind_a = mdp_change_indices[ind]
        ind_b = ind_a + 1
        task_a = dist.task_dist.task_pool[dist.task_dist.seq[ind_a]]
        task_b = dist.task_dist.task_pool[dist.task_dist.seq[ind_b]]
        env_a = MetaWorldWrapper(task_name=task_a)
        env_b = MetaWorldWrapper(task_name=task_b)
        ns_a = dist
        ns_b = copy(dist)
        ns_a.set_sequence_ind(ind=ind_a)
        ns_b.set_sequence_ind(ind=ind_b)
        ns_a.freeze()
        ns_b.freeze()
        env_a.ns_dist = ns_a
        env_b.ns_dist = ns_b
        env_a.reset()
        env_b.reset()
        # Calc wasserstein distance between MDPs with different
        state_space = env_a.observation_space
        action_space = env_a.action_space
        sampler = MDPDifferenceSampler(env_a=env_a,
                                       env_b=env_b,
                                       state_space=state_space,
                                       action_space=action_space)
        w1s = []
        for _ in range(n_reps):
            w1 = sampler.get_difference(n_states=n_states, n_transitions=n_transitions)
            w1s.append(w1)
        # Append distance to w1s
        pd_row = pd.DataFrame(w1s).T
        pd_row = pd_row.add_prefix('w1_')
        pd_row['ind_a'], pd_row['ind_b'] = ind_a, ind_b
        results = pd.concat((results, pd_row))
    return results


seed = 0
init_task = task_pool[0]
tmp_env = MetaWorldWrapper(task_name=init_task)
state_space = copy(tmp_env.observation_space)
action_space = copy(tmp_env.action_space)
del tmp_env

sequence_length = 500 * 10000
low_ns = MWNSDistribution(seed, state_space=state_space, action_space=action_space, current_task=init_task)
low_ns.sensor_dist.set_degradation_params(bias_prob=2e-4, error_prob=2e-4, bias_max_delta=0.1, error_max_delta=0.1)
low_ns.actuator_dist.set_degradation_params(bias_prob=2e-4, error_prob=2e-4, bias_max_delta=0.1, error_max_delta=0.1)
low_ns.generate_sequence(sequence_length=sequence_length, append=False)
indices = get_mdp_change_indices(dist=low_ns)
print(str(len(indices)))

low_ns.sensor_dist.degrad_params

w1s = calc_w1_of_nonstat_dist(dist=low_ns, n_states=50, n_transitions=5, n_reps=1)
w1s.to_csv()
