import numpy as np
from src.metaworld.nonstationarity_distribution import MWNSDistribution
from src.utils.consts import ns_seq_length
# # # # # # # # # #
# Defining a standard sensor-actuator degradation for use across experiments
# # # # # # # # # #

def get_standard_ns_dist(env, seed=0):
    state_space = env.observation_space
    action_space = env.action_space

    ns_dist = MWNSDistribution(seed=seed, state_space=state_space,
                               action_space=action_space, current_task=env.task_name)

    ns_dist.task_dist.set_prob_task_change(probability=0.0)
    ns_dist.maint_dist.set_prob_maintenance(probability=0.0)

    ns_dist.sensor_dist.set_degradation_params(bias_prob=5e-2, error_prob=5e-2,
                                               bias_max_delta=0.002, error_max_delta=0.002)
    ns_dist.actuator_dist.set_degradation_params(bias_prob=5e-2, error_prob=5e-2,
                                                 bias_max_delta=0.002, error_max_delta=0.002)

    ns_dist.generate_sequence(sequence_length=ns_seq_length + 1)

    return ns_dist