import torch
import numpy as np
from src.policy_selector.sac_policy import SACPolicy

class SimplePolicySelector:
    def __init__(self, env, method, device, n_policies = None, task_policy_mapping=None):
        assert method in ['precomputed', 'bandit'], 'undefined method'
        if method == 'precomputed':
            assert task_policy_mapping is not None, "task_policy_mapping must be provided for method 'precomputed'"
            assert type(task_policy_mapping) == np.ndarray, "task_policy_mapping must be a numpy array"
            n_policies = np.max(task_policy_mapping[:, 1]) + 1
        if method == 'bandit':
            assert n_policies is not None, "n_policies must be provided for method 'bandit'"
            task_policy_mapping = None
        self.env = env
        self.method = method
        self.n_policies = n_policies
        self.task_policy_mapping = task_policy_mapping
        self.device = device

        self.hidden_layer_sizes = (128, 256)
        self.alpha = 1e-4
        self.gamma = 0.99
        self.batch_size = 500
        self.memory_length = 1e6
        self.polyak = 0.995


        self.policies = [SACPolicy(environment=env,
                                   hidden_layer_sizes=self.hidden_layer_sizes,
                                   alpha=self.alpha,
                                   gamma=self.gamma,
                                   batch_size=self.batch_size,
                                   memory_length=self.memory_length,
                                   device=self.device,
                                   polyak=self.polyak) for _ in range(self.n_policies)]

        self.active_policy = None

        self.current_task = self.env.task_name

    def play_episode(self):
        self.choose_policy()
        ep_rewards, ep_return = self.active_policy.train_agent()
        return ep_rewards, ep_return

    def choose_policy(self):
        if self.method == 'precomputed':
            current_task = self.env.task_name
            assert current_task in self.task_policy_mapping[:, 0], 'unmapped task encountered by precomputed mapping'
            task_policy_row = self.task_policy_mapping[self.task_policy_mapping[:, 0] == current_task][0]
            new_policy_ind = task_policy_row[1]
            self.active_policy = self.policies[new_policy_ind]
        else:
            raise NotImplementedError