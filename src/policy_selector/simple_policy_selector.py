import torch
import numpy as np
from src.policy_selector.sac_policy import SACPolicy

class SimplePolicySelector:
    def __init__(self, env, method, device, task_names=None, n_policies = None, task_policy_mapping=None):
        assert method in ['precomputed', 'bandit'], 'undefined method'
        if method == 'precomputed':
            assert task_policy_mapping is not None, "task_policy_mapping must be provided for method 'precomputed'"
            assert task_names is None, "method 'precomputed' does not take 'task_names'"
            assert type(task_policy_mapping) == np.ndarray, "task_policy_mapping must be a numpy array"
            n_policies = np.max(task_policy_mapping[:, 1]) + 1
            task_policy_updates = None
        if method == 'bandit':
            assert task_policy_mapping is None, "method 'bandit' does not take 'task_policy_mapping'"
            assert task_names is not None, "task_names must be provided for method 'bandit'"
            assert n_policies is not None, "n_policies must be provided for method 'bandit'"
            task_policy_mapping = np.zeros(shape=(len(task_names), n_policies), dtype=np.float32)
            task_policy_updates = np.ones(shape=(len(task_names), n_policies), dtype=np.float32)
        self.env = env
        self.method = method
        self.task_names = task_names
        self.n_policies = n_policies
        self.task_policy_mapping = task_policy_mapping
        self.task_policy_map_updates = task_policy_updates
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
        if self.method == 'bandit':
            self.update_bandit(ep_return=ep_return)
        return ep_rewards, ep_return

    def choose_policy(self):
        if self.method == 'precomputed':
            current_task = self.env.task_name
            assert current_task in self.task_policy_mapping[:, 0], 'unmapped task encountered by precomputed mapping'
            task_policy_row = self.task_policy_mapping[self.task_policy_mapping[:, 0] == current_task][0]
            new_policy_ind = task_policy_row[1]
            self.active_policy = self.policies[new_policy_ind]
        elif self.method == 'bandit':
            # LPR: select random action with p=0.1 else multi-armed bandit
            if np.random.uniform() < 0.1:
                new_policy_ind = np.random.choice(a=range(self.n_policies), size=1).item()
            else:
                current_task = self.env.task_name
                assert current_task in self.task_names, 'unmapped task encountered by bandit'
                task_ind = self.task_names.index(current_task)
                task_policy_row = self.task_policy_mapping[task_ind, :]
                new_policy_ind = np.argmax(task_policy_row,axis=0)
            self.active_policy_ind = new_policy_ind
            self.active_policy = self.policies[new_policy_ind]

    def update_bandit(self, ep_return):
        current_task = self.env.task_name
        assert current_task in self.task_names, 'unmapped task encountered by bandit'
        task_ind = self.task_names.index(current_task)
        inds = (task_ind, self.active_policy_ind)
        self.task_policy_mapping[inds] += (1 / self.task_policy_map_updates[inds]) * (ep_return - self.task_policy_mapping[inds])
        self.task_policy_map_updates[inds] += 1
