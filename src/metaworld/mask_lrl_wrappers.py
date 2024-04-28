from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool
import torch

class MaskMetaworldWrapper(MetaWorldWrapper):
    def __init__(self, config_dir=None):
        # Config_dir is used as an argument purely for compatibility without allowing all arguments for safety
        # The initial task is irrelevant as we manually set another
        init_task = task_pool[0]
        self.name = 'MetaWorld' # Used internally by mask_lrl logic
        super().__init__(task_name=init_task)

    def get_task(self):
        return {'task_label': self.task_name}

    def get_all_tasks(self, requires_task_label):
        # Simply a compatibility function, returns all the tasks available from the task_pool
        tasks = [{'name': name, 'task': name, 'task_label': [ind]} for ind, name in enumerate(task_pool)]
        return tasks

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.negate_rewards:
            reward *= -1.0
        self.state = state
        self.reward = reward
        self.done = terminated  # changing 'terminated' to the admittedly more vague 'done' for compatibility
        self.truncated = truncated
        self.info = [info]
        self.curr_path_length = self.env.curr_path_length
        return self.state, self.reward, self.done, self.truncated, self.info

    def reset(self):
        # this version of reset() does not return self.info
        state, info = self.env.reset()
        self.env._partially_observable = False
        self.curr_path_length = 0
        self.state = state
        self.info = info
        return self.state.reshape(1, -1)

    def reset_task(self, task_info):
        self.change_task(task_name=task_info['name'])
        return self.reset()