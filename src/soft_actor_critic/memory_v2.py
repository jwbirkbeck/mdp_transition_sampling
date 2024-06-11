import os
import torch
import json
from random import choices


class MemoryV2:
    def __init__(self, state_length, action_length, max_memories, device):
        self.state_length = state_length
        self.action_length = action_length
        self.max_memories = max_memories
        self.device = device
        self.states = torch.zeros(size=(self.max_memories, self.state_length), device=self.device, requires_grad=False)
        self.actions = torch.zeros(size=(self.max_memories, self.action_length), device=self.device, requires_grad=False)
        self.next_states = torch.zeros(size=(self.max_memories, self.state_length), device=self.device, requires_grad=False)
        self.rewards = torch.zeros(size=(self.max_memories, 1), device=self.device, requires_grad=False)
        self.terminateds = torch.zeros(size=(self.max_memories, 1), dtype=torch.int, device=self.device, requires_grad=False)
        
        self.filled = False
        self.mem_ptr = 0

    def append(self, states, actions, next_states, rewards, terminateds):
        batch_len = states.shape[0]
        assert batch_len <= self.max_memories, 'batch size must be smaller than the maximum number of memories'
        if self.mem_ptr + batch_len <= self.max_memories:
            self.states[self.mem_ptr:self.mem_ptr + batch_len, :] = states
            self.actions[self.mem_ptr:self.mem_ptr + batch_len, :] = actions
            self.next_states[self.mem_ptr:self.mem_ptr + batch_len, :] = next_states
            self.rewards[self.mem_ptr:self.mem_ptr + batch_len, :] = rewards
            self.terminateds[self.mem_ptr:self.mem_ptr + batch_len, :] = terminateds
            self.mem_ptr += batch_len
        elif self.mem_ptr + batch_len > self.max_memories:
            split_a = self.max_memories - self.mem_ptr
            self.states[self.mem_ptr:self.max_memories, :] = states[:split_a, :]
            self.actions[self.mem_ptr:self.max_memories, :] = actions[:split_a, :]
            self.next_states[self.mem_ptr:self.max_memories, :] = next_states[:split_a, :]
            self.rewards[self.mem_ptr:self.max_memories, :] = rewards[:split_a, :]
            self.terminateds[self.mem_ptr:self.max_memories, :] = terminateds[:split_a, :]

            split_b = batch_len - split_a
            self.states[0:split_b, :] = states[split_a:, :]
            self.actions[0:split_b, :] = actions[split_a:, :]
            self.next_states[0:split_b, :] = next_states[split_a:, :]
            self.rewards[0:split_b, :] = rewards[split_a:, :]
            self.terminateds[0:split_b, :] = terminateds[split_a:, :]
            self.mem_ptr = split_b
            self.filled = True

    def sample(self, sample_size):
        max_ind = self.mem_ptr if not self.filled else self.max_memories
        sample_index = choices(population=range(max_ind), k=sample_size)  # Sampling with replacement

        states = self.states[sample_index]
        actions = self.actions[sample_index]
        next_states = self.next_states[sample_index]
        rewards = self.rewards[sample_index]
        terminateds = self.terminateds[sample_index]
        return states, actions, next_states, rewards, terminateds

    def get_memory_length(self):
        if self.filled:
            return self.max_memories
        else:
            return self.mem_ptr

    def save(self, save_path):

        config = {'state_length': self.state_length,
                  'action_length': self.action_length,
                  'max_memories': self.max_memories,
                  'filled': self.filled,
                  'mem_ptr': self.mem_ptr}

        json_dict = json.dumps(config)
        full_path = os.path.join(save_path, 'mem_config.json')
        with open(full_path, "w") as file:
            file.write(json_dict)
            os.chmod(full_path, 0o666)

        torch.save(self.states, os.path.join(save_path, 'states.pt'))
        torch.save(self.actions, os.path.join(save_path, 'actions.pt'))
        torch.save(self.next_states, os.path.join(save_path, 'next_states.pt'))
        torch.save(self.rewards, os.path.join(save_path, 'rewards.pt'))
        torch.save(self.terminateds, os.path.join(save_path, 'terminateds.pt'))

        os.chmod(os.path.join(save_path, 'states.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'actions.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'next_states.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'rewards.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'terminateds.pt'), mode=0o666)

    def load(self, load_path):

        with open(os.path.join(load_path, 'mem_config.json'), "r") as file:
            config = json.load(file)

        self.state_length = config['state_length']
        self.action_length = config['action_length']
        self.max_memories = config['max_memories']
        self.filled = config['filled']
        self.mem_ptr = config['mem_ptr']

        self.states = torch.load(os.path.join(load_path, 'states.pt'))
        self.actions = torch.load(os.path.join(load_path, 'actions.pt'))
        self.next_states = torch.load(os.path.join(load_path, 'next_states.pt'))
        self.rewards = torch.load(os.path.join(load_path, 'rewards.pt'))
        self.terminateds = torch.load(os.path.join(load_path, 'terminateds.pt'))

    def wipe(self):
        self.__init__(state_length=self.state_length,
                      action_length=self.action_length,
                      max_memories=self.max_memories,
                      device=self.device)
