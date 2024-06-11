import os
import torch
from random import choices


class SACMemory:
    def __init__(self, memory_length, device):
        self.memory_length = memory_length
        self.device = device
        self.state = torch.tensor([]).to(self.device)
        self.action = torch.tensor([]).to(self.device)
        self.next_state = torch.tensor([]).to(self.device)
        self.reward = torch.tensor([]).to(self.device)
        self.terminated = torch.tensor([], dtype=torch.int).to(self.device)

    def append(self, state, action, next_state, reward, done):
        self._assert_lengths_equal()
        mem_len = self.state.shape[0]
        if mem_len < self.memory_length:
            self.state = torch.cat((self.state, state), dim=0)
            self.action = torch.cat((self.action, action), dim=0)
            self.next_state = torch.cat((self.next_state, next_state), dim=0)
            self.reward = torch.cat((self.reward, reward), dim=0)
            self.terminated = torch.cat((self.terminated, done), dim=0)
        else:
            self.state = torch.cat((self.state[1:], state), dim=0)
            self.action = torch.cat((self.action[1:], action), dim=0)
            self.next_state = torch.cat((self.next_state[1:], next_state), dim=0)
            self.reward = torch.cat((self.reward[1:], reward), dim=0)
            self.terminated = torch.cat((self.terminated[1:], done), dim=0)

    def sample(self, sample_size):
        self._assert_lengths_equal()
        mem_len = self.state.shape[0]
        if mem_len < sample_size:
            sample_size = mem_len
        sample_index = choices(population=range(mem_len), k=sample_size)  # Sampling with replacement

        state = self.state[sample_index]
        action = self.action[sample_index]
        next_state = self.next_state[sample_index]
        reward = self.reward[sample_index]
        done = self.terminated[sample_index]
        return state, action, next_state, reward, done

    def _assert_lengths_equal(self):
        assert self.state.shape[0] == \
               self.next_state.shape[0] == \
               self.action.shape[0] == \
               self.reward.shape[0] == \
               self.terminated.shape[0], "Memory for an option has different lengths across memory tensors"

    def get_memory_length(self):
        self._assert_lengths_equal()
        return self.state.shape[0]

    def save(self, save_path):
        torch.save(self.state,os.path.join(save_path, 'state.pt'))
        torch.save(self.action,os.path.join(save_path, 'action.pt'))
        torch.save(self.next_state,os.path.join(save_path, 'next_state.pt'))
        torch.save(self.reward,os.path.join(save_path, 'reward.pt'))
        torch.save(self.terminated, os.path.join(save_path, 'done.pt'))

        os.chmod(os.path.join(save_path, 'state.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'action.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'next_state.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'reward.pt'), mode=0o666)
        os.chmod(os.path.join(save_path, 'done.pt'), mode=0o666)


    def load(self, load_path):
        self.state = torch.load(os.path.join(load_path, 'state.pt'))
        self.action = torch.load(os.path.join(load_path, 'action.pt'))
        self.next_state = torch.load(os.path.join(load_path, 'next_state.pt'))
        self.reward = torch.load(os.path.join(load_path, 'reward.pt'))
        self.terminated = torch.load(os.path.join(load_path, 'done.pt'))

    def wipe(self):
        self.__init__(memory_length=self.memory_length, device=self.device)
