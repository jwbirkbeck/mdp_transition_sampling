import torch
from random import choices


class DQNMemory:
    def __init__(self, memory_length, device):
        self.memory_length = memory_length
        self.device = device
        self.state = torch.tensor([]).to(self.device)
        self.action = torch.tensor([]).to(self.device)
        self.next_state = torch.tensor([]).to(self.device)
        self.reward = torch.tensor([]).to(self.device)
        self.done = torch.tensor([], dtype=torch.int).to(self.device)

    def get_memory_length(self):
        return self.state.shape[0]

    def _assert_lengths_equal(self):
        assert self.state.shape[0] == \
               self.action.shape[0] == \
               self.next_state.shape[0] == \
               self.reward.shape[0] == \
               self.done.shape[0], "Memory for an option has different lengths across memory tensors"

    def update_memory(self, state, action, next_state, reward, done):
        mem_len = self.state.shape[0]
        if mem_len < self.memory_length:
            self.state = torch.cat((self.state, state), dim=0)
            self.action = torch.cat((self.action, action), dim=0)
            self.next_state = torch.cat((self.next_state, next_state), dim=0)
            self.reward = torch.cat((self.reward, reward), dim=0)
            self.done = torch.cat((self.done, done), dim=0)
        else:
            self.state = torch.cat((self.state[1:], state), dim=0)
            self.action = torch.cat((self.action[1:], action), dim=0)
            self.next_state = torch.cat((self.next_state[1:], next_state), dim=0)
            self.reward = torch.cat((self.reward[1:], reward), dim=0)
            self.done = torch.cat((self.done[1:], done), dim=0)

    def wipe(self):
        self.__init__(self, memory_length=self.memory_length, device=self.device)

    def sample_memory(self, sample_size):
        self._assert_lengths_equal()
        mem_len = self.state.shape[0]
        if mem_len < sample_size:
            sample_size = mem_len
        sample_index = choices(population=range(mem_len), k=sample_size)  # Sampling with replacement
        state = self.state[sample_index]
        action = self.action[sample_index]
        next_state = self.next_state[sample_index]
        reward = self.reward[sample_index]
        done = self.done[sample_index]
        return state, action, next_state, reward, done
