import torch
import torch.nn.functional as funcs
from torch import nn
from torch import optim
from random import randint, uniform
# from src.soft_actor_critic.memory import SACMemory
from src.soft_actor_critic.memory_v2 import MemoryV2
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, environment, alpha, epsilon, gamma, batch_size, memory_length, device,
                 hidden_layer_sizes=(64, 128)):
        super().__init__()

        self.environment = environment
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_length = memory_length
        self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device

        self.memory = MemoryV2(state_length=self.environment.observation_space.shape[0],
                               action_length = 1,
                               max_memories=self.memory_length,
                               device=self.device)

        self.training_steps = 0
        self.losses = []

        input_size = self.environment.observation_space.shape[0]
        output_size = self.environment.action_space.n
        self.fc1 = nn.Linear(input_size, self.hidden_layer_sizes[0])
        self.fc2 = nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
        self.fc3 = nn.Linear(self.hidden_layer_sizes[1], output_size)

        self.optimiser = optim.Adam(self.parameters(), lr=self.alpha)
        self.to(self.device)

    def train_agent(self, train=True, render=False):
        ep_rewards = ep_return = 0
        discount = 1.0
        terminated = truncated = False
        state_raw, _ = self.environment.reset()
        next_state = self.process_state(state_raw)
        while not (terminated or truncated):
            state = next_state
            current_q = self.forward(x=state)
            action = self.choose_action(predictions=current_q)
            next_state_raw, reward, terminated, truncated, info = self.environment.step(action)
            next_state = self.process_state(state_raw=next_state_raw)
            if render:
                self.environment.render()
            if train:
                self.training_steps += 1
                self.update_memory(state=state, action=action, next_state=next_state, reward=reward, terminated=terminated)
                self.batch_train()
            ep_rewards += reward.item()
            ep_return += reward.item() * discount
            discount *= self.gamma
        return ep_rewards, ep_return

    @torch.no_grad()
    def update_memory(self, state, action, next_state, reward, terminated):
        state = state.reshape((1, -1))
        action = torch.tensor([action], dtype=torch.int, device=self.device).reshape((1, -1))
        next_state = next_state.reshape((1, -1))
        reward = reward.reshape((1, -1))
        terminated = torch.tensor([terminated], dtype=torch.int, device=self.device).reshape((1, -1))
        self.memory.append(states=state, actions=action, next_states=next_state, rewards=reward, terminateds=terminated)

    def batch_train(self):
        m_size = self.memory.get_memory_length()
        if m_size >= self.batch_size:
            state_b, action_b, next_state_b, reward_b, done_b = self.memory.sample(sample_size=self.batch_size)
            action_b = action_b.to(torch.int)
            current_q = self.forward(state_b)
            with torch.no_grad():
                updated_q = current_q.detach().clone()
                next_q = self.forward(next_state_b)
                updated_q[torch.arange(self.batch_size), action_b.squeeze(1)] = \
                    reward_b.reshape(-1) + (1 - done_b).reshape(-1) * self.gamma * torch.max(next_q, dim=1).values
            self.optimiser.zero_grad(set_to_none=True)
            loss = funcs.mse_loss(updated_q, current_q)
            loss.backward()
            self.optimiser.step()
            if self.batch_size == self.memory_length:
                self.memory.wipe()
            self.losses.append(loss.item())

    def forward(self, x):
        out = self.fc1(x)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        out = self.fc3(out)
        return out

    @torch.no_grad()
    def choose_action(self, predictions):
        if uniform(0, 1) < self.epsilon:
            action = randint(0, self.environment.action_space.n - 1)
        else:
            action = torch.argmax(predictions).item()
        return action

    @torch.no_grad()
    def process_state(self, state_raw):
        return state_raw.flatten()
