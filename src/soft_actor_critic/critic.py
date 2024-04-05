import torch
import torch.nn.functional as funcs
from torch import nn
from torch import optim
import json
import os

class Critic(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, action_size, output_size, alpha, device):
        super().__init__()
        assert len(hidden_layer_sizes) == 2 and isinstance(hidden_layer_sizes, (list, tuple)), \
            "hidden_layer_sizes must be a list or tuple of length 2"
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.action_size = action_size
        self.output_size = output_size
        self.alpha = alpha
        self.device = device

        self.fc1 = nn.Linear(self.input_size + self.action_size, self.hidden_layer_sizes[0])
        self.fc2 = nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
        self.fc3 = nn.Linear(self.hidden_layer_sizes[1], output_size)
        self.optimiser = optim.Adam(self.parameters(), lr=self.alpha)
        self.to(self.device)

    def forward(self, observation, action):
        out = self.fc1(torch.cat([observation, action], dim=1))
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        out = self.fc3(out)
        return out


    def checkpoint(self, location, name):
        self.save_config(option_location=location, name=name)
        self.pytorch_save(option_location=location, name=name)

    def save_config(self, option_location, name):
        config = {'input_size': self.input_size,
                  'hidden_layer_sizes': self.hidden_layer_sizes,
                  'action_size': self.action_size,
                  'output_size': self.output_size,
                  'alpha': self.alpha,
                  'device': str(self.device)}
        json_dict = json.dumps(config)
        full_path = os.path.join(option_location, name + '_config.json')
        with open(full_path, "w") as file:
            file.write(json_dict)
            os.chmod(full_path, 0o777)

    def pytorch_save(self, option_location, name):
        full_state_dict = {'model_dict': self.state_dict(),
                           'optimiser_dict': self.optimiser.state_dict()}
        torch.save(full_state_dict, os.path.join(option_location, name + '.pt'))

    def load_from_checkpoint(self, location, name):
        self.load_config(location=location, name=name)
        self.pytorch_load(location=location, name=name)

    def load_config(self, location, name):
        with open(os.path.join(location, name + '_config.json'), "r") as file:
            config = json.load(file)
        self.input_size = config['input_size']
        self.hidden_layer_sizes = config['hidden_layer_sizes']
        self.action_size = config['action_size']
        self.output_size = config['output_size']
        self.alpha = config['alpha']
        self.device = torch.device(config['device'])

    def pytorch_load(self, location, name):
        full_state_dict = torch.load(os.path.join(location, name + '.pt'))
        self.load_state_dict(full_state_dict['model_dict'])
        self.optimiser.load_state_dict(full_state_dict['optimiser_dict'])
