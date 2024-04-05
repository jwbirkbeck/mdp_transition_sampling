import torch
import torch.nn.functional as funcs
from torch.distributions.normal import Normal
from torch import nn
from torch import optim
import json
import os


class Actor(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, action_size, alpha, device):
        super().__init__()
        assert len(hidden_layer_sizes) == 2 and isinstance(hidden_layer_sizes, (list, tuple)), \
            "hidden_layer_sizes must be a list or tuple of length 2"
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.action_size = action_size
        self.alpha = alpha
        self.device = device
        # sac specific constants:
        self.min_log_sigma = -20
        self.max_log_sigma = 2
        self.reparameterisation_noise = 1e-6
        #
        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_sizes[0])
        self.fc2 = nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
        self.mu = nn.Linear(self.hidden_layer_sizes[1], self.action_size)
        self.log_sigma = nn.Linear(self.hidden_layer_sizes[1], self.action_size)
        self.optimiser = optim.Adam(self.parameters(), lr=self.alpha)
        self.to(self.device)

    def forward(self, observation):
        out = self.fc1(observation)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        # the output layers predict the mean and variance of the action (for each dimension independently)
        mu = self.mu(out)
        log_sigma = self.log_sigma(out)
        log_sigma = torch.clamp(log_sigma, min=self.min_log_sigma, max=self.max_log_sigma)
        return mu, log_sigma

    def sample_normal(self, observation, reparameterisation):
        if reparameterisation:
            pred_mu, pred_log_sigma = self.forward(observation)
            pred_sigma = torch.exp(pred_log_sigma)
            action_sampling_distribution = Normal(pred_mu, pred_sigma)
            actions_base = action_sampling_distribution.rsample()
        else:
            with torch.no_grad():  # it's quicker when the gradients aren't tracked
                pred_mu, pred_log_sigma = self.forward(observation)
                pred_sigma = torch.exp(pred_log_sigma)
                action_sampling_distribution = Normal(pred_mu, pred_sigma)
                actions_base = action_sampling_distribution.sample()
        tanh_actions = torch.tanh(actions_base)
        # The lob probs are used for calculating the actor and critic losses
        log_probs = action_sampling_distribution.log_prob(actions_base) - \
                    torch.log(1 - tanh_actions.pow(2) + self.reparameterisation_noise)
        # sum across the log probabilities as each dimension is modelled as an independent normal. Therefore,
        # the sum of the log probs across the columns is the log probs of the 8-dimensional action
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(1, keepdim=True)
        return tanh_actions, log_probs

    def checkpoint(self, location):
        self.save_config(location=location)
        self.pytorch_save(location=location)

    def save_config(self, location):
        config = {'input_size': self.input_size,
                  'hidden_layer_sizes': self.hidden_layer_sizes,
                  'action_size': self.action_size,
                  'alpha': self.alpha,
                  'device': str(self.device),
                  'min_log_sigma': self.min_log_sigma,
                  'max_log_sigma': self.max_log_sigma,
                  'reparameterisation_noise': self.reparameterisation_noise}
        json_dict = json.dumps(config)
        full_path = os.path.join(location, 'actor_config.json')
        with open(full_path, "w") as file:
            file.write(json_dict)
            os.chmod(full_path, 0o777)

    def pytorch_save(self, location):
        full_state_dict = {'model_dict': self.state_dict(),
                           'optimiser_dict': self.optimiser.state_dict()}
        torch.save(full_state_dict, os.path.join(location, 'actor.pt'))

    def load_from_checkpoint(self, location):
        self.load_config(location=location)
        self.pytorch_load(location=location)

    def load_config(self, location):
        with open(os.path.join(location, 'actor_config.json'), "r") as file:
            config = json.load(file)
        self.input_size = config['input_size']
        self.hidden_layer_sizes = config['hidden_layer_sizes']
        self.action_size = config['action_size']
        self.alpha = config['alpha']
        self.device = torch.device(config['device'])
        self.min_log_sigma = config['min_log_sigma']
        self.max_log_sigma = config['max_log_sigma']
        self.reparameterisation_noise = config['reparameterisation_noise']

    def pytorch_load(self, location):
        full_state_dict = torch.load(os.path.join(location, 'actor.pt'))
        self.load_state_dict(full_state_dict['model_dict'])
        self.optimiser.load_state_dict(full_state_dict['optimiser_dict'])