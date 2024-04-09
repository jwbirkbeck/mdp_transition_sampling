import torch
import numpy as np
import ot

class GaussianWorldModel(torch.nn.Module):
    def __init__(self, state_size, action_size, alpha, device, hidden_layer_sizes=(64, 64)):
        super().__init__()
        assert len(hidden_layer_sizes) == 2 and isinstance(hidden_layer_sizes, (list, tuple)), \
            "hidden_layer_sizes must be a list or tuple of length 2"
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = state_size + 1
        self.alpha = alpha
        self.device = device

        # constants used to clamp the Gaussian distribution's variance
        self.min_log_sigma = -20
        self.max_log_sigma = 2

        self.fc1 = torch.nn.Linear(self.state_size + self.action_size, self.hidden_layer_sizes[0])
        self.fc2 = torch.nn.Linear(self.hidden_layer_sizes[0], self.hidden_layer_sizes[1])
        self.mu = torch.nn.Linear(self.hidden_layer_sizes[1], self.output_size)
        self.log_sigma = torch.nn.Linear(self.hidden_layer_sizes[1], self.output_size)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.alpha)
        self.to(self.device)

    def forward(self, state, action):
        obs = torch.cat((state, action), dim=1)
        out = self.fc1(obs)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        # For Bayesian
        mu = self.mu(out)
        log_sigma = self.log_sigma(out)
        log_sigma = torch.clamp(log_sigma, min=self.min_log_sigma, max=self.max_log_sigma)
        return mu, log_sigma

    def sample(self, state, action, training):
        if training:
            pred_mu, pred_log_sigma = self.forward(state=state, action=action)
            distribution = torch.distributions.normal.Normal(pred_mu, torch.exp(pred_log_sigma))
            prediction = distribution.rsample()
        else:
            with torch.no_grad():  # it's quicker when the gradients aren't tracked
                pred_mu, pred_log_sigma = self.forward(state=state, action=action)
                distribution = torch.distributions.normal.Normal(pred_mu, torch.exp(pred_log_sigma))
                prediction = distribution.sample()
        return prediction

    def train_step(self, state, action, next_state, reward, loss_func=torch.nn.L1Loss()):
        # For a single state action pair, minimizing the L1Loss is equivalent to minimizing the W1 distance
        # as for a single sample, the infinum is trivially eliminated
        predictions = self.sample(state=state, action=action, training=True)
        actuals = torch.cat((next_state, reward), dim=1)
        self.optimiser.zero_grad(set_to_none=True)
        loss = loss_func(predictions, actuals)
        loss.backward()
        self.optimiser.step()
        return loss.detach().item()

    def evaluate(self, states, actions, next_states, rewards, loss_func=torch.nn.L1Loss()):
        with torch.no_grad():
            predictions = self.sample(state=states, action=actions, training=False)
            actuals = torch.cat((next_states, rewards), dim=1)
            loss = loss_func(predictions, actuals)
        return loss.detach().item()

    def calc_w1_dist(self, predictions, next_states, rewards):
        # Predict the next states and rewards for the provided states and actions
        # Calculate the W1 distance between the true (s', r) and the predicted (s', r) samples
        actuals = torch.cat((next_states, rewards), dim=1).numpy()
        a = np.ones(shape=predictions.shape[0]) / predictions.shape[0]
        b = np.ones(shape=actuals.shape[0]) / actuals.shape[0]
        M = ot.dist(predictions.numpy(), actuals, metric='euclidean', p=1)
        return ot.emd2(a=a, b=b, M=M)
