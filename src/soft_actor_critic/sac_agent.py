import os, json
import torch
import torch.nn.functional as funcs
from src.soft_actor_critic.actor import Actor
from src.soft_actor_critic.critic import Critic
from src.soft_actor_critic.memory import SACMemory


class SACAgent:
    def __init__(self, environment, hidden_layer_sizes, alpha, gamma, batch_size, memory_length, device,
                 polyak=0.995):
        self.environment = environment
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_alpha = torch.tensor([[0.1]]).to(device)
        self.target_entropy = torch.tensor([[-4.0]]).to(device)
        self.batch_size = batch_size
        self.memory_length = memory_length
        self.device = device
        self.polyak = polyak
        action_low_bound = torch.from_numpy(self.environment.action_space.low).to(self.device)
        action_high_bound = torch.from_numpy(self.environment.action_space.high).to(self.device)
        self.action_bounds = [action_low_bound, action_high_bound]
        self.policy_train_delay_modulus = 2  # Polyak update the target networks every training step if this is 1
        self.updates = 0  # iterator used for modulating how often actor and soft updates occur
        self.steps = 0
        self.training_steps = 0

        self.memory = SACMemory(memory_length=self.memory_length, device=self.device)

        input_size = self.environment.observation_space.shape[0]
        action_size = self.environment.action_space.shape[0]
        self.actor = Actor(input_size=input_size, hidden_layer_sizes=self.hidden_layer_sizes, action_size=action_size,
                           alpha=self.alpha, device=self.device)
        self.critic1 = Critic(input_size=input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                              action_size=action_size, output_size=1, alpha=self.alpha, device=self.device)
        self.critic2 = Critic(input_size=input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                              action_size=action_size, output_size=1, alpha=self.alpha, device=self.device)
        self.target_critic1 = Critic(input_size=input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                     action_size=action_size, output_size=1, alpha=self.alpha, device=self.device)
        self.target_critic2 = Critic(input_size=input_size, hidden_layer_sizes=self.hidden_layer_sizes,
                                     action_size=action_size, output_size=1, alpha=self.alpha, device=self.device)
        self.polyak_update(polyak=1)

    def train_agent(self, train=True, render=False):
        ep_rewards = ep_return = 0
        discount = 1.0
        done = truncated = False
        state_raw, _ = self.environment.reset()
        next_state = self.process_state(state_raw)
        while not (done or truncated):
            self.steps += 1
            state = next_state
            action = self.choose_action(observation=state)
            scaled_action = self.rescale_action(action=action)
            next_state_raw, reward, done, truncated, info = self.environment.step(scaled_action.cpu().numpy())
            next_state = self.process_state(state_raw=next_state_raw)
            if render:
                self.environment.render()
            if train:
                self.training_steps += 1
                self.update_memory(state=state, action=action, next_state=next_state, reward=reward, done=done)
                self.update_parameters()
            ep_rewards += reward
            ep_return += reward * discount
            discount *= self.gamma
        return ep_rewards, ep_return

    def update_parameters(self):
        m_size = self.memory.get_memory_length()
        if m_size >= self.batch_size:
            state, action, next_state, reward, done = self.memory.sample(sample_size=self.batch_size)
            self.updates += 1 * self.batch_size
            # Critic update:
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample_normal(observation=next_state,
                                                                        reparameterisation=False)
                target1_q = self.target_critic1.forward(observation=next_state, action=next_actions)
                target2_q = self.target_critic2.forward(observation=next_state, action=next_actions)
                next_min_q = torch.min(target1_q, target2_q) - self.entropy_alpha * next_log_probs
                next_q = reward + self.gamma * (1 - done) * next_min_q
            critic1_q = self.critic1.forward(observation=state, action=action)
            critic2_q = self.critic2.forward(observation=state, action=action)
            critic1_loss = funcs.mse_loss(critic1_q, next_q)
            critic2_loss = funcs.mse_loss(critic2_q, next_q)
            self.critic1.optimiser.zero_grad(set_to_none=True)
            critic1_loss.backward()
            self.critic1.optimiser.step()
            self.critic2.optimiser.zero_grad(set_to_none=True)
            critic2_loss.backward()
            self.critic2.optimiser.step()

            # Actor update:
            if self.updates % self.policy_train_delay_modulus == 0:
                pred_action, pred_log_probs = self.actor.sample_normal(observation=state, reparameterisation=True)
                pred1_q = self.critic1.forward(observation=state, action=pred_action)
                pred2_q = self.critic2.forward(observation=state, action=pred_action)
                min_pred_q = torch.min(pred1_q, pred2_q)
                policy_loss = torch.mean(-min_pred_q + self.entropy_alpha * pred_log_probs.reshape(-1))
                self.actor.optimiser.zero_grad(set_to_none=True)
                policy_loss.backward()
                self.actor.optimiser.step()
                self.polyak_update()

                # Entropy alpha update performed whenever actor updated:
                log_entr_alpha = torch.log(self.entropy_alpha).detach()
                log_entr_alpha.grad = None
                log_entr_alpha.requires_grad = True
                entropy_loss = torch.mean(-1.0 * log_entr_alpha * (pred_log_probs.detach() + self.target_entropy))
                entropy_loss.backward()
                with torch.no_grad():
                    log_entr_alpha = log_entr_alpha - self.alpha * log_entr_alpha.grad
                    self.entropy_alpha = torch.exp(log_entr_alpha)
                    if self.entropy_alpha > 0.1:
                        self.entropy_alpha = torch.tensor([[0.1]], device=self.device)
            if self.batch_size == self.memory_length:
                self.memory.wipe()

    @torch.no_grad()
    def choose_action(self, observation):
        # Because we aren't using the gradients here, we don't need to reparameterise or know the log probs
        action, _ = self.actor.sample_normal(observation=observation, reparameterisation=False)
        return action

    @torch.no_grad()
    def rescale_action(self, action):
        scaling_factor = (self.action_bounds[1] - self.action_bounds[0]) / 2.0
        position_adjustment = self.action_bounds[0] - (-1.0 * scaling_factor)  # -1.0 is always the low bound of tanh
        scaled_action = action * scaling_factor + position_adjustment
        return scaled_action

    @torch.no_grad()
    def process_state(self, state_raw):
        return torch.tensor(state_raw, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def update_memory(self, state, action, next_state, reward, done):
        state = state.reshape((1, -1))
        action = action.reshape((1, -1))
        next_state = next_state.reshape((1, -1))
        reward = torch.tensor([reward], dtype=torch.float, device=self.device).reshape((1, -1))
        done = torch.tensor([done], device=self.device).reshape((1, -1))
        self.memory.append(state=state, action=action, next_state=next_state, reward=reward, done=done)

    @torch.no_grad()
    def polyak_update(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))


    def _save_config(self, save_path):
        config = {'hidden_layer_sizes': self.hidden_layer_sizes,
                  'alpha': self.alpha,
                  'gamma': self.gamma,
                  'batch_size': self.batch_size,
                  'memory_length': self.memory_length,
                  'polyak': str(self.polyak)}
        json_dict = json.dumps(config)
        full_path = os.path.join(save_path, 'config.json')
        with open(full_path, "w") as file:
            file.write(json_dict)
            os.chmod(full_path, 0o777)

    def _check_config(self, load_path):
        full_path = os.path.join(load_path, 'config.json')

        assert os.path.isfile(full_path), 'config error: a config file does not exist at provided load_path'
        with open(full_path, "r") as file:
            config = json.load(file)

        assert config['hidden_layer_sizes'] == list(self.hidden_layer_sizes),\
            'config error: current agent does not match loaded agent hidden_layer_sizes: ' + str(self.hidden_layer_sizes)
        assert config['alpha'] == self.alpha,\
            'config error: current agent does not match loaded agent alpha: ' + str(self.alpha)
        assert config['gamma'] == self.gamma,\
            'config error: current agent does not match loaded agent gamma: ' + str(self.gamma)
        assert config['batch_size'] == self.batch_size,\
            'config error: current agent does not match loaded agent batch_size: ' + str(self.batch_size)
        assert config['memory_length'] == self.memory_length,\
            'config error: current agent does not match loaded agent memory_length: ' + str(self.memory_length)
        assert config['polyak'] == str(self.polyak),\
            'config error: current agent does not match loaded agent polyak: ' + str(self.polyak)

    @torch.no_grad()
    def save(self, save_path):
        os.makedirs(save_path) if not os.path.isdir(save_path) else None
        self._save_config(save_path=save_path)
        self.memory.save(save_path=save_path)
        for model_name in ['actor', 'critic1', 'critic2', 'target_critic1', 'target_critic2']:
            model = getattr(self, model_name)
            torch.save(model.state_dict(), os.path.join(save_path, model_name + '.pt'))

    @torch.no_grad()
    def load(self, load_path):
        assert os.path.isdir(load_path), 'load error: load_path is not an existing directory'
        self._check_config(load_path=load_path)
        self.memory.load(load_path=load_path)
        for model_name in ['actor', 'critic1', 'critic2', 'target_critic1', 'target_critic2']:
            model_path = os.path.join(load_path, model_name + '.pt')
            assert os.path.isfile(model_path), 'load error: ' + model_name + '.pt does not exist at load_path'
            model = getattr(self, model_name)
            model.load_state_dict(torch.load(model_path))