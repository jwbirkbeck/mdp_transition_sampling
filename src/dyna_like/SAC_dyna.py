import torch
import numpy as np
from math import isnan, floor
from src.soft_actor_critic.sac_agent import SACAgent
from src.dyna_like.gaussian_world_model import GaussianWorldModel

class SACDynaLike(SACAgent):
    def __init__(self, environment, hidden_layer_sizes, alpha, gamma, batch_size, memory_length, device,
                 polyak=0.995, dist_decay=0.995):
        super().__init__(environment=environment,
                         hidden_layer_sizes=hidden_layer_sizes,
                         alpha=alpha,
                         gamma=gamma,
                         batch_size=batch_size,
                         memory_length=memory_length,
                         device=device,
                         polyak=polyak)
        state_size = environment.observation_space.shape[0]
        action_size = environment.action_space.shape[0]
        self.worldmodel = GaussianWorldModel(state_size=state_size,
                                             action_size=action_size,
                                             alpha=alpha,
                                             device=device,
                                             hidden_layer_sizes=hidden_layer_sizes)
        self.wm_losses = []
        self.wm_w1_low = 9e9
        self.wm_w1_high = -9e9
        self.dist_decay=dist_decay
        self.sim_updates = 0

    def train_agent(self, train=True, render=False):
        ep_rewards = ep_return = 0
        discount = 1.0
        done = truncated = False
        state_raw,  _ = self.environment.reset()
        next_state = self.process_state(state_raw)
        while not (done or truncated):
            self.steps += 1
            state = next_state
            action = self.choose_action(observation=state)
            scaled_action = self.rescale_action(action=action)
            next_state_raw, reward, done, truncated, info = self.environment.step(scaled_action.cpu().numpy())
            next_state = self.process_state(state_raw=next_state_raw)
            wm_loss = self.worldmodel.train_step(state=state.reshape(1, -1),
                                                 action=action.reshape(1, -1),
                                                 next_state=next_state.reshape(1, -1),
                                                 reward=torch.tensor([reward]).reshape(1, -1))
            self.wm_losses.append(wm_loss) # Used for diagnostics - NOT for determining how many sim samples to use
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
            states, actions, next_states, rewards, dones = self.memory.sample(sample_size=self.batch_size)

            # Check worldmodel accuracy with W1 dist:
            wm_predictions = self.worldmodel.sample(state=states, action=actions, training=False)
            w1_dist = self.worldmodel.calc_w1_dist(predictions=wm_predictions, next_states=next_states, rewards=rewards)
            self.wm_w1_low *= 2 - self.dist_decay
            self.wm_w1_high *= self.dist_decay

            # Update accuracy bound trackers:
            self.wm_w1_low = w1_dist if w1_dist < self.wm_w1_low else self.wm_w1_low
            self.wm_w1_high = w1_dist if w1_dist > self.wm_w1_high else self.wm_w1_high

            numerator = (self.wm_w1_high - w1_dist)
            denominator = (self.wm_w1_high - self.wm_w1_low)
            denominator = float('nan') if denominator == 0 else denominator
            w1_dist_scaled = numerator / denominator
            w1_dist_scaled = 0 if isnan(w1_dist_scaled) else w1_dist_scaled
            # TODO: consider random actions instead of the actions we actually took
            # TODO: Read DYNA again - the above might be how it works to begin with!
            # TODO: read how DYNA deals with predicting terminal states
            sim_batch_size = floor(self.batch_size * w1_dist_scaled)
            self.sim_updates  += sim_batch_size
            if sim_batch_size > 0:
                sim_inds = torch.randint(high=self.batch_size, size=(sim_batch_size, ))
                # sim_states, sim_actions = states[sim_inds, :], actions[sim_inds, :]
                sim_states = states[sim_inds, :]
                sim_actions = torch.Tensor(np.array([self.environment.action_space.sample() for _ in sim_inds]))
                new_wm_preds = self.worldmodel.sample(state=sim_states, action=sim_actions, training=False)
                # sim_next_states, sim_rewards = new_wm_preds[sim_inds,:-1], new_wm_preds[sim_inds, -1].reshape(-1, 1)
                sim_next_states, sim_rewards = new_wm_preds[:, :-1], new_wm_preds[:, -1].reshape(-1, 1)
                sim_dones = dones[sim_inds, :]

                all_states = torch.cat((states, sim_states), dim=0)
                all_actions = torch.cat((actions, sim_actions), dim=0)
                all_next_states = torch.cat((next_states, sim_next_states), dim=0)
                all_rewards = torch.cat((rewards, sim_rewards), dim=0)
                all_dones = torch.cat((dones, sim_dones), dim=0)
            else:
                all_states = states
                all_actions = actions
                all_next_states = next_states
                all_rewards = rewards
                all_dones =dones

            self.updates += 1
            # Critic update:
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample_normal(observation=all_next_states,
                                                                 reparameterisation=False)
                target1_q = self.target_critic1.forward(observation=all_next_states, action=next_actions)
                target2_q = self.target_critic2.forward(observation=all_next_states, action=next_actions)
                next_min_q = torch.min(target1_q, target2_q) - self.entropy_alpha * next_log_probs
                next_q = all_rewards + self.gamma * (1 - all_dones) * next_min_q
            critic1_q = self.critic1.forward(observation=all_states, action=all_actions)
            critic2_q = self.critic2.forward(observation=all_states, action=all_actions)
            critic1_loss = torch.nn.MSELoss()(critic1_q, next_q)
            critic2_loss = torch.nn.MSELoss()(critic2_q, next_q)
            self.critic1.optimiser.zero_grad(set_to_none=True)
            critic1_loss.backward()
            self.critic1.optimiser.step()
            self.critic2.optimiser.zero_grad(set_to_none=True)
            critic2_loss.backward()
            self.critic2.optimiser.step()

            # Actor update:
            if self.updates % self.policy_train_delay_modulus == 0:
                pred_action, pred_log_probs = self.actor.sample_normal(observation=all_states, reparameterisation=True)
                pred1_q = self.critic1.forward(observation=all_states, action=pred_action)
                pred2_q = self.critic2.forward(observation=all_states, action=pred_action)
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

