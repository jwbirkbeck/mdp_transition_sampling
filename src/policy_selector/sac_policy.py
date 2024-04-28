from src.soft_actor_critic.sac_agent import SACAgent


class SACPolicy(SACAgent):
    def __init__(self, environment, hidden_layer_sizes, alpha, gamma, batch_size, memory_length, device,
                 polyak=0.995):
        super().__init__(environment=environment,
                         hidden_layer_sizes=hidden_layer_sizes,
                         alpha=alpha,
                         gamma=gamma,
                         batch_size=batch_size,
                         memory_length=memory_length,
                         device=device,
                         polyak=polyak)

    def single_step(self, train, render):
        state_raw = self.environment.get_observation()
        state = self.process_state(state_raw)
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

        return state, action, next_state, reward, done, truncated, info