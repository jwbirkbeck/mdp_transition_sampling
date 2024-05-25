from src.metaworld.wrapper import MetaWorldWrapper
from src.utils.consts import task_pool

env = MetaWorldWrapper(task_name='reach-v2', render_mode='human')

for _ in range(500):
    state, _ = env.reset()

state[3] = 0.5

for _ in range(1):
    env.set_internals_from_state(state=state)
    env.render()
