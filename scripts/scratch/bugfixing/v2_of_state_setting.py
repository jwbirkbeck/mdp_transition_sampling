import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.metaworld.wrapper import MetaWorldWrapper
from src.sampler.samplers import MDPDifferenceSampler
from src.utils.consts import task_pool, bounded_state_space
import mujoco
from scipy.spatial.transform import Rotation
np.set_printoptions(suppress=True)

task = task_pool[16]
env = MetaWorldWrapper(task_name=task, render_mode='human')
env.reset()
for _ in range(250):
    env.change_task()

state = np.zeros(shape=[1, 39])
# state[:, 0:3] = np.array([[0.3, 0.8, 0.1]])
# state[:, 4:7] = np.array([[0.1, 0.8, 0.0001]])
# state[:, 7:11] = np.array([[1.0, 0.0, 0.0, 0.0]])
# state[:, 7:11] = np.array([np.cos(0.5 / 2), 0.0, 0.0, np.sin(0.5/2)])
#
# hand_xyz = state[:, 0:3]
# gripper_open = state[:, 3]
# obj1_xyz = state[:, 4:7]
# obj1_qpos = state[:, 7:11]
# obj2_xyz = state[:, 11:14]
# obj2_qpos = state[:, 14:18]
# goal_xyz = state[:, 36:39]
#
# env.env.objHeight = env.env.data.geom("handle").xpos[2]
# goal_pos = obj1_xyz.copy() + np.array([0.2, -0.2, 0.0])
# env.env._target_pos = goal_pos
# env.env.model.body("door").pos = obj1_xyz
# env.env.model.site("goal").pos = goal_pos
#
env.env.model.geom()
env.env.model.body()
env.env.model.joint()
env.env.model.site('goal').pos
#
#
#
# obj1_qpos[1:3] = 0.0
# obj1_qpos = obj1_qpos / np.sqrt((obj1_qpos ** 2).sum())
# # Test radian angles are equal:
# ang = Rotation.from_quat(obj1_qpos).as_rotvec()[0][0]
# env.env._set_obj_xyz(ang)
# mujoco.mj_forward(env.env.model, env.env.data)
# env.render()

state = np.zeros(shape=[1, 39])
for _ in range(250):
    state[:, 0:3] = np.random.uniform(low=bounded_state_space.low[0:3], high=bounded_state_space.high[0:3])
    state[:, 4:7] = np.random.uniform(low=bounded_state_space.low[4:7], high=bounded_state_space.high[4:7])
    state[:, 7:11] = np.random.uniform(low=bounded_state_space.low[7:11], high=bounded_state_space.high[7:11])
    state[:, 36:39] = np.random.uniform(low=bounded_state_space.low[36:39], high=bounded_state_space.high[36:39])
    env.set_internals_from_state(state=state[0])
    env.render()




