import os
import numpy as np
from gymnasium.spaces import Box


"""
Task pool selection process:
- Beginning with the 52 Metaworld tasks,
- 21 remaining after elimiating tasks not perfectly solvable by single-task agents (see Metaworld paper Appendix).
    Rationale: we wish to investigate difficulty specifically induced by non-stationarity.
    Having tasks so difficult they are difficult to learn at all may reduce the ability to analyse results for this end.
- 18 remaining after eliminating tasks which have reward functions dependent upon internal mujoco state.
    Rationale: We cannot fully set the internal state in Metaworld's MuJoCo implementation, so we use only tasks where
    the rewards are calculable from state space information alone.
    As an example, internal forces are too complex to set manually within MuJoCo, but may feature indirectly
    in certain tasks' rewards calculations.
- 17 remaining after eliminating sweep-into-v2 due to a unique hang on calling reset(). This is due to the methods used 
    to manually set the states during transition sampling causing a while loop to never be satisfied. 
"""

task_pool = ['button-press-topdown-v2',
             'button-press-v2',
             'button-press-wall-v2',
             'coffee-button-v2',
             'coffee-push-v2',
             'dial-turn-v2',
             'door-close-v2',
             'door-unlock-v2',
             'handle-press-side-v2',
             'handle-press-v2',
             'peg-insert-side-v2',
             'plate-slide-back-v2',
             'plate-slide-v2',
             'push-back-v2',
             'reach-v2',
             'reach-wall-v2',
             'soccer-v2']

task_pool_10 = ['coffee-button-v2',
                'dial-turn-v2',
                'door-unlock-v2',
                'handle-press-side-v2',
                'handle-press-v2',
                'plate-slide-back-v2',
                'plate-slide-v2',
                'push-back-v2',
                'reach-v2',
                'reach-wall-v2']


# # # # # # # # # #
# The default observation space of Metaworld is too wide - the lower and upper bounds are infinite in many cases.
# For sampling, we need to define finite bounds which still encompass all possible states, e.g. [-10, 10] in most cases
# # # # # # # # # #
hand_low = np.array((-0.525, 0.348, -0.0525))
hand_high = np.array((0.525, 1.025, 0.7))
# Note an error in Metaworld's default bounds - the gripper is NOT bounded by [-1, 1], as evidenced by
#   the calculation `gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)`
#   in the '_get_curr_obs_combined_no_goal' method of MetaWorld's `sawyer_xyz_env.py`
grip_low = np.array([0.0])
grip_high = np.array([1.0])
obj_xyz_low = np.array((-10., -10., -10.))
obj_xyz_high = np.array((10., 10., 10.))
obj_quat_low = -1.0 * np.array((1.0, 1.0, 1.0, 1.0))
obj_quat_high = np.array((1.0, 1.0, 1.0, 1.0))
goal_xyz_low = np.array((-10., -10., -10.))
goal_xyz_high = np.array((10, 10, 10))

# The below is the description of the state space from the paper's code for reference.
# end_eff_xpos = state[0:3]
# gripper_open_normalized = state[3]
# obj1_xpos = state[4:7]
# obj1_qpos = state[7:11]
# obj2_xpos = state[11:14]
# obj2_qpos = state[14:18]
# prev_obs = state[18:36]
# goal_xpos = state[36:39]

low = np.concatenate((hand_low, grip_low, obj_xyz_low, obj_quat_low, obj_xyz_low, obj_quat_low,
                      hand_low, grip_low, obj_xyz_low, obj_quat_low, obj_xyz_low, obj_quat_low,
                      goal_xyz_low), axis=0)
high = np.concatenate((hand_high, grip_high, obj_xyz_high, obj_quat_high, obj_xyz_high, obj_quat_high,
                       hand_high, grip_high, obj_xyz_high, obj_quat_high, obj_xyz_high, obj_quat_high,
                       goal_xyz_high), axis=0)

bounded_state_space = Box(low=low, high=high)


# # # # # # # # # #
# non-stationarity distribution constants
# # # # # # # # # #
ns_seq_length = 100_000
ns_test_inds = np.arange(0, ns_seq_length + 1, 2500, dtype=int)
