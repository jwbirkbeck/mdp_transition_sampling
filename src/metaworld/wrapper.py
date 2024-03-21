import gymnasium as gym
import metaworld
import mujoco
import numpy as np
from random import choice
from metaworld.envs import reward_utils


class MetaWorldWrapper(gym.Env):
    def __init__(self, task_name, render_mode=None):
        self.task_name = task_name
        self.ml1 = metaworld.ML1(self.task_name)
        self.render_mode = render_mode
        self.env = self.ml1.train_classes[self.task_name](render_mode=self.render_mode)
        self.task = choice(self.ml1.train_tasks)
        self.env.set_task(self.task)
        self.env._partially_observable = False

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.negate_rewards = False

        self.state = None
        self.prev_state = None
        self.reward = None
        self.done = None
        self.truncated = None
        self.info = None
        self.max_path_length = self.env.max_path_length
        self.curr_path_length = 0
        self.reward_bounds = (0, 10)  # As described in the meta-world paper

    def change_task(self, task_info=None, task_number=None):
        if task_info is not None:
            self.task_name = task_info
            self.ml1 = metaworld.ML1(self.task_name)
            self.env = self.ml1.train_classes[self.task_name](render_mode=self.render_mode)

        if task_number is not None:
            self.task = self.ml1.train_tasks[task_number]
        else:
            self.task = choice(self.ml1.train_tasks)

        self.env.set_task(self.task)
        self.env._partially_observable = False
        self.reset()

    def reset(self, seed=None):
        state, info = self.env.reset()
        self.env._partially_observable = False
        self.curr_path_length = 0
        self.state = state
        self.info = info
        return self.state, self.info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.negate_rewards:
            reward *= -1.0
        self.state = state
        self.reward = reward
        self.done = terminated
        self.truncated = truncated
        self.info = info
        self.curr_path_length = self.env.curr_path_length
        return self.state, self.reward, self.truncated, self.done, self.info

    def get_observation(self):
        # A compatibility method for wider testing
        return self.state

    def render(self):
        self.env.render()

    def new_task(self, env_name=None):
        if env_name is None:
            self.task = choice(self.ml1.train_tasks)
        else:
            self.ml1 = metaworld.ML1(env_name)
            self.env = self.ml1.train_classes[env_name]()
            self.task = choice(self.ml1.train_tasks)
        self.env.set_task(self.task)
        self.env._partially_observable = False

    def sample_reward_function(self, n_samples):
        reward_sample = np.random.uniform(low=self.reward_bounds[0], high=self.reward_bounds[1], size=n_samples)
        return reward_sample

    def sample_action(self, s_reward):
        s_action = self.env.action_space.sample()
        converged = True
        return s_action, converged

    def get_manual_step(self, state, action):
        self.set_internals_from_state(hand_xyz=state[0:3], obj_xyz=state[4:7], goal_xyz=state[36:39])
        next_state, reward, done, truncated, info = self.step(action=action)
        # end_eff_xpos = next_state[0:3]
        # obj1_xpos = next_state[4:7]
        # goal_xpos = next_state[36:39]
        # Rather than returning the full next state, we only need a subset of this info for manual reward calculation
        return next_state, reward, done, truncated, info

    def set_internals_from_state(self, hand_xyz=None, obj_xyz=None, goal_xyz=None):
        """
        While every Metaworld task uses the same states and action space, they each have differing internal
        methods for states-setting. This method stores all unique states-setting code for each task.
        """
        if self.task_name == 'button-press-topdown-v2' and goal_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env.obj_init_pos = goal_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            ] = self.env.obj_init_pos
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env._target_pos = self.env._get_site_pos("hole")
            self.env._obj_to_target_init = abs(
                self.env._target_pos[2] - self.env._get_site_pos("buttonStart")[2])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos  = old_obj_init_pos

        elif self.task_name == 'button-press-v2' and goal_xyz is not None:
            # See reset model in sawyer_button_press_v2.py
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            goal_pos = goal_xyz
            self.env.obj_init_pos = goal_pos
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz(0)
            self.env._target_pos = self.env._get_site_pos("hole")
            self.env.obj_init_pos = old_obj_init_pos
            self.env._obj_to_target_init = abs(
                self.env._target_pos[1] - self.env._get_site_pos("buttonStart")[1])
            self.env._set_pos_site('goal', self.env._target_pos)

        elif self.task_name == 'button-press-wall-v2' and goal_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            goal_pos = goal_xyz
            self.env.obj_init_pos = goal_pos
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz(0)
            self.env._target_pos = self.env._get_site_pos("hole")
            self.env._obj_to_target_init = abs(
                self.env._target_pos[1] - self.env._get_site_pos("buttonStart")[1])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'coffee-button-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.obj_init_pos = obj_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
            ] = self.env.obj_init_pos
            pos_mug = self.env.obj_init_pos + np.array([0.0, -0.22, 0.0])
            self.env._set_obj_xyz(pos_mug)
            pos_button = self.env.obj_init_pos + np.array([0.0, -0.22, 0.3])
            self.env._target_pos = pos_button + np.array([0.0, self.env.max_dist, 0.0])
            self.env.goal = self.env._target_pos
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'coffee-push-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.obj_init_pos = obj_xyz
            self.env._set_obj_xyz(obj_xyz)
            pos_machine = goal_xyz + np.array([0.0, 0.22, 0.0])
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
            ] = pos_machine
            self.env._target_pos = goal_xyz
            self.env.goal = self.env._target_pos
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'dial-turn-v2' and goal_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.prev_obs = self.env._get_curr_obs_combined_no_goal()
            goal_pos = goal_xyz
            self.env.obj_init_pos = goal_pos[:3]
            final_pos = goal_pos.copy() + np.array([0, 0.03, 0.03])
            self.env._target_pos = final_pos
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "dial")
            ] = self.env.obj_init_pos
            self.env.dial_push_position = self.env._get_pos_objects() + np.array([0.05, 0.02, 0.09])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos
            mujoco.mj_forward(self.env.model, self.env.data)

        elif self.task_name == 'door-close-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.objHeight = self.env.data.geom("handle").xpos[2]
            obj_pos = obj_xyz
            self.env.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy() + np.array([0.2, -0.2, 0.0])
            self.env._target_pos = goal_pos
            self.env.model.body("door").pos = self.env.obj_init_pos
            self.env.model.site("goal").pos = self.env._target_pos
            # keep the door open after resetting initial positions
            self.env._set_obj_xyz(-1.5708)
            self.env.goal = goal_xyz
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'door-unlock-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.model.body("door").pos = obj_xyz
            self.env._set_obj_xyz(1.5708)
            self.env.obj_init_pos = self.env.data.body("lock_link").xpos
            self.env._target_pos = self.env.obj_init_pos + np.array([0.1, -0.04, 0.0])
            self.env.goal = goal_xyz
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'drawer-close-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            # Compute nightstand position
            self.env.obj_init_pos = obj_xyz
            # Set mujoco body to computed position
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
            ] = self.env.obj_init_pos
            # Set _target_pos to current drawer position (closed)
            self.env._target_pos = self.env.obj_init_pos + np.array([0.0, -0.16, 0.09])
            # Pull drawer out all the way and mark its starting position
            self.env._set_obj_xyz(-self.env.maxDist)
            self.env.obj_init_pos = self.env._get_pos_objects()
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'handle-press-side-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.obj_init_pos = obj_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz(-0.001)
            self.env._target_pos = self.env._get_site_pos("goalPress")
            self.env._handle_init_pos = self.env._get_pos_objects()
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'handle-press-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.obj_init_pos = obj_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz(-0.001)
            self.env._target_pos = self.env._get_site_pos("goalPress")
            self.env.maxDist = np.abs(
                self.env.data.site_xpos[
                    mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, "handleStart")
                ][-1]
                - self.env._target_pos[-1]
            )
            self.env.target_reward = 1000 * self.env.maxDist + 1000 * 2
            self.env._handle_init_pos = self.env._get_pos_objects()
            self.env.goal = goal_xyz
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'peg-insert-side-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            pos_peg, pos_box = obj_xyz, goal_xyz
            self.env.obj_init_pos = pos_peg
            self.env.peg_head_pos_init = self.env._get_site_pos("pegHead")
            self.env._set_obj_xyz(self.env.obj_init_pos)
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            ] = pos_box
            self.env._target_pos = pos_box + np.array([0.03, 0.0, 0.13])
            self.env.goal = goal_xyz
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'plate-slide-back-side-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            puck_offset = self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_channel")
            ]
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = goal_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_goal")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz((obj_xyz - puck_offset)[:2])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'plate-slide-back-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            puck_offset = self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_channel")
            ]
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = goal_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_goal")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz((obj_xyz - puck_offset)[:2])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'plate-slide-side-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            puck_offset = self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_channel")
            ]
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = goal_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_goal")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz((obj_xyz - puck_offset)[:2])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'plate-slide-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            puck_offset = self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_channel")
            ]
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = goal_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "puck_goal")
            ] = self.env.obj_init_pos
            self.env._set_obj_xyz((obj_xyz - puck_offset)[:2])
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'push-back-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = self.env.adjust_initObjPos(obj_xyz)
            self.env.obj_init_angle = self.env.init_config["obj_init_angle"]
            self.env._set_obj_xyz(self.env.obj_init_pos)
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'reach-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = self.env.fix_extreme_obj_pos(obj_xyz)
            self.env.obj_init_angle = self.env.init_config["obj_init_angle"]
            self.env._set_obj_xyz(self.env.obj_init_pos)
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'reach-wall-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = obj_xyz
            self.obj_init_angle = self.env.init_config["obj_init_angle"]
            self.env._set_obj_xyz(self.env.obj_init_pos)
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'soccer-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_angle = self.env.init_config["obj_init_angle"]
            # goal_pos = self.env._get_state_rand_vec()
            # self.env._target_pos = goal_pos[3:]
            self.env.obj_init_pos = obj_xyz
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "goal_whole")
            ] = self.env._target_pos
            self.env._set_obj_xyz(self.env.obj_init_pos)
            self.env.maxPushDist = np.linalg.norm(
                self.env.obj_init_pos[:2] - np.array(self.env._target_pos)[:2]
            )
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'sweep-into-v2' and goal_xyz is not None and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.goal = goal_xyz
            self.env._target_pos = self.env.goal.copy()
            self.env.obj_init_pos = obj_xyz
            self.env.obj_init_angle = self.env.init_config["obj_init_angle"]
            self.env.objHeight = obj_xyz[2]
            # goal_pos = goal_xyz
            # self.env.obj_init_pos = np.concatenate((goal_pos[:2], [self.env.obj_init_pos[-1]]))
            self.env._set_obj_xyz(self.env.obj_init_pos)
            self.env.maxPushDist = np.linalg.norm(
                self.env.obj_init_pos[:2] - np.array(self.env._target_pos)[:2]
            )
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        elif self.task_name == 'window-open-v2' and obj_xyz is not None:
            old_obj_init_pos = self.env.obj_init_pos
            self.env.prev_obs = self.env._get_curr_obs_combined_no_goal()
            self.obj_init_pos = obj_xyz
            self.env._target_pos = self.env.obj_init_pos + np.array([0.2, 0.0, 0.0])
            self.env.model.body_pos[
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "window")
            ] = self.env.obj_init_pos
            self.env.window_handle_pos_init = self.env._get_pos_objects()
            self.env.data.joint("window_slide").qpos = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env._set_pos_site('goal', self.env._target_pos)
            self.env.obj_init_pos = old_obj_init_pos

        else:
            raise NotImplementedError('custom methods not defined for this task, or missing goal and object inputs')

        if hand_xyz is not None:
            old_hand_init_pos = self.env.hand_init_pos
            old_init_tcp = self.env.init_tcp
            self.env.hand_init_pos = hand_xyz
            self.env._reset_hand()
            self.env.hand_init_pos = old_hand_init_pos
            self.env.init_tcp = old_init_tcp

    def compute_reward_wrap(self, states, actions):
        """
        Each Metaworld task has a unique reward function, but their calculation relies on mujoco internal values. This
        method replicates their calculations from the states and actions inputs. Small errors are possible - for instance,
        the constant adjustment of np.array([0.0, 0.0, 0.045]) is only an estimate of a mildly (~1e-4) varying difference.
        """
        rewards = []
        if self.task_name == 'button-press-topdown-v2':
            #         hand_low = (-0.5, 0.40, 0.05)
            #         hand_high = (0.5, 1, 0.5)
            #         obj_low = (-0.1, 0.8, 0.115)
            #         obj_high = (0.1, 0.9, 0.115)
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                # the tcp center is below the states provided to the agent. This is a constant approximate adjustment
                # from the wrist joint to the COM of the fingers.
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])

                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(obj - self.env.init_tcp)
                obj_to_target = abs(self.env._target_pos[2] - obj[2])

                tcp_closed = 1 - obs[3]
                near_button = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, 0.01),
                    margin=tcp_to_obj_init,
                    sigmoid="long_tail",
                )
                button_pressed = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, 0.005),
                    margin=self.env._obj_to_target_init,
                    sigmoid="long_tail",
                )

                reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
                if tcp_to_obj <= 0.03:
                    reward += 5 * button_pressed
                rewards.append(reward)

        elif self.task_name == 'button-press-v2':
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])

                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(obj - self.env.init_tcp)
                obj_to_target = abs(self.env._target_pos[1] - obj[1])

                tcp_closed = max(obs[3], 0.0)
                near_button = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, 0.05),
                    margin=tcp_to_obj_init,
                    sigmoid="long_tail",
                )
                button_pressed = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, 0.005),
                    margin=self.env._obj_to_target_init,
                    sigmoid="long_tail",
                )

                reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
                if tcp_to_obj <= 0.05:
                    reward += 8 * button_pressed
                rewards.append(reward)

        elif self.task_name == 'button-press-wall-v2':
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])

                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(obj - self.env.init_tcp)
                obj_to_target = abs(self.env._target_pos[1] - obj[1])

                near_button = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, 0.01),
                    margin=tcp_to_obj_init,
                    sigmoid="long_tail",
                )
                button_pressed = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, 0.005),
                    margin=self.env._obj_to_target_init,
                    sigmoid="long_tail",
                )

                reward = 0.0
                if tcp_to_obj > 0.07:
                    tcp_status = (1 - obs[3]) / 2.0
                    reward = 2 * reward_utils.hamacher_product(tcp_status, near_button)
                else:
                    reward = 2
                    reward += 2 * (1 + obs[3])
                    reward += 4 * button_pressed ** 2
                rewards.append(reward)

        elif self.task_name == 'coffee-button-v2':
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])

                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(obj - self.env.init_tcp)
                obj_to_target = abs(self.env._target_pos[1] - obj[1])

                tcp_closed = max(obs[3], 0.0)
                near_button = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, 0.05),
                    margin=tcp_to_obj_init,
                    sigmoid="long_tail",
                )
                button_pressed = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, 0.005),
                    margin=self.env.max_dist,
                    sigmoid="long_tail",
                )

                reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
                if tcp_to_obj <= 0.05:
                    reward += 8 * button_pressed
                rewards.append(reward)

        elif self.task_name == 'coffee-push-v2':
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                target = self.env._target_pos.copy()
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])

                # Emphasize X and Y errors
                scale = np.array([2.0, 2.0, 1.0])
                target_to_obj = (obj - target) * scale
                target_to_obj = np.linalg.norm(target_to_obj)
                target_to_obj_init = (self.env.obj_init_pos - target) * scale
                target_to_obj_init = np.linalg.norm(target_to_obj_init)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, 0.05),
                    margin=target_to_obj_init,
                    sigmoid="long_tail",
                )
                tcp_opened = obs[3]
                tcp_to_obj = np.linalg.norm(obj - tcp)

                object_grasped = self.env._gripper_caging_reward(
                    act,
                    obj,
                    object_reach_radius=0.04,
                    obj_radius=0.02,
                    pad_success_thresh=0.05,
                    xz_thresh=0.05,
                    desired_gripper_effort=0.7,
                    medium_density=True,
                )

                reward = reward_utils.hamacher_product(object_grasped, in_place)

                if tcp_to_obj < 0.04 and tcp_opened > 0:
                    reward += 1.0 + 5.0 * in_place
                if target_to_obj < 0.05:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'dial-turn-v2':
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                dial_push_position = obs[4:7] + np.array([0.05, 0.02, 0.09])
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                target = self.env._target_pos.copy()

                target_to_obj = obj - target
                target_to_obj = np.linalg.norm(target_to_obj)
                target_to_obj_init = self.env.dial_push_position - target
                target_to_obj_init = np.linalg.norm(target_to_obj_init)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=abs(target_to_obj_init - self.env.TARGET_RADIUS),
                    sigmoid="long_tail",
                )

                dial_reach_radius = 0.005
                tcp_to_obj = np.linalg.norm(dial_push_position - tcp)
                tcp_to_obj_init = np.linalg.norm(self.env.dial_push_position - self.env.init_tcp)
                reach = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, dial_reach_radius),
                    margin=abs(tcp_to_obj_init - dial_reach_radius),
                    sigmoid="gaussian",
                )
                gripper_closed = min(max(0, act[-1]), 1)
                reach = reward_utils.hamacher_product(reach, gripper_closed)
                reward = 10 * reward_utils.hamacher_product(reach, in_place)
                rewards.append(reward)

        elif self.task_name == 'door-close-v2':
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                obj = obs[4:7]
                target = self.env._target_pos

                tcp_to_target = np.linalg.norm(tcp - target)
                obj_to_target = np.linalg.norm(obj - target)

                in_place_margin = np.linalg.norm(self.env.obj_init_pos - target)
                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin,
                    sigmoid="gaussian",
                )

                hand_margin = np.linalg.norm(self.env.hand_init_pos - obj) + 0.1
                hand_in_place = reward_utils.tolerance(
                    tcp_to_target,
                    bounds=(0, 0.25 * _TARGET_RADIUS),
                    margin=hand_margin,
                    sigmoid="gaussian",
                )

                reward = 3 * hand_in_place + 6 * in_place

                if obj_to_target < _TARGET_RADIUS:
                    reward = 10
                rewards.append(reward)

        elif self.task_name == 'door-unlock-v2':
            for obs, act in zip(states, actions):
                gripper = obs[:3]
                lock = obs[4:7]

                # Add offset to track gripper's shoulder, rather than fingers
                offset = np.array([0.0, 0.055, 0.07])

                scale = np.array([0.25, 1.0, 0.5])
                shoulder_to_lock = (gripper + offset - lock) * scale
                shoulder_to_lock_init = (self.env.init_tcp + offset - self.env.obj_init_pos) * scale

                # This `ready_to_push` reward should be a *hint* for the agent, not an
                # end in itself. Make sure to devalue it compared to the value of
                # actually unlocking the lock
                ready_to_push = reward_utils.tolerance(
                    np.linalg.norm(shoulder_to_lock),
                    bounds=(0, 0.02),
                    margin=np.linalg.norm(shoulder_to_lock_init),
                    sigmoid="long_tail",
                )

                obj_to_target = abs(self.env._target_pos[0] - lock[0])
                pushed = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, 0.005),
                    margin=self.env._lock_length,
                    sigmoid="long_tail",
                )

                reward = 2 * ready_to_push + 8 * pushed
                rewards.append(reward)

        elif self.task_name == 'drawer-close-v2':
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                target = self.env._target_pos.copy()

                target_to_obj = obj - target
                target_to_obj = np.linalg.norm(target_to_obj)
                target_to_obj_init = self.env.obj_init_pos - target
                target_to_obj_init = np.linalg.norm(target_to_obj_init)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=abs(target_to_obj_init - self.env.TARGET_RADIUS),
                    sigmoid="long_tail",
                )

                handle_reach_radius = 0.005
                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(self.env.obj_init_pos - self.env.init_tcp)
                reach = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, handle_reach_radius),
                    margin=abs(tcp_to_obj_init - handle_reach_radius),
                    sigmoid="gaussian",
                )
                gripper_closed = min(max(0, act[-1]), 1)

                reach = reward_utils.hamacher_product(reach, gripper_closed)

                reward = reward_utils.hamacher_product(reach, in_place)
                if target_to_obj <= self.env.TARGET_RADIUS + 0.015:
                    reward = 1.0

                reward *= 10
                rewards.append(reward)

        elif self.task_name == 'handle-press-side-v2':
            rewards = []
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                target = self.env._target_pos.copy()

                target_to_obj = obj[2] - target[2]
                target_to_obj = np.linalg.norm(target_to_obj)
                target_to_obj_init = self.env._handle_init_pos[2] - target[2]
                target_to_obj_init = np.linalg.norm(target_to_obj_init)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=abs(target_to_obj_init - self.env.TARGET_RADIUS),
                    sigmoid="long_tail",
                )

                handle_radius = 0.02
                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(self.env._handle_init_pos - self.env.init_tcp)
                reach = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, handle_radius),
                    margin=abs(tcp_to_obj_init - handle_radius),
                    sigmoid="long_tail",
                )

                reward = reward_utils.hamacher_product(reach, in_place)
                reward = 1 if target_to_obj <= self.env.TARGET_RADIUS else reward
                reward *= 10
                rewards.append(reward)

        elif self.task_name == 'handle-press-v2':
            rewards = []
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                target = self.env._target_pos.copy()

                target_to_obj = obj[2] - target[2]
                target_to_obj = np.linalg.norm(target_to_obj)
                target_to_obj_init = self.env._handle_init_pos[2] - target[2]
                target_to_obj_init = np.linalg.norm(target_to_obj_init)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=abs(target_to_obj_init - self.env.TARGET_RADIUS),
                    sigmoid="long_tail",
                )

                handle_radius = 0.02
                tcp_to_obj = np.linalg.norm(obj - tcp)
                tcp_to_obj_init = np.linalg.norm(self.env._handle_init_pos - self.env.init_tcp)
                reach = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, handle_radius),
                    margin=abs(tcp_to_obj_init - handle_radius),
                    sigmoid="long_tail",
                )
                tcp_opened = 0
                object_grasped = reach

                reward = reward_utils.hamacher_product(reach, in_place)
                reward = 1 if target_to_obj <= self.env.TARGET_RADIUS else reward
                reward *= 10
                rewards.append(reward)

        elif self.task_name == 'peg-insert-side-v2':
            rewards = []
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                obj_head = self.env._get_site_pos("pegHead")
                tcp_opened = obs[3]
                target = self.env._target_pos
                tcp_to_obj = np.linalg.norm(obj - tcp)
                scale = np.array([1.0, 2.0, 2.0])
                #  force agent to pick up object then insert
                obj_to_target = np.linalg.norm((obj_head - target) * scale)

                in_place_margin = np.linalg.norm((self.env.peg_head_pos_init - target) * scale)
                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=in_place_margin,
                    sigmoid="long_tail",
                )
                ip_orig = in_place
                brc_col_box_1 = self.env._get_site_pos("bottom_right_corner_collision_box_1")
                tlc_col_box_1 = self.env._get_site_pos("top_left_corner_collision_box_1")

                brc_col_box_2 = self.env._get_site_pos("bottom_right_corner_collision_box_2")
                tlc_col_box_2 = self.env._get_site_pos("top_left_corner_collision_box_2")
                collision_box_bottom_1 = reward_utils.rect_prism_tolerance(
                    curr=obj_head, one=tlc_col_box_1, zero=brc_col_box_1
                )
                collision_box_bottom_2 = reward_utils.rect_prism_tolerance(
                    curr=obj_head, one=tlc_col_box_2, zero=brc_col_box_2
                )
                collision_boxes = reward_utils.hamacher_product(
                    collision_box_bottom_2, collision_box_bottom_1
                )
                in_place = reward_utils.hamacher_product(in_place, collision_boxes)

                pad_success_margin = 0.03
                object_reach_radius = 0.01
                x_z_margin = 0.005
                obj_radius = 0.0075

                object_grasped = self.env._gripper_caging_reward(
                    act,
                    obj,
                    object_reach_radius=object_reach_radius,
                    obj_radius=obj_radius,
                    pad_success_thresh=pad_success_margin,
                    xz_thresh=x_z_margin,
                    high_density=True,
                )
                if (
                        tcp_to_obj < 0.08
                        and (tcp_opened > 0)
                        and (obj[2] - 0.01 > self.env.obj_init_pos[2])
                ):
                    object_grasped = 1.0
                in_place_and_object_grasped = reward_utils.hamacher_product(
                    object_grasped, in_place
                )
                reward = in_place_and_object_grasped

                if (
                        tcp_to_obj < 0.08
                        and (tcp_opened > 0)
                        and (obj[2] - 0.01 > self.env.obj_init_pos[2])
                ):
                    reward += 1.0 + 5 * in_place

                if obj_to_target <= 0.07:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'plate-slide-back-side-v2':
            rewards = []
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                target = self.env._target_pos

                obj_to_target = np.linalg.norm(obj - target)
                in_place_margin = np.linalg.norm(self.env.obj_init_pos - target)
                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin - _TARGET_RADIUS,
                    sigmoid="long_tail",
                )

                tcp_to_obj = np.linalg.norm(tcp - obj)
                obj_grasped_margin = np.linalg.norm(self.env.init_tcp - self.env.obj_init_pos)
                object_grasped = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, _TARGET_RADIUS),
                    margin=obj_grasped_margin - _TARGET_RADIUS,
                    sigmoid="long_tail",
                )

                reward = 1.5 * object_grasped

                if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
                    reward = 2 + (7 * in_place)

                if obj_to_target < _TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'plate-slide-back-v2':
            rewards = []
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                target = self.env._target_pos

                obj_to_target = np.linalg.norm(obj - target)
                in_place_margin = np.linalg.norm(self.env.obj_init_pos - target)
                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin - _TARGET_RADIUS,
                    sigmoid="long_tail",
                )

                tcp_to_obj = np.linalg.norm(tcp - obj)
                obj_grasped_margin = np.linalg.norm(self.env.init_tcp - self.env.obj_init_pos)
                object_grasped = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, _TARGET_RADIUS),
                    margin=obj_grasped_margin - _TARGET_RADIUS,
                    sigmoid="long_tail",
                )

                reward = 1.5 * object_grasped

                if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
                    reward = 2 + (7 * in_place)

                if obj_to_target < _TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'plate-slide-side-v2':
            rewards = []
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                target = self.env._target_pos

                obj_to_target = np.linalg.norm(obj - target)
                in_place_margin = np.linalg.norm(self.env.obj_init_pos - target)
                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin - _TARGET_RADIUS,
                    sigmoid="long_tail",
                )

                tcp_to_obj = np.linalg.norm(tcp - obj)
                obj_grasped_margin = np.linalg.norm(self.env.init_tcp - self.env.obj_init_pos)
                object_grasped = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, _TARGET_RADIUS),
                    margin=obj_grasped_margin - _TARGET_RADIUS,
                    sigmoid="long_tail",
                )

                # in_place_and_object_grasped = reward_utils.hamacher_product(
                #     object_grasped, in_place
                # )
                reward = 1.5 * object_grasped

                if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
                    reward = 2 + (7 * in_place)

                if obj_to_target < _TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'plate-slide-v2':
            rewards = []
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                target = self.env._target_pos

                obj_to_target = np.linalg.norm(obj - target)
                in_place_margin = np.linalg.norm(self.env.obj_init_pos - target)

                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin,
                    sigmoid="long_tail",
                )

                tcp_to_obj = np.linalg.norm(tcp - obj)
                obj_grasped_margin = np.linalg.norm(self.env.init_tcp - self.env.obj_init_pos)

                object_grasped = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, _TARGET_RADIUS),
                    margin=obj_grasped_margin,
                    sigmoid="long_tail",
                )

                in_place_and_object_grasped = reward_utils.hamacher_product(
                    object_grasped, in_place
                )
                reward = 8 * in_place_and_object_grasped

                if obj_to_target < _TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'push-back-v2':
            rewards = []
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                tcp_to_obj = np.linalg.norm(obj - tcp)
                target_to_obj = np.linalg.norm(obj - self.env._target_pos)
                target_to_obj_init = np.linalg.norm(self.env.obj_init_pos - self.env._target_pos)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=target_to_obj_init,
                    sigmoid="long_tail",
                )
                object_grasped = self.env._gripper_caging_reward(act, obj, self.env.OBJ_RADIUS)

                reward = reward_utils.hamacher_product(object_grasped, in_place)

                if (
                        (tcp_to_obj < 0.01)
                        and (0 < tcp_opened < 0.55)
                        and (target_to_obj_init - target_to_obj > 0.01)
                ):
                    reward += 1.0 + 5.0 * in_place
                if target_to_obj < self.env.TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'reach-v2':
            rewards = []
            for obs in states:
                _TARGET_RADIUS = 0.05
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                obj = obs[3:6]
                # tcp_opened = obs[3]
                # target = obs[6:9]
                target = self.env._target_pos

                tcp_to_target = np.linalg.norm(tcp - target)
                # obj_to_target = np.linalg.norm(obj - target)
                in_place_margin = (np.linalg.norm(np.array([0., 0.6, 0.2]) - target))
                in_place = reward_utils.tolerance(tcp_to_target,
                                                  bounds=(0, _TARGET_RADIUS),
                                                  margin=in_place_margin,
                                                  sigmoid='long_tail', )
                rewards.append(10.0 * in_place)

        elif self.task_name == 'reach-wall-v2':
            rewards = []
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                # obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                # tcp_opened = obs[3]
                target = self.env._target_pos

                tcp_to_target = np.linalg.norm(tcp - target)
                # obj_to_target = np.linalg.norm(obj - target)

                in_place_margin = np.linalg.norm(self.env.hand_init_pos - target)
                in_place = reward_utils.tolerance(
                    tcp_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin,
                    sigmoid="long_tail",
                )
                reward = 10 * in_place
                rewards.append(reward)

        elif self.task_name == 'soccer-v2':
            rewards = []
            for obs, act in zip(states, actions):
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                x_scaling = np.array([3.0, 1.0, 1.0])
                tcp_to_obj = np.linalg.norm(obj - tcp)
                target_to_obj = np.linalg.norm((obj - self.env._target_pos) * x_scaling)
                target_to_obj_init = np.linalg.norm((obj - self.env.obj_init_pos) * x_scaling)

                in_place = reward_utils.tolerance(
                    target_to_obj,
                    bounds=(0, self.env.TARGET_RADIUS),
                    margin=target_to_obj_init,
                    sigmoid="long_tail",
                )

                goal_line = self.env._target_pos[1] - 0.1
                if obj[1] > goal_line and abs(obj[0] - self.env._target_pos[0]) > 0.10:
                    in_place = np.clip(
                        in_place - 2 * ((obj[1] - goal_line) / (1 - goal_line)), 0.0, 1.0
                    )

                object_grasped = self.env._gripper_caging_reward(act, obj, self.env.OBJ_RADIUS)

                reward = (3 * object_grasped) + (6.5 * in_place)

                if target_to_obj < self.env.TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        elif self.task_name == 'sweep-into-v2':
            rewards = []
            for obs, act in zip(states, actions):
                _TARGET_RADIUS = 0.05
                obj = obs[4:7]
                tcp = obs[0:3] - np.array([0.0, 0.0, 0.045])
                tcp_opened = obs[3]
                target = np.array([self.env._target_pos[0], self.env._target_pos[1], obj[2]])

                obj_to_target = np.linalg.norm(obj - target)
                tcp_to_obj = np.linalg.norm(obj - tcp)
                in_place_margin = np.linalg.norm(self.env.obj_init_pos - target)

                in_place = reward_utils.tolerance(
                    obj_to_target,
                    bounds=(0, _TARGET_RADIUS),
                    margin=in_place_margin,
                    sigmoid="long_tail",
                )

                object_grasped = self.env._gripper_caging_reward(act, obj, self.env.OBJ_RADIUS)
                in_place_and_object_grasped = reward_utils.hamacher_product(
                    object_grasped, in_place
                )

                reward = (2 * object_grasped) + (6 * in_place_and_object_grasped)

                if obj_to_target < _TARGET_RADIUS:
                    reward = 10.0
                rewards.append(reward)

        else:
            raise NotImplementedError('custom methods not defined for this task')

        return rewards