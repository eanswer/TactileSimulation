import os
import sys
from envs.redmax_torch_functions import StepSimFunction

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import numpy as np
from utils import math
from envs.redmax_torch_env import RedMaxTorchEnv
from scipy.spatial.transform import Rotation
from utils.common import *
import utils.torch_utils as tu
from gym import spaces
import torch
import cv2
import time

class TactilePushEnv(RedMaxTorchEnv):
    def __init__(self, use_torch = False, gradient = False, observation_type = "tactile_map", render_tactile = False, verbose = False, seed = 0):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        model_path = os.path.join(asset_folder, 'pusher/pusher.xml')

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose
        self.render_tactile = render_tactile
        self.dtype = torch.double

        self.tactile_rows = 13
        self.tactile_cols = 10

        if self.observation_type == "tactile_flatten":
            self.tactile_obs_shape = (self.tactile_rows * self.tactile_cols * 3, )
            self.state_obs_shape = (3, )
        elif self.observation_type == "tactile_map":
            self.tactile_obs_shape = (3, self.tactile_rows, self.tactile_cols)
            self.state_obs_shape = (3, )
            self.mixed_observation_space = (self.tactile_obs_shape, self.state_obs_shape)
        elif self.observation_type == "privilege":
            self.tactile_obs_shape = (1, 1)
            self.state_obs_shape = (6, )
        elif self.observation_type == "no_tactile":
            self.tactile_obs_shape = (1, 1)
            self.state_obs_shape = (3, )
        else:
            raise NotImplementedError
        
        self.tactile_force_buf = torch.zeros((1, 1, self.tactile_rows, self.tactile_cols, 3), dtype = self.dtype)
        self.tactile_obs_buf = torch.zeros(self.tactile_obs_shape, dtype = self.dtype)
        self.state_q = torch.zeros(7, dtype = self.dtype)
        self.goal = torch.zeros(3, dtype = self.dtype)
        self.tactile_force_his = []
        self.tactile_obs_his = []
        self.state_obs_buf = torch.zeros(self.state_obs_shape, dtype = self.dtype)
        self.reward_buf = 0.
        self.done_buf = False
        self.gradient = gradient
        self.info_buf = {}
        self.external_force = np.zeros(2)
        self.current_step = 0

        super(TactilePushEnv, self).__init__(model_path, seed = seed)

        self.frame_skip = 5
        self.dt = self.sim.options.h * self.frame_skip

        self.ndof_u = 3
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)
    
    def _get_obs(self):
        # construct tactile obs
        self.tactile_obs_buf = self.tactile_force_buf.clone()
        
        if self.observation_type == "tactile_flatten":
            self.tactile_obs_buf = self.tactile_obs_buf.reshape(-1)
        elif self.observation_type == "tactile_map":
            self.tactile_obs_buf = self.tactile_obs_buf \
                            .permute(0, 1, 4, 2, 3) \
                            .reshape(-1, self.tactile_rows, self.tactile_cols)

        # construct state obs
        gripper_rot = self.state_q[0:1]
        gripper_pos_local = self.state_q[1:3]
        object_pos_world = self.state_q[3:5]
        object_rot = self.state_q[6:7]
        goal_pos_world = self.goal[0:2]
        goal_rot = self.goal[2:3]
        # convert to gripper local frame
        c, s = torch.cos(-gripper_rot), torch.sin(-gripper_rot)
        rot_mat = torch.zeros((2, 2), dtype = self.dtype)
        rot_mat[0, 0] = c
        rot_mat[0, 1] = -s
        rot_mat[1, 0] = s
        rot_mat[1, 1] = c
        
        object_pos_local = rot_mat @ object_pos_world - gripper_pos_local
        object_rot_local = object_rot - gripper_rot
        goal_pos_local = rot_mat @ goal_pos_world - gripper_pos_local
        goal_rot_local = goal_rot - gripper_rot

        if self.observation_type == "privilege":
            self.state_obs_buf = torch.cat([object_pos_local, object_rot_local, goal_pos_local, goal_rot_local])
        else:
            self.state_obs_buf = torch.cat([goal_pos_local, goal_rot_local])
            
        if self.observation_type == 'privilege':
            obs = self.state_obs_buf
            if not self.use_torch:
                return obs.detach().cpu().numpy()
            else:
                return obs
        elif self.observation_type == "tactile_flatten":
            obs = torch.cat([self.state_obs_buf, self.tactile_obs_buf])
            if not self.use_torch:
                return obs.detach().cpu().numpy()
            else:
                return obs
        elif self.observation_type == "tactile_map":
            obs = self.state_obs_buf
            if not self.use_torch:
                return (self.tactile_obs_buf.detach().cpu().numpy(), obs.detach().cpu().numpy())
            else:
                return (self.tactile_obs_buf, obs)
        elif self.observation_type == "no_tactile":
            obs = self.state_obs_buf
            if not self.use_torch:
                return obs.detach().cpu().numpy()
            else:
                return obs
    
    def reset(self):
        self.state_q = torch.tensor(self.sim.get_q_init().copy(), dtype = self.dtype)
        self.state_q[1] = -0.001
        self.state_q[4] = self.np_random.uniform(low = -0.02, high = 0.02)

        self.sim.set_q_init(self.state_q.numpy())

        self.robot_rot_prev = 0.

        # randomize the goal
        self.goal = torch.zeros(3, dtype = self.dtype)
        goal_xy = self.np_random.uniform(low = [0.15, -0.2], high = [0.25, 0.2])
        self.goal[0:2] = torch.tensor(goal_xy, dtype = self.dtype)
        self.goal[2] = torch.tensor(self.np_random.uniform(low = goal_xy[1] * np.pi - np.pi / 16., high = goal_xy[1] * np.pi + np.pi / 16.), dtype = self.dtype)
        
        goal_pos = np.array([self.goal[0] + 0.05, self.goal[1], 0.025])
        goal_quat = Rotation.from_rotvec([0., 0., self.goal[2]]).as_quat()
        goal_quat = [goal_quat[3], goal_quat[0], goal_quat[1], goal_quat[2]]

        self.sim.update_virtual_object("goal", np.concatenate([goal_pos, goal_quat]))

        self.sim.reset(backward_flag = self.gradient)
        
        # compute the initial obs
        tactiles = self.sim.get_tactile_force_vector()

        self.tactile_force_buf = torch.tensor(tactiles.reshape(1, 1, self.tactile_rows, self.tactile_cols, 3))

        self.prev_error_pos = self.state_q[3:5] - self.goal[0:2]
        self.prev_error_rot = self.state_q[6] - self.goal[2]

        obs = self._get_obs()

        self.external_force = np.zeros(2)
        self.current_step = 0

        self.tactile_force_his = []
        self.tactile_obs_his = []

        return obs
    
    def step(self, u):
        if not self.use_torch:
            u = torch.tensor(u)
        action = torch.tanh(u)
        
        robot_action = torch.zeros(6, dtype = self.dtype)
        robot_action[0] = action[0]
        robot_action[1] = action[1]
        robot_action[2] = action[2]

        # apply random force
        if self.current_step % 10 == 0:
            random_force_p = self.np_random.uniform(low = 0, high = 1.)
            if random_force_p < 0.5:
                self.external_force = self.np_random.uniform(-1., 1., 2)
            else:
                self.external_force = np.zeros(2)
        
        robot_action[3] = self.external_force[0]
        robot_action[4] = self.external_force[1]
        
        q, var, tactiles = StepSimFunction.apply(robot_action, self.frame_skip, self.sim, self.gradient)
        
        self.state_q = q
        self.tactile_force_buf = tactiles.reshape(1, 1, self.tactile_rows, self.tactile_cols, 3)
        
        obs = self._get_obs()
        
        # compute reward
        object_pos = q[3:5]
        object_rot = q[6]

        reward_pos = -torch.sum(((object_pos - self.goal[0:2]) / 0.01) ** 2) * 0.01
        reward_rot = -(((object_rot - self.goal[2]) / (np.pi / 36.)) ** 2) * 0.1
        reward_touch = -torch.sum((var[0:3] - var[3:6]) ** 2) / (0.02 ** 2)
        reward_action = - torch.sum(u ** 2) * 0.1

        reward = reward_pos + reward_rot + reward_touch + reward_action

        if not self.use_torch:
            reward = reward.detach().cpu().numpy()

        # construct info
        info = {}
        info['reward_pos'] = reward_pos
        info['reward_rot'] = reward_rot
        info['reward_touch'] = reward_touch
        info['reward_action'] = reward_action
        info['final_pos_error'] = torch.norm((object_pos - self.goal[0:2]))
        info['final_rot_error'] = torch.abs(object_rot - self.goal[2])

        # append tactile his
        if self.render_tactile:
            self.tactile_force_his.append(self.get_tactile_forces_array())
            self.tactile_obs_his.append(self.get_tactile_obs_array())

        # print(obs)

        return obs, reward, False, info
    
    def visualize_tactile(self, tactile_array):
        resolution = 20
        horizontal_space = 20
        vertical_space = 40
        T = len(tactile_array)
        nrows = tactile_array.shape[2]
        ncols = tactile_array.shape[3]

        imgs_tactile = np.zeros((ncols * resolution + vertical_space * 2, nrows * resolution * T + horizontal_space * (T + 1), 3), dtype = float)

        for timestep in range(T):
            for finger_idx in range(1):
                for row in range(nrows):
                    for col in range(ncols):
                        loc0_x = row * resolution + resolution // 2 + timestep * nrows * resolution + timestep * horizontal_space + horizontal_space
                        loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                        loc1_x = loc0_x + tactile_array[timestep][finger_idx][row, col][0]
                        loc1_y = loc0_y + tactile_array[timestep][finger_idx][row, col][1]
                        depth_ratio = min(1., np.abs(tactile_array[timestep][finger_idx][row, col][2]) / 200.)
                        color = (0.0, 1.0 - depth_ratio, depth_ratio)
                        cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)), color, 2, tipLength = 0.3)
        
        return imgs_tactile

    def render(self, mode = 'once'):
        if self.render_tactile:
            for t in range(len(self.tactile_force_his)):
                img_tactile_clean = self.visualize_tactile(self.tactile_force_his[t])
                img_tactile_noise = self.visualize_tactile(self.tactile_obs_his[t])
                img_tactile = np.zeros((img_tactile_clean.shape[0] + img_tactile_noise.shape[0], img_tactile_clean.shape[1], 3))
                img_tactile[0:img_tactile_clean.shape[0], :] = img_tactile_clean
                img_tactile[img_tactile_noise.shape[0]:, :] = img_tactile_noise
                cv2.imshow("tactile", img_tactile)
                cv2.waitKey(1)

                time.sleep(0.05)

        super().render(mode) 

    # return tactile obs array: shape (T, 2, nrows, ncols, 2)
    def get_tactile_obs_array(self):
        if self.observation_type == 'tactile_flatten' or self.observation_type == 'privilege' or self.observation_type == "tactile_flatten_his" or self.observation_type == "no_tactile":
            tactile_obs = self.tactile_obs_buf.reshape(1, 1, self.tactile_rows, self.tactile_cols, 3)
        elif self.observation_type == 'tactile_map':
            tactile_obs = self.tactile_obs_buf.reshape(1, 1, 3, self.tactile_rows, self.tactile_cols) \
                                .permute(0, 1, 3, 4, 2)
        
        return tactile_obs.detach().cpu().numpy()

    def get_tactile_forces_array(self):
        tactile_force_array = self.tactile_force_buf.clone().detach().cpu().numpy()
        tactile_force_array[...,0:2] = tactile_force_array[...,0:2] / 0.000003
        tactile_force_array[...,2:3] = tactile_force_array[...,2:3] / 0.003
        return tactile_force_array