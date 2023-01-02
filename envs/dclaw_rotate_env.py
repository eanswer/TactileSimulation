import os
from re import I
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
from envs.redmax_torch_env import RedMaxTorchEnv
from utils.common import *
from gym import spaces
import torch
import cv2
import time

class DClawRotateEnv(RedMaxTorchEnv):
    def __init__(self, use_torch = False, observation_type = "tactile",
                 render_tactile = True, verbose = False, seed = 0,
                 torque_control = False, relative_control=True):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        self.is_torque_control = torque_control
        self.relative_control = relative_control
        self.relative_q_scale = 0.06
        self.rot_coef = 1.0
        self.power_coef = 0.005
        self.has_been_reset = False
        if torque_control:
            model_path = os.path.join(asset_folder, 'dclaw_rotate/dclaw_torque_control.xml')
        else:
            model_path = os.path.join(asset_folder, 'dclaw_rotate/dclaw_position_control.xml')

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose
        self.render_tactile = render_tactile

        self.tactile_rows = 20
        self.tactile_cols = 20

        if self.observation_type == "tactile_flatten":
            self.tactile_obs_shape = (self.tactile_rows * self.tactile_cols * 3 * 3, )
        elif self.observation_type == "tactile":
            self.tactile_obs_shape = (3 * 3, self.tactile_rows, self.tactile_cols)
        elif self.observation_type == "no_tactile":
            self.tactile_obs_shape = (18, )
        else:
            raise NotImplementedError

        self.tactile_force_buf = torch.zeros(self.tactile_obs_shape)
        self.tactile_obs_buf = torch.zeros(self.tactile_obs_shape)
        self.reward_buf = 0.
        self.done_buf = False
        self.tactile_force_his = []
        self.tactile_obs_his = []
        self.info_buf = {}

        super(DClawRotateEnv, self).__init__(model_path, seed = seed)

        self.frame_skip = 5
        self.dt = self.sim.options.h * self.frame_skip
        print(f'Dt:{self.dt}')

        self.ndof_u = 9
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)

        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = (np.array([4., -5., 4]) - self.sim.viewer_options.camera_lookat) / 1.8 + self.sim.viewer_options.camera_lookat

        image_poses = self.sim.get_tactile_image_pos('one3_link_fingertip')
        self.tactile_mask = np.full((20, 20), 0, dtype = bool)
        for i in range(len(image_poses)):
            self.tactile_mask[image_poses[i][0]][image_poses[i][1]] = True

        self.q_init = self.sim.get_q_init().copy()
        print(f'Q_init:{self.q_init}')
        self.q_init[1], self.q_init[4], self.q_init[7] = -0.5, -0.5, -0.5
        self.q_init[2], self.q_init[5], self.q_init[8] = 0.8, 0.8, 0.8
        self.dof_limit = np.array([
            [-0.45, 1.35],
            [-2, 2],
            [1, 2],
            [-0.45, 1.35],
            [-2, 2],
            [1, 2],
            [-0.45, 1.35],
            [-2, 2],
            [1, 2],
        ])
        self.sim.set_q_init(self.q_init)
        self.cap_top_surface_z = 0.05
        self.cap_center = np.array([0, 0, 0.035])
    
    def _get_obs(self):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()
        hand_dofs = q[:9].copy() # joint angles of the hand
        
        variables = self.sim.get_variables().copy() # the variables contains the positions of three finger tips [0:3], [3:6], [6:9]
        fingertip_pos_world = variables[:9]
        
        if self.observation_type == "no_tactile":
            state = np.concatenate((q[:9], fingertip_pos_world))
        else:
            tactiles = np.array(self.sim.get_tactile_flow_images())

            self.tactile_force_buf = torch.tensor(tactiles) # (3, 20, 20, 3)

            self.tactile_obs_buf = self.tactile_force_buf.clone()

            if self.observation_type == "tactile_flatten":
                self.tactile_obs_buf = self.tactile_obs_buf.reshape(-1)
            elif self.observation_type == "tactile":
                self.tactile_obs_buf = self.tactile_obs_buf \
                                .permute(0, 3, 1, 2) \
                                .reshape(-1, self.tactile_rows, self.tactile_cols)

            obs = self.tactile_obs_buf
        
            state = np.concatenate((q[:9], fingertip_pos_world, obs.reshape(-1)))

        return state
    
    def _get_reward(self, action):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()
        hand_dofs = q[:9].copy() # joint angles of the hand
        
        variables = self.sim.get_variables().copy() # the variables contains the positions of three finger tips [0:3], [3:6], [6:9]
        fingertip_pos_world = np.array(variables[:9])
        
        cap_angle = q[-1]

        tactile_force = self.tactile_force_buf.cpu().numpy()
        tactile_force_grid = np.linalg.norm(tactile_force, axis=-1)
        finger_tactile_force = tactile_force_grid.sum(-1).sum(-1)
        not_in_contact = finger_tactile_force < 1.0

        reward = 0.

        reward -= not_in_contact.sum() * 0.5

        max_rotation_angle = np.pi / 4
        rotation_reward = -self.rot_coef * (min(cap_angle - max_rotation_angle, 0)) ** 2
        reward += rotation_reward

        action_penalty = -self.power_coef * np.sum(action ** 2)
        reward += action_penalty

        done = False
        success = False

        if np.any(fingertip_pos_world[2::3] > self.cap_top_surface_z):
            done = True
            reward += -50
            if self.verbose:
                print(f'fingertip pos:{fingertip_pos_world[2::3]}')

        if cap_angle > max_rotation_angle:
            reward += 50
            success = True
            done = True
        if self.verbose:
            print(f'cap angle:{cap_angle}, target:{max_rotation_angle}   done:{done}')
        return reward, done, success

    def reset(self):
        q_init = self.q_init.copy() # (10, ), the last DoF is for the cap rotation angle
        q_init[0:9] = q_init[0:9] + self.np_random.randn(9) * 0.05 # the first 9 Dof are the joint angles of dclaw
        qdot_init = np.zeros(10)

        damping = self.np_random.uniform(low = 0.01, high = 0.7)
        radius = self.np_random.uniform(low=0.02, high=0.08)
        dx, dy = self.np_random.uniform(low=-0.02, high=0.02, size=2)

        self.sim.update_joint_damping("cap", damping)

        self.sim.update_body_size("cap", np.array([0.03, radius]))
        self.sim.update_endeffector_position('cap', np.array([radius, 0, 0]))

        self.sim.update_joint_location('cap', np.array([dx, dy, 0.075]))
        if self.verbose:
            print(f'='*50)
            print(f'Damping:{damping}  Radius:{radius}  dx:{dx}  dy:{dy}')
        self.sim.set_state_init(q_init, qdot_init)
        
        self.sim.reset(backward_flag = False)
        self.energy_usage = 0
        self.has_been_reset = True

        self.tactile_force_his = []
        self.tactile_obs_his = []

        return self._get_obs()

    def step(self, u):
        if self.use_torch:
            u = u.detach().cpu().numpy()

        action = np.clip(u, -1., 1.)
        policy_out = u.copy()

        if not self.is_torque_control:
            if self.relative_control:
                cur_q = self.sim.get_q().copy()[:9]
                action = cur_q + action * self.relative_q_scale
                action = np.clip(action, self.dof_limit[:, 0], self.dof_limit[:, 1])
            else:
                action = scale(action, self.dof_limit[:, 0], self.dof_limit[:, 1])
        self.sim.set_u(action)

        self.sim.forward(self.frame_skip, verbose = False, test_derivatives = False)

        reward, done, success = self._get_reward(policy_out)

        # append tactile his
        if self.render_tactile:
            self.tactile_force_his.append(self.get_tactile_forces_array())
            self.tactile_obs_his.append(self.get_tactile_obs_array())

        info = {'success': success}
        if done:
            info['reward_energy'] = self.energy_usage

        return self._get_obs(), reward, done, info

    def visualize_tactile(self, tactile_array):
        resolution = 40
        horizontal_space = 10
        vertical_space = 20
        T = len(tactile_array)
        N = tactile_array.shape[1]
        nrows = tactile_array.shape[2]
        ncols = tactile_array.shape[3]

        imgs_tactile = np.zeros((ncols * resolution * N + vertical_space * (N + 1), nrows * resolution * T + horizontal_space * (T + 1), 3), dtype=float)
    
        for timestep in range(T):
            for finger_idx in range(N):
                for row in range(nrows):
                    for col in range(ncols):
                        if self.tactile_mask[row][col]:
                            loc0_x = row * resolution + resolution // 2 + timestep * nrows * resolution + timestep * horizontal_space + horizontal_space
                            loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                            loc1_x = loc0_x + tactile_array[timestep][finger_idx][row, col][0] / 10. * resolution
                            loc1_y = loc0_y + tactile_array[timestep][finger_idx][row, col][1] / 10. * resolution
                            depth_ratio = min(1., np.abs(tactile_array[timestep][finger_idx][row, col][2]) / 0.2)
                            color = (0.0, 1.0 - depth_ratio, depth_ratio)
                            cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)), color, 12, tipLength=0.4)

        return imgs_tactile

    def render(self, mode = 'once'):
        if self.render_tactile and self.observation_type != "no_tactile":
            for t in range(len(self.tactile_force_his)):
                img_tactile_1 = self.visualize_tactile(self.tactile_force_his[t][:, 0:1, ...])
                img_tactile_2 = self.visualize_tactile(self.tactile_force_his[t][:, 1:2, ...])
                img_tactile_3 = self.visualize_tactile(self.tactile_force_his[t][:, 2:3, ...])
                img_tactile_1 = img_tactile_1.transpose([1, 0, 2])
                img_tactile_2 = img_tactile_2.transpose([1, 0, 2])
                img_tactile_3 = img_tactile_3.transpose([1, 0, 2])
                cv2.imshow("tactile_1", img_tactile_1)
                cv2.imshow("tactile_2", img_tactile_2)
                cv2.imshow("tactile_3", img_tactile_3)
                
                cv2.waitKey(1)

                time.sleep(0.05)

        super().render(mode) 

    def get_tactile_forces_array(self):
        tactile_force_array = self.tactile_force_buf.clone().detach().cpu().numpy()
        tactile_force_array[...,0:2] = tactile_force_array[...,0:2] / 0.03
        tactile_force_array[...,2:3] = tactile_force_array[...,2:3] / 3.
        return np.expand_dims(tactile_force_array, axis=0)

    # return tactile obs array: shape (T, 2, nrows, ncols, 2)
    def get_tactile_obs_array(self):
        if self.observation_type == 'tactile_flatten' or self.observation_type == 'privilege' or self.observation_type == "tactile_flatten_his" or self.observation_type == "no_tactile":
            tactile_obs = self.tactile_obs_buf.reshape(1, 3, self.tactile_rows, self.tactile_cols, 3)
        elif self.observation_type == 'tactile':
            tactile_obs = self.tactile_obs_buf.reshape(1, 3, 3, self.tactile_rows, self.tactile_cols) \
                .permute(0, 1, 3, 4, 2)

        return tactile_obs.detach().cpu().numpy()


def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)