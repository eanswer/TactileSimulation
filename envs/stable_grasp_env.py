import os
from re import I
import sys
from envs.redmax_torch_functions import EpisodicSimFunction

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import numpy as np
from utils import math
from envs.redmax_torch_env import RedMaxTorchEnv
from utils.common import *
from gym import spaces
import torch
import cv2
from copy import deepcopy

class StableGraspEnv(RedMaxTorchEnv):
    def __init__(self, use_torch = False, observation_type = "tactile_map",
                 render_tactile = False, verbose = False, seed = 0):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        model_path = os.path.join(asset_folder, 'stable_grasp/stable_grasp.xml')

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose
        self.render_tactile = render_tactile

        self.tactile_rows = 13
        self.tactile_cols = 10

        if self.observation_type == "tactile_flatten":
            self.ndof_obs = self.tactile_rows * self.tactile_cols * 2 * 2
            self.obs_shape = (self.tactile_rows * self.tactile_cols * 2 * 2, )
        elif self.observation_type == "tactile_map":
            self.obs_shape = (2 * 2, self.tactile_rows, self.tactile_cols)
        else:
            raise NotImplementedError

        self.tactile_samples = 1
        self.tactile_force_buf = torch.zeros(self.obs_shape)
        self.obs_buf = torch.zeros(self.obs_shape)
        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}

        super(StableGraspEnv, self).__init__(model_path, seed = seed)

        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = np.array([3., -1., 1.7])

        self.qpos_init_reference, self.qvel_init_reference = self.generate_initial_state()
        
        self.ndof_u = 1
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)
        self.action_scale = 0.05

        self.grasp_position_bound = 0.11
    
    def _get_obs(self):
        if not self.use_torch:
            return self.obs_buf.detach().cpu().numpy()
        else:
            return self.obs_buf

    '''
    reset function randomize the box density.
    It works in a way so that the COM of the bar follows a uniform distribution
    '''
    def reset(self): 
        self._progress_step = 0
        density_range = [600., 700.]
        num_blocks = 11
        box_ids = [9, 8, 1, 2, 3, 4, 5, 6, 7, 10, 11]
        com_y = self.np_random.uniform(1, num_blocks - 1, 1)
        if self.verbose:
            print(f'COM-y:{com_y}')
        num_left_blocks = int(com_y)
        num_right_blocks = num_blocks - 1 - num_left_blocks
        mid_block_left_ratio = com_y - num_left_blocks
        mid_block_right_ratio = 1 - mid_block_left_ratio

        mid_block_density = self.np_random.uniform(density_range[0], density_range[1], 1)[0]

        if mid_block_left_ratio < 0.5:
            right_blocks_total_density = self.np_random.uniform(density_range[0] * num_right_blocks,
                                                                density_range[1] * num_right_blocks,
                                                                1)[0]
            left_blocks_total_density = right_blocks_total_density + (1 - mid_block_left_ratio * 2) * mid_block_density
        else:
            left_blocks_total_density = self.np_random.uniform(density_range[0] * num_left_blocks,
                                                                density_range[1] * num_left_blocks,
                                                                1)[0]
            right_blocks_total_density = left_blocks_total_density + (mid_block_left_ratio * 2 - 1) * mid_block_density

        left_density_ratio = self.np_random.random(num_left_blocks) + 0.1
        left_density_ratio /= left_density_ratio.sum()
        right_density_ratio = self.np_random.random(num_right_blocks) + 0.1
        right_density_ratio /= right_density_ratio.sum()

        right_blocks_density = right_blocks_total_density * right_density_ratio
        left_blocks_density = left_blocks_total_density * left_density_ratio

        block_densitys = left_blocks_density.tolist()
        if mid_block_left_ratio > 0:
            block_densitys.append(mid_block_density)
        block_densitys.extend(right_blocks_density.tolist())
        block_densitys = np.array(block_densitys)

        total_density_range = [3000, 7000]
        block_densitys = block_densitys / block_densitys.sum() * np.clip(block_densitys.sum(), total_density_range[0], total_density_range[1])
        assert len(block_densitys) == num_blocks
        total_density = sum(block_densitys)
        self.block_densitys = deepcopy(block_densitys)
        if self.verbose:
            print(f'Density distribution:{block_densitys}')
            print(f'Total density:{total_density}')
        for idx in range(num_blocks):
            box_id = box_ids[idx]
            box_name = "box_{}".format(box_id)
            self.sim.update_body_density(box_name, block_densitys[idx])
            # color = (1. - block_densitys[idx] / total_density) * np.ones(3)
            color = (1. - block_densitys[idx] / 1000.) * np.ones(3)
            color_light = np.ones(3)
            color_heavy = np.array([254., 122., 21.]) / 255.
            color = min(1000, block_densitys[idx]) / 1000. * (color_heavy - color_light) + color_light
            self.sim.update_body_color(box_name, color)
        self.grasp_position = 0.
        self.prev_grasp_position = 0.
        self.current_q = self.qpos_init_reference.clone()
        
        self.sim.clearBackwardCache()
        
        self.record_idx += 1
        self.record_episode_idx = 0

        self.grasp()

        return self._get_obs()

    def step(self, u):
        self._progress_step += 1
        if self.use_torch:
            action = torch.clip(u, -1., 1.)
        else:
            action = torch.clip(torch.tensor(u), -1., 1.)
        
        action_unnorm = action * self.action_scale

        self.grasp_position = torch.clip(self.grasp_position + action_unnorm[0], -self.grasp_position_bound, self.grasp_position_bound)

        self.grasp()

        reward, done = self.reward_buf, self.done_buf

        if not self.use_torch:
            reward = reward.detach().cpu().item()
        
        if done:
            self.render('loop')

        obs = self._get_obs()
        return obs, reward, done, {'success': self.is_success}

    def generate_initial_state(self):
        qpos_init = self.sim.get_q_init().copy()
        grasp_height = 0.2
        qpos_init[2] = grasp_height
        qpos_init[4] = -0.03
        qpos_init[5] = -0.03
    
        self.sim.set_q_init(qpos_init)

        self.sim.reset(backward_flag = False)
        
        u = qpos_init[0:6].copy()
        u[2] += 0.003 # hard-coded feedforward term

        self.sim.set_u(u)
        self.sim.forward(500, verbose = False, test_derivatives = False)

        initial_qpos = self.sim.get_q().copy()

        initial_qvel = self.sim.get_qdot().copy()

        return torch.tensor(initial_qpos), torch.tensor(initial_qvel)

    '''
    each grasp consists of five stage:
    (1) move to the target grasp position (achieve by setting the q_init directly)
    (2) close the gripper
    (3) lift and capture the tactile frame
    (4) put down
    (5) open the gripper
    '''
    def grasp(self):
        lift_height = 0.2029862 + 0.03
        grasp_height = 0.2029862
        grasp_finger_position = -0.008

        qpos_init = self.current_q.clone().cpu().numpy()

        qpos_init[1] = self.grasp_position
        
        self.prev_grasp_position = self.grasp_position

        target_qs = []

        target_qs.append(qpos_init[:6]) # stage 1

        target_qs.append(np.array([0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position])) # stage 2
        target_qs.append(np.array([0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position]))
        target_qs.append(np.array([0.0, self.grasp_position, lift_height, 0.0, grasp_finger_position, grasp_finger_position])) # stage 3
        target_qs.append(np.array([0.0, self.grasp_position, lift_height, 0.0, grasp_finger_position, grasp_finger_position]))
        target_qs.append(np.array([0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position])) # stage 4
        target_qs.append(np.array([0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position]))
        target_qs.append(np.array([0.0, self.grasp_position, grasp_height, 0.0, qpos_init[4], qpos_init[5]])) # stage 5

        num_steps = [20, 10, 50, 20, 50, 10, 20]

        assert len(num_steps) == len(target_qs) - 1
        actions = []

        for stage in range(len(target_qs) - 1):
            for i in range(num_steps[stage]):
                u = (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage] * (i + 1) + target_qs[stage] # linearly interpolate the target joint positions
                actions.append(u)
        
        actions = torch.tensor(np.array(actions))
        tactile_masks = torch.zeros(actions.shape[0], dtype = bool)
        capture_frame = 60
        tactile_masks[capture_frame] = True
        
        #################################################################################
        # qs is the states information of the simulation trajectory
        # qs is (T, ndof_q), each time step, q is a ndof_q-dim vector consisting:
        # qs[t, 0:3]: position of gripper base
        # qs[t, 3]: z-axis rotation of gripper revolute joint
        # qs[t, 4:6]: the positions of two gripper fingers (along x axis)
        # qs[t, 6:9]: the position of the object
        # qs[t, 9:12]: the orientation of the object in rotvec representation
        #################################################################################
        # tactiles are the tactile vectors acquired at the time steps specified by tactile_masks
        #################################################################################
        qs, _, tactiles = EpisodicSimFunction.apply(torch.tensor(qpos_init), torch.zeros(self.ndof_q), actions, tactile_masks, self.sim, False)

        self.tactile_force_buf = tactiles.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 3)[...,0:2].clone()

        self.obs_buf = self.tactile_force_buf.clone()

        self.obs_buf = self.normalize_tactile(self.obs_buf)

        if self.observation_type == "tactile_flatten":
            self.obs_buf = self.obs_buf.reshape(-1)
        elif self.observation_type == "tactile_map":
            self.obs_buf = self.obs_buf \
                            .permute(0, 1, 4, 2, 3) \
                            .reshape(-1, self.tactile_rows, self.tactile_cols)

        # compute reward
        object_rotvec = qs[capture_frame, 9:12]
        abs_angle = np.linalg.norm(object_rotvec)
        if abs_angle < 0.02 and qs[capture_frame, -4] > 0.005:
            success = True
        else:
            success = False
        
        if success:
            if self.verbose:
                print('Success: ', np.linalg.norm(object_rotvec))
            self.reward_buf = torch.tensor(100.)
            self.done_buf = True
            self.is_success = True
        else:
            if self.verbose:
                print('Failure: ', np.linalg.norm(object_rotvec))
            self.reward_buf = torch.tensor(-abs_angle * 10.)
            self.done_buf = False
            self.is_success = False
        
        self.current_q = qs[-1].clone()

    '''
    normalize the shear force field
    input: dimension (T, 2, nrows, ncols, 2)
    output: dimension (T, 2, nrows, ncols, 2)
    '''
    def normalize_tactile(self, tactile_arrays):
        normalized_tactile_arrays = tactile_arrays.clone()

        lengths = torch.norm(tactile_arrays, dim = -1)
        
        max_length = np.max(lengths.numpy()) + 1e-5
        normalized_tactile_arrays = normalized_tactile_arrays / (max_length / 30.)
        
        return normalized_tactile_arrays

    def visualize_tactile(self, tactile_array):
        resolution = 40
        horizontal_space = 20
        vertical_space = 40
        T = len(tactile_array)
        N = tactile_array.shape[1]
        nrows = tactile_array.shape[2]
        ncols = tactile_array.shape[3]

        imgs_tactile = np.zeros((ncols * resolution * N + vertical_space * (N + 1), nrows * resolution * T + horizontal_space * (T + 1), 3), dtype = float)

        for timestep in range(T):
            for finger_idx in range(N):
                for row in range(nrows):
                    for col in range(ncols):
                        loc0_x = row * resolution + resolution // 2 + timestep * nrows * resolution + timestep * horizontal_space + horizontal_space
                        loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                        loc1_x = loc0_x + tactile_array[timestep][finger_idx][row, col][0] * 1.
                        loc1_y = loc0_y + tactile_array[timestep][finger_idx][row, col][1] * 1.
                        color = (0.0, 1.0, 0.0)
                        cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)), color, 2, tipLength = 0.3)
        
        return imgs_tactile
        
    def render(self, mode = 'once'):
        if self.render_tactile:
            tactile_obs = self.get_tactile_obs_array()
            img_tactile_left = self.visualize_tactile(tactile_obs[:, 0:1, ...])
            img_tactile_right = self.visualize_tactile(tactile_obs[:, 1:2, ...])
            img_tactile_left = img_tactile_left.transpose([1, 0, 2])
            img_tactile_right = img_tactile_right.transpose([1, 0, 2])

            cv2.imshow("tactile_left", img_tactile_left)
            cv2.imshow("tactile_right", img_tactile_right)
            
            print_info('Press [Esc] to continue.')
            cv2.waitKey(0)
    
        if mode == 'record':
            super().render(mode = 'record')
        else:
            print_info('Press [Esc] to continue.')
            super().render(mode)

    # return tactile obs array: shape (T, 2, nrows, ncols, 2)
    def get_tactile_obs_array(self):
        if self.observation_type == 'tactile_flatten':
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2)
        elif self.observation_type == 'tactile_map':
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, 2, self.tactile_rows, self.tactile_cols) \
                                .permute(0, 1, 3, 4, 2)
        
        return tactile_obs.detach().cpu().numpy()

    def get_tactile_forces_array(self):
        return self.tactile_force_buf.detach().cpu().numpy() / 0.000005
