import os
from re import S
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from envs.redmax_torch_functions import EpisodicSimFunction
import numpy as np
from gym import spaces
from utils import math, torch_utils
from envs.redmax_torch_env import RedMaxTorchEnv
from utils.common import *
import cv2
import shutil
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class TactileInsertionEnv(RedMaxTorchEnv):
    def __init__(self, use_torch = False, observation_type = "tactile_flatten", observation_noise = True, \
                normalize_tactile_obs = True, allow_translation = True, allow_rotation = False, \
                num_obs_frames = 5, action_xy_scale = 0.02, action_rot_scale = np.pi / 18., \
                action_type = 'relative', reward_type = "absolute", domain_randomization = False, \
                seed = 0, render_tactile = True, verbose = False):

        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        model_path = os.path.join(asset_folder, 'tactile_insertion/tactile_insertion.xml')

        self.allow_translation = allow_translation
        self.allow_rotation = allow_rotation

        self.working_space_boundary = torch.tensor([0.015, 0.015])
        self.working_rotation_boundary = np.pi / 12.
        
        self.max_error = np.array([0.006, 0.006, np.pi / 18.])

        self.grasp_force_range = np.array([1. / 8., 0.8])

        self.relative_translation_limit = action_xy_scale
        self.relative_translation_xy = None
        self.relative_rotation_angle_limit = action_rot_scale
        self.relative_rotation_angle = None
        self.action_type = action_type
        self.reward_type = reward_type

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.observation_noise = observation_noise
        self.normalize_tactile_obs = normalize_tactile_obs
        self.domain_randomization = domain_randomization
        self.verbose = verbose

        self.execution_num_steps = 45
        if self.observation_type == "privilege":
            self.ndof_obs = 2
            self.obs_shape = (2,)
        elif self.observation_type == "tactile_flatten":
            self.tactile_rows = 13
            self.tactile_cols = 10
            self.tactile_initial_frame = 15
            self.tactile_samples = num_obs_frames
            self.obs_frequency = (self.execution_num_steps - self.tactile_initial_frame) // self.tactile_samples
            self.ndof_obs = self.tactile_rows * self.tactile_cols * 2 * 2 * self.tactile_samples
            self.obs_shape = (self.tactile_rows * self.tactile_cols * 2 * 2 * self.tactile_samples, )
        elif self.observation_type == "tactile_map":
            self.tactile_rows = 13
            self.tactile_cols = 10
            self.tactile_initial_frame = 12
            self.tactile_samples = num_obs_frames
            self.obs_frequency = (self.execution_num_steps - self.tactile_initial_frame) // self.tactile_samples
            self.obs_shape = (2 * 2 * self.tactile_samples, self.tactile_rows, self.tactile_cols)
        else:
            raise NotImplementedError
        self.tactile_masks = torch.zeros(self.execution_num_steps, dtype = bool)
        self.tactile_masks[self.tactile_initial_frame + self.obs_frequency - 1::self.obs_frequency] = True # observation frames
        self.tactile_masks[6] = True # reference frame
        
        self.mask_prob = torch.full((self.tactile_samples, 2, 1, 1, 1), 0.8)
        self.scale_mask = torch.ones((self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2))
        center_1 = np.array([8., 6.])
        center_2 = np.array([2.5, 5.5])
        for row in range(self.tactile_rows):
            for col in range(self.tactile_cols):
                if np.linalg.norm(np.array([row, col], dtype = float) - center_1) < 5. + 1e-5:
                    self.scale_mask[:, 0, row, col, :] = 0.4
                if np.linalg.norm(np.array([row, col], dtype = float) - center_2) < 3. + 1e-5:
                    self.scale_mask[:, 1, row, col, :] = 0.6
                
        self.scale_mask[:, 0, ]
        self.tactile_force_buf = torch.zeros(self.obs_shape)
        self.obs_buf = torch.zeros(self.obs_shape)
        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.initial_sensor_locations = None
        self.grasp_force = 1.

        super(TactileInsertionEnv, self).__init__(model_path, seed = seed)

        self.render_mode = 'step'
        self.render_tactile = render_tactile
        
        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = np.array([2., -2.5, 2])

        self.q_init_reference, self.qdot_init_reference = self.generate_initial_pose()
        
        self.ndof_u = 0
        if self.allow_translation:
            if self.allow_rotation:
                self.ndof_u = 3
                self.action_scale = torch.tensor([self.relative_translation_limit, self.relative_translation_limit, self.relative_rotation_angle_limit])
            else:
                self.ndof_u = 2
                self.action_scale = torch.tensor([self.relative_translation_limit, self.relative_translation_limit])
        else:
            assert self.allow_rotation
            self.ndof_u = 1
            self.action_scale = torch.tensor([self.relative_rotation_angle_limit])

        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)

        self.prev_object_pose = None

    def generate_initial_pose(self):
        q_init = self.sim.get_q_init().copy()
        grasp_height = 0.2
        q_init[2] = grasp_height
        q_init[4] = -0.03
        q_init[5] = -0.03
        initial_object_height = 0.026 + 0.003 # 0.003 is feedforward term
        self.sim.set_q_init(q_init)

        self.sim.reset(backward_flag = False)
        target_qs = [np.array([q_init[0], q_init[1], q_init[2], q_init[4], 0., 0.])]
        target_qs.append(np.array([0.0, 0.0, grasp_height, 0.0, 0., 0.]))
        target_qs.append(np.array([0.0, 0.0, grasp_height, 0.0, 1., 1.]))
        target_qs.append(np.array([0.0, 0.0, grasp_height, 0.0, 1., 1.]))

        num_steps = [100, 100, 300]

        heights = []
        for stage in range(len(target_qs) - 1):
            for i in range(num_steps[stage]):
                u = (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage] * (i + 1) + target_qs[stage]
                self.sim.set_u(u)
                self.sim.forward(1, verbose = False, test_derivatives = False)
                q = self.sim.get_q()
                if stage == 4:
                    heights.append(q[8])

        initial_q = self.sim.get_q().copy()

        initial_q[2] += initial_object_height
        initial_q[8] += initial_object_height

        self.sim.set_q_init(initial_q)
        self.sim.reset(backward_flag = False)
        u = initial_q[:6]
        u[4:6] = 1.
        
        self.sim.set_u(u)
        self.sim.forward(500, verbose = False, test_derivatives = False)

        initial_q = self.sim.get_q().copy()

        initial_qdot = self.sim.get_qdot().copy()

        return initial_q, initial_qdot
    
    def _get_obs(self):
        if not self.use_torch:
            return self.obs_buf.detach().cpu().numpy()
        else:
            return self.obs_buf
    
    def apply_relative_motion(self, q, relative_position, relative_rotation, grasp_height_noise = torch.tensor(0.)):
        new_q = q.clone()
        
        # apply position change
        if relative_position.shape[0] == 2:
            new_q[0:2] += relative_position
            new_q[6:8] += relative_position
        else:
            new_q[0:3] += relative_position
            new_q[6:9] += relative_position
        
        new_q[2] += grasp_height_noise

        # apply rotation change
        new_q[3] = new_q[3] + relative_rotation

        # make it into torch version
        new_q[9:12] = torch_utils.rotvec_mul(q[9:12], torch.tensor([0., 0., relative_rotation], dtype = q.dtype))

        return new_q

    '''
    reset by randomizing the initial gripper position and rotation (no rotation for now)
    '''
    def reset(self, position_noise = None, rotation_noise = None, grasp_height_noise = None):
        if position_noise is None:
            if self.allow_translation:
                position_noise = self.np_random.uniform(low = [-self.max_error[0], -self.max_error[1], -0.0002], high = [self.max_error[0], self.max_error[1], 0.0002], size = 3)
            else:
                position_noise = np.zeros(2)

        if rotation_noise is None:
            if self.allow_rotation:
                rotation_noise = self.np_random.uniform(low = -self.max_error[2], high = self.max_error[2])
            else:
                rotation_noise = 0.

        if grasp_height_noise is None:
            grasp_height_noise = self.np_random.uniform(low = -0.01, high = 0.005)

        q_init = self.q_init_reference.copy()

        self.current_q_init = self.apply_relative_motion(torch.tensor(q_init), torch.tensor(position_noise), torch.tensor(rotation_noise), torch.tensor(grasp_height_noise))

        self.original_q_init = self.current_q_init.clone()
        
        self.prev_object_pose = torch.tensor([self.current_q_init[0], self.current_q_init[1], self.current_q_init[3]])

        self.sim.clearBackwardCache()
        self.record_idx += 1
        self.record_episode_idx = 0
        
        # domain randomization
        if self.domain_randomization:
            self.do_domain_randomization()

        self.execute_insertion()
        
        return self._get_obs()
    
    def do_domain_randomization(self):
        contact_kn_range = [2e3, 14e3]
        contact_kt_range = [20., 140.]
        contact_mu_range = [0.5, 2.5]
        contact_damping_range = [1e3, 1e3]

        tactile_kn_range = [50, 450]
        tactile_kt_range = [0.2, 2.3]
        tactile_mu_range = [0.5, 2.5]
        tactile_damping_range = [0, 100]

        contact_kn = self.np_random.uniform(*contact_kn_range)
        contact_kt = self.np_random.uniform(*contact_kt_range)
        contact_mu = self.np_random.uniform(*contact_mu_range)
        contact_damping = self.np_random.uniform(*contact_damping_range)

        self.sim.update_contact_parameters('tactile_pad_left', 'box', \
                                            kn = contact_kn,\
                                            kt = contact_kt,\
                                            mu = contact_mu,\
                                            damping = contact_damping)
        self.sim.update_contact_parameters('tactile_pad_right', 'box', \
                                            kn = contact_kn,\
                                            kt = contact_kt,\
                                            mu = contact_mu,\
                                            damping = contact_damping)
        
        tactile_kn = self.np_random.uniform(*tactile_kn_range)
        tactile_kt = self.np_random.uniform(*tactile_kt_range)
        tactile_mu = self.np_random.uniform(*tactile_mu_range)
        tactile_damping = self.np_random.uniform(*tactile_damping_range)

        self.sim.update_tactile_parameters('tactile_pad_left', \
                                            kn = tactile_kn,\
                                            kt = tactile_kt,\
                                            mu = tactile_mu,\
                                            damping = tactile_damping)
        self.sim.update_tactile_parameters('tactile_pad_right', \
                                            kn = tactile_kn,\
                                            kt = tactile_kt,\
                                            mu = tactile_mu,\
                                            damping = tactile_damping)     

        self.grasp_force = self.np_random.uniform(self.grasp_force_range[0], self.grasp_force_range[1])

    '''
    action is a 2-d vector specify the relative location change for the pre-grasp location
    '''
    def step(self, u):
        if self.use_torch:
            action = torch.clip(u, -1., 1.)
        else:
            action = torch.clip(torch.tensor(u), -1., 1.)

        action_unnorm = action * self.action_scale
        
        if self.action_type == 'relative':
            base_idx = 0
            if self.allow_translation:
                relative_translation_xy = torch.clip(action_unnorm[0:2], \
                                                        -self.working_space_boundary - self.current_q_init[0:2], \
                                                        self.working_space_boundary - self.current_q_init[0:2])
                base_idx = 2
            else:
                relative_translation_xy = torch.zeros(2)
                base_idx = 0

            if self.allow_rotation:
                relative_rotation_angle = torch.clip(action_unnorm[base_idx], \
                                                        -self.working_rotation_boundary - self.current_q_init[3], \
                                                        self.working_rotation_boundary)
            else:
                relative_rotation_angle = torch.tensor(0.)

            self.current_q_init = self.apply_relative_motion(self.current_q_init, relative_translation_xy, relative_rotation_angle)
        elif self.action_type == 'accumulative':
            base_idx = 0
            if self.allow_translation:
                relative_translation_xy = action_unnorm[0:2]
                base_idx = 2
            else:
                relative_translation_xy = torch.zeros(2)
                base_idx = 0

            if self.allow_rotation:
                relative_rotation_angle = action_unnorm[base_idx]
            else:
                relative_rotation_angle = torch.tensor(0.)
            self.current_q_init = self.apply_relative_motion(self.original_q_init, relative_translation_xy, relative_rotation_angle)
        else:
            raise NotImplementedError
        
        self.execute_insertion()

        reward, done = self.reward_buf, self.done_buf
        
        if not self.use_torch:
            reward = reward.detach().cpu().item()

        obs = self._get_obs()
        
        return obs, reward, done, self.info_buf.copy()

    '''
    execute an insertion by starting from self.current_location_xy
    '''
    def execute_insertion(self):
        actions = torch.zeros((self.execution_num_steps, 6))

        init_joint_positions = self.current_q_init[:6]
        target_joint_positions = init_joint_positions.clone()

        target_joint_positions[2] -= 0.0011

        for step in range(self.execution_num_steps):
            u = (target_joint_positions - init_joint_positions) / self.execution_num_steps * (step + 1) + init_joint_positions
            u[2] += 0.003 # feedforward term
            u[4] = self.grasp_force
            u[5] = self.grasp_force
            actions[step, :] = u

        qs, _, tactiles = EpisodicSimFunction.apply(self.current_q_init, torch.zeros(self.ndof_q), actions, self.tactile_masks, self.sim, False)

        # use the relative tactile forces
        tactiles = tactiles - tactiles[0]
        tactiles = tactiles[1:]
        
        self.tactile_force_buf = tactiles.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 3)[...,0:2].clone()

        self.obs_buf = self.tactile_force_buf.clone()
            
        if self.observation_noise:
            noise_std = 0.00001
            noise = self.np_random.randn(*self.obs_buf.shape) * noise_std
            self.obs_buf += torch.from_numpy(noise)

        if self.normalize_tactile_obs:
            self.obs_buf = self.normalize_tactile(self.obs_buf)

        if self.observation_type == "tactile_flatten":
            self.obs_buf = self.obs_buf.reshape(-1)
        elif self.observation_type == "tactile_map":
            self.obs_buf = self.obs_buf \
                            .permute(0, 1, 4, 2, 3) \
                            .reshape(-1, self.tactile_rows, self.tactile_cols)

        self.current_object_pose = torch.tensor([self.current_q_init[0], self.current_q_init[1], self.current_q_init[3]])

        success = False
        if not self.allow_rotation:
            success = abs(qs[-1, 6]) <= 0.0022 and abs(qs[-1, 7]) <= 0.0022
        else:
            success = qs[-1, 8] < 0.0247

        pose_normed_prev = self.prev_object_pose / self.max_error
        pose_normed_now = self.current_object_pose / self.max_error

        improve = torch.norm(pose_normed_prev) > torch.norm(pose_normed_now)
        self.info_buf['prev_object_pose'] = self.prev_object_pose.detach().cpu().numpy()
        self.info_buf['new_object_pose'] = self.current_object_pose.detach().cpu().numpy()
        self.info_buf['improve'] = improve

        if self.reward_type == "absolute":
            self.reward_buf = -torch.sum(self.current_q_init[0:2] ** 2) * 10000 - torch.sum(self.current_q_init[3] ** 2) * 20.
        elif self.reward_type == "delta":
            self.reward_buf = (torch.norm(self.prev_object_pose / self.max_error) - torch.norm(self.current_object_pose / self.max_error)) * 10.
            if success:
                self.reward_buf += 20.
            else:
                self.reward_buf -= 1.
        else:
            raise NotImplementedError
        self.prev_object_pose = self.current_object_pose

        if success:
            if self.verbose:
                print_info('[Success] x {:.5f}, y {:.5f}'.format(qs[-1, 6].detach().cpu().item(), qs[-1, 7].detach().cpu().item()), \
                        'angle = {:.2f}'.format(np.rad2deg(self.current_q_init[3].detach().cpu().item())),\
                        'reward = {:.3f}'.format(self.reward_buf.detach().cpu().item()), \
                        'position_reward = {:.3f}'.format(-(torch.sum(self.current_q_init[0:2] ** 2) * 10000).detach().cpu().item()), \
                        'rotation_reward = {:.3f}'.format(-(torch.sum(self.current_q_init[3] ** 2) * 20).detach().cpu().item()))
            self.done_buf = True
            self.info_buf['success'] = True
        else:
            if self.verbose:
                print_info('[Failure] x {:.5f}, y {:.5f}'.format( qs[-1, 6].detach().cpu().item(), qs[-1, 7].detach().cpu().item()), \
                        'angle = {:.2f}'.format(np.rad2deg(self.current_q_init[3].detach().cpu().item())),\
                        'reward = {:.3f}'.format(self.reward_buf.detach().cpu().item()), \
                        'position_reward = {:.3f}'.format(-(torch.sum(self.current_q_init[0:2] ** 2) * 10000).detach().cpu().item()), \
                        'rotation_reward = {:.3f}'.format(-(torch.sum(self.current_q_init[3] ** 2) * 20).detach().cpu().item()))
            self.done_buf = False
            self.info_buf['success'] = False
    
    '''
    input: dimension (T, 2, nrows, ncols, 2)
    output: dimension (T, 2, nrows, ncols, 2)
    '''
    def normalize_tactile(self, tactile_arrays):
        lengths = torch.norm(tactile_arrays, dim = -1)
        
        max_length = np.max(lengths.numpy()) + 1e-5
        normalized_tactile_arrays = tactile_arrays / (max_length / 30.)
        
        return normalized_tactile_arrays

    def draw_tactile_force(self, img, loc0, loc1, tactile_force):
        end_loc0 = loc0 + int(tactile_force[0].item())
        end_loc1 = loc1 + int(tactile_force[1].item())
        
        color = (0., 1., 0.)
        
        img[:, :, :] = cv2.arrowedLine(img, (loc0, loc1), (end_loc0, end_loc1), color, 2, tipLength = 0.3)

    def draw_tactile_forces(self, img, tactile_forces, resolution):
        for i in range(tactile_forces.shape[0]):
            for j in range(tactile_forces.shape[1]):
                self.draw_tactile_force(img, i * resolution + resolution // 2, j * resolution + resolution // 2, tactile_forces[i, j])

    def visualize_tactile(self, tactile_array):
        resolution = 20
        horizontal_space = 20
        vertical_space = 40
        T = len(tactile_array)
        nrows = tactile_array.shape[2]
        ncols = tactile_array.shape[3]

        imgs_tactile = np.zeros((ncols * resolution * 2 + vertical_space * 3, nrows * resolution * T + horizontal_space * (T + 1), 3), dtype = float)

        for timestep in range(T):
            for finger_idx in range(2):
                for row in range(nrows):
                    for col in range(ncols):
                        loc0_x = row * resolution + resolution // 2 + timestep * nrows * resolution + timestep * horizontal_space + horizontal_space
                        loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                        loc1_x = loc0_x + tactile_array[timestep][finger_idx][row, col][0]
                        loc1_y = loc0_y + tactile_array[timestep][finger_idx][row, col][1]
                        cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)), (0.0, 1.0, 0.0), 2, tipLength = 0.3)
        
        return imgs_tactile
        
    def render(self, mode = 'once'):
        if self.render_tactile:
            tactile_forces = self.get_tactile_forces_array()
            tactile_obs = self.get_tactile_obs_array()
            img_tactile_clean = self.visualize_tactile(tactile_forces)
            img_tactile_noise = self.visualize_tactile(tactile_obs)
            img_tactile = np.zeros((img_tactile_clean.shape[0] + img_tactile_noise.shape[0], img_tactile_clean.shape[1], 3))
            img_tactile[0:img_tactile_clean.shape[0], :] = img_tactile_clean
            img_tactile[img_tactile_noise.shape[0]:, :] = img_tactile_noise
    
            cv2.imshow("tactile", img_tactile)
            print_info('Press [Esc] to continue...')
            cv2.waitKey(0)

        if mode == 'loop':
            print_info('Press [Esc] to continue...')

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
        return self.tactile_force_buf.detach().cpu().numpy() / 0.000002