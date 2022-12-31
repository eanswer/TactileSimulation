import os
import sys

from scipy.spatial.transform.rotation import Rotation

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import OrderedDict
from utils.renderer import SimRenderer

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class RedMaxEnv(gym.Env):
    def __init__(self, model_path, frame_skip, gradient = False):
        self.sim = redmax.Simulation(model_path, verbose = True)

        self.frame_skip = frame_skip
        self.gradient = gradient

        self.ndof_q = self.sim.ndof_r
        self.ndof_var = self.sim.ndof_var
        self.ndof_u = self.sim.ndof_u
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)
        self.render_mode = 'episodic'
        
        obs_tmp = self._get_obs()
        self.observation_space = convert_observation_to_space(obs_tmp)

        if self.gradient:
            self.dq, self.daction, self.dvar = None, None, None
            
        self.record_folder = './record/'
        self.record_idx = 0
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_random_rotation(self):
        '''
        follow the formula in http://planning.cs.uiuc.edu/node198.html
        '''
        ran = self.np_random.rand(3)
        r1, r2, r3 = ran[0], ran[1], ran[2]
        pi2 = 2 * np.pi
        r1_1 = np.sqrt(1.0 - r1)
        r1_2 = np.sqrt(r1)
        t1 = pi2 * r2
        t2 = pi2 * r3

        quat = np.zeros(4)
        quat[3] = r1_1 * (np.sin(t1)) # w
        quat[0] = r1_1 * (np.cos(t1)) # x
        quat[1] = r1_2 * (np.sin(t2)) # y
        quat[2] = r1_2 * (np.cos(t2)) # z

        return Rotation.from_quat(quat)

    def render(self, mode = 'once'):
        if mode == 'loop':
            self.sim.viewer_options.loop = True
            self.sim.viewer_options.infinite = True
        else:
            self.sim.viewer_options.loop = False
            self.sim.viewer_options.infinite = False
        if mode == 'record':
            self.sim.viewer_options.speed = 0.2
        elif mode == 'loop':
            self.sim.viewer_options.speed = 1.
        else:
            self.sim.viewer_options.speed = 2.
        if mode == 'record':
            os.makedirs(self.record_folder, exist_ok = True)
            SimRenderer.replay(self.sim, record = True, record_path = os.path.join(self.record_folder, '{}.gif'.format(self.record_idx)))
            self.record_idx += 1
        else:
            self.sim.replay()

    # methods to override:
    # -------------------------
    def reset(self):
        raise NotImplementedError

    def reset_from_checkpoint(self, state_checkpoint):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def get_simulation_state(self):
        raise NotImplementedError
