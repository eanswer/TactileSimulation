import os
import sys

from scipy.spatial.transform.rotation import Rotation

from utils.common import get_time_stamp

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from utils.renderer import SimRenderer

def convert_observation_to_space(observation):
    if hasattr(observation, 'shape'):
        if len(observation.shape) == 1:
            low = np.full(observation.shape, -float('inf'), dtype=np.float32)
            high = np.full(observation.shape, float('inf'), dtype=np.float32)
            space = spaces.Box(low, high, dtype=np.float32)
        elif len(observation.shape) == 3:
            space = spaces.Box(low = -np.inf, high = np.inf, shape = observation.shape, dtype = np.float32)
    else:
        return None
    
    return space

class RedMaxTorchEnv(gym.Env):
    def __init__(self, model_path, seed = 0):
        self.sim = redmax.Simulation(model_path, verbose = True)

        self.ndof_q = self.sim.ndof_r
        self.ndof_var = self.sim.ndof_var
        self.ndof_u = self.sim.ndof_u
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)
        
        obs_tmp = self._get_obs()
        self.observation_space = convert_observation_to_space(obs_tmp)
            
        self.record_folder = os.path.join('./record/', get_time_stamp())
        self.record_idx = 0
        self.record_episode_idx = 0
        self.seed(seed = seed)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
            os.makedirs(os.path.join(self.record_folder, '{}'.format(self.record_idx)), exist_ok = True)
            SimRenderer.replay(self.sim, record = True, record_path = os.path.join(self.record_folder, '{}'.format(self.record_idx), '{}.gif'.format(self.record_episode_idx)))
            self.record_episode_idx += 1
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
