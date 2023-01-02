import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(project_base_dir)

import envs
import gym

if __name__ == '__main__':
    env = gym.make('TactileRotation-v1', use_torch = False, observation_type = "tactile", seed = 0)

    action_space = env.action_space

    env.reset()
    for i in range(1000):
        action = action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            env.render(mode = 'once')
            print('reset')
            obs = env.reset()

