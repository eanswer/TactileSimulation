import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(project_base_dir)

import envs
import gym

if __name__ == '__main__':
    env = gym.make('StableGrasp-v1', observation_type = 'tactile_map', verbose = True, render_tactile = True)

    action_space = env.action_space

    env.reset()

    env.render(mode = 'loop')

    for i in range(100):
        action = action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render(mode = 'loop')
        if done:
            print('reset')
            obs = env.reset()
            env.render(mode = 'loop')