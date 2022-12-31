import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(project_base_dir)

import envs
import gym

if __name__ == '__main__':
    env = gym.make('TactilePush-v1', observation_type = 'tactile_map', verbose = True, render_tactile = True)

    action_space = env.action_space

    env.reset()

    for i in range(100):
        action = action_space.sample()
        action[0] = 1.
        action[1] = action[2] = 0.
        obs, reward, done, _ = env.step(action)
        if done:
            env.render(mode = 'loop')
            obs = env.reset()
            