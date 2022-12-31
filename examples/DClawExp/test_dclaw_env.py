import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(project_base_dir)

import envs
import gym
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type = str, default = 'TactileRotation-v1')
    parser.add_argument('--steps', type = int, default = 1000)

    args = parser.parse_args()

    env = gym.make(args.env, use_torch = False, observation_type = "tactile", seed = 0)

    action_space = env.action_space

    env.reset()
    for i in range(args.steps):
        action = action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            env.render(mode = 'loop')
            print('reset')
            obs = env.reset()

