
import sys, os

from cv2 import determinant

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_dir)

from utils.common import *

import gym
gym.logger.set_level(40)
import numpy as np
import time
from collections import deque
import copy
import yaml
from tensorboardX import SummaryWriter
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import envs
from utils import model
from utils.torch_utils import *
from utils.running_mean_std import RunningMeanStd
from copy import deepcopy

class GD:
    def __init__(self, cfg):
        torch.set_num_threads(1) # TODO: check the effect

        self.seed = cfg['params']['general']['seed']
        self.env_name = cfg['params']['env']['name']
        self.device = cfg['params']['general'].get('device', 'cpu')
        env_params = copy.deepcopy(cfg['params']['env'])
        env_params.pop('name', None)

        set_random_seed(self.seed)

        # create render env
        self.render_env = gym.make(self.env_name, use_torch = True, gradient = False, verbose = True, render_tactile = True, **env_params)
        self.render_env.seed(self.seed + 1)

        # create policy
        actor_fn = getattr(model, cfg['params']['network'].get('actor', 'DiagGaussianPolicy'))

        if self.render_env.observation_space != None:
            self.actor = actor_fn(self.render_env.observation_space.shape, self.render_env.action_space.shape[0], cfg['params']['network'])
        else:
            self.actor = actor_fn(self.render_env.mixed_observation_space, self.render_env.action_space.shape[0], cfg['params']['network'])
        
        if cfg['params']['general']['checkpoint'] != None:
            self.actor = torch.load(cfg['params']['general']['checkpoint'])[0]

        self.actor.to(self.device)

        if cfg['params']['general']['train']:
            # self.total_env_steps = cfg['params']['config']['num_env_steps']
            self.num_epochs = cfg['params']['config']['num_epochs']
            self.num_episodes = cfg['params']['config']['num_episodes']
            self.num_processes = cfg['params']['config'].get('num_processes', 1)
            self.lr = cfg['params']['config'].get('lr', 3e-4)
            self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
            assert self.lr_schedule in ['linear', 'constant']

            self.truncate_grads = cfg['params']['config'].get('truncate_grads', False)
            self.grad_norm = cfg['params']['config'].get('grad_norm', 1.0)

            self.obs_rms = None
            if cfg['params']['config'].get('obs_rms', False):
                self.obs_rms = RunningMeanStd(shape = self.render_env.observation_space.shape, device = self.device)

            self.gamma = cfg['params']['config'].get('gamma', 0.99)
            self.env = gym.make(self.env_name, use_torch = True, gradient = True, verbose = False, **env_params)
            self.env.seed(self.seed)

            # initialize optimizer
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.lr)

            # interval-related arguments
            self.save_interval = cfg['params']['general'].get('save_interval', 50)
            self.console_log_interval = cfg['params']['general'].get('log_interval', 1)
            self.render_interval = cfg['params']['general'].get('render_interval', 10)

            # create logging folder
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)
            os.makedirs(os.path.join(self.log_dir, 'models'), exist_ok = True)

            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]
            
            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))

            # create logging file
            self.training_log_path = os.path.join(self.log_dir, 'logs.txt')
            fp_log = open(self.training_log_path, 'w')
            fp_log.close()

            # initialize writer
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))

            # print summary of training
            print_info('-'*43)
            print_info('| {:^40}|'.format('[Training Info Summary]'))
            print_info('| {:<40}|'.format('env name = {}'.format(self.env_name)))
            for key in env_params:
                print_info('| {:<40}|'.format('env.{} = {}'.format(key, env_params[key])))
            # print_info('| {:<40}|'.format('total_env_steps = {}'.format(self.total_env_steps)))
            print_info('| {:<40}|'.format('num_epochs = {}'.format(self.num_epochs)))
            print_info('| {:<40}|'.format('num_episodes = {}'.format(self.num_episodes)))
            print_info('| {:<40}|'.format('num_processes = {}'.format(self.num_processes)))
            print_info('| {:<40}|'.format('lr = {}'.format(self.lr)))
            print_info('| {:<40}|'.format('lr schedule = {}'.format(self.lr_schedule)))
            print_info('| {:<40}|'.format('gamma = {}'.format(self.gamma)))
            print_info('| {:<40}|'.format('truncate_grads = {}'.format(self.truncate_grads)))
            print_info('| {:<40}|'.format('grad_norm = {}'.format(self.grad_norm)))
            print_info('| {:<40}|'.format('save_interval = {}'.format(self.save_interval)))
            print_info('| {:<40}|'.format('console_log_interval = {}'.format(self.console_log_interval)))
            print_info('| {:<40}|'.format('render_interval = {}'.format(self.render_interval)))
            print_info('-'*43)
        
        # self.test_simple_gradient()
        # self.test_gradient()

    def train(self):
        episode_rewards = deque(maxlen=200)
        episode_lens = deque(maxlen=200)

        time_start = time.time()
        
        self.total_num_steps = 0

        # save initial policy
        self.save('init_policy')

        best_episode_rewards = -np.inf
        for epoch in range(self.num_epochs):
            if self.lr_schedule == 'linear':
                lr = (1e-5 - self.lr) * float(epoch / self.num_epochs) + self.lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = self.lr

            self.actor_optimizer.zero_grad()

            rewards, lens = self.compute_reward_and_grad(self.actor, self.num_episodes)

            grad_norm_before_clip = grad_norm(self.actor.parameters())

            if self.truncate_grads:
                clip_grad_norm_(self.actor.parameters(), self.grad_norm)

            grad_norm_after_clip = grad_norm(self.actor.parameters())

            self.actor_optimizer.step()
            # print(self.actor.logstd)
            for reward, len in zip(rewards, lens):
                episode_rewards.append(reward)
                episode_lens.append(len)
            
            # logging
            time_end = time.time()
            time_elapse = time_end - time_start
            fps = self.total_num_steps / time_elapse
            
            # mean_rewards = np.mean(rewards)
            mean_rewards = np.mean(episode_rewards)

            # logging to console
            if self.console_log_interval > 0 and epoch % self.console_log_interval == 0:
                # if np.mean(episode_rewards) > best_episode_rewards:
                if mean_rewards > best_episode_rewards:
                    print_ok('epoch {}: num steps = {}, FPS = {:.1f}, mean(reward) = {:.6f}, mean(len) = {:.1f}, grad_norm_before = {:.3f}, grad_norm_after = {:.3f}'.format(epoch, self.total_num_steps, fps, mean_rewards, np.mean(episode_lens), grad_norm_before_clip, grad_norm_after_clip))
                else:
                    print('epoch {}: num steps = {}, FPS = {:.1f}, mean(reward) = {:.6f}, mean(len) = {:.1f}, grad_norm = {:.3f}, grad_norm_after = {:.3f}'.format(epoch, self.total_num_steps, fps, mean_rewards, np.mean(episode_lens), grad_norm_before_clip, grad_norm_after_clip))

            # save the best model
            if mean_rewards > best_episode_rewards:
                best_episode_rewards = mean_rewards
                print_info('saving best policy with reward {:.2f}'.format(best_episode_rewards))
                self.save()

            # save models during training
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self.save('policy_iter{}_reward{:.2f}'.format(epoch, mean_rewards))

            # evaluate
            if self.render_interval > 0 and epoch % self.render_interval == 0:
                self.evaluate(self.actor, render = True)

            # logging to writer
            self.writer.add_scalar('rewards/step', mean_rewards, self.total_num_steps)
            self.writer.add_scalar('rewards/time', mean_rewards, time_elapse)
            self.writer.add_scalar('rewards/iter', mean_rewards, epoch)
            self.writer.add_scalar('episode_lengths/step', np.mean(episode_lens), self.total_num_steps)
            self.writer.add_scalar('episode_lengths/time', np.mean(episode_lens), time_elapse)
            self.writer.add_scalar('episode_lengths/iter', np.mean(episode_lens), epoch)
            self.writer.add_scalar('stats/rewards/step', mean_rewards, self.total_num_steps)
            self.writer.add_scalar('stats/episode_lengths/step', np.mean(episode_lens), self.total_num_steps)
            self.writer.add_scalar('stats/lr/step', lr, self.total_num_steps)

            self.writer.flush()

        # save final policy
        self.save('final_policy')

        self.writer.close()
        self.render_env.close()
        self.env.close()

    def compute_reward_and_grad(self, actor, num_episodes, render = False):
        episode_rewards = []
        episode_lens = []
        
        for i in range(num_episodes):
            if self.obs_rms:
                obs_rms = copy.deepcopy(self.obs_rms)

            obs = self.env.reset()

            if self.obs_rms is not None:
                with torch.no_grad():
                    self.obs_rms.update(obs.unsqueeze(0))
                obs = obs_rms.normalize(obs)
                
            episode_reward = 0.
            episode_len = 0
            gamma = 1.
            done = False
            while not done:
                u = actor.act(obs, deterministic = True).squeeze(0)
                obs, reward, done, info = self.env.step(u)
                
                if self.obs_rms is not None:
                    with torch.no_grad():
                        self.obs_rms.update(obs.unsqueeze(0))
                    obs = obs_rms.normalize(obs)

                # episode_reward += gamma * reward
                episode_reward += reward
            
                episode_len += 1
                self.total_num_steps += 1
                gamma *= self.gamma

            episode_rewards.append(episode_reward.detach().cpu().item())
            episode_lens.append(episode_len)

            loss = episode_reward * -1. / num_episodes
            loss.backward()
            # grad = flatten_grad(self.actor)
            # import IPython
            # IPython.embed()

        return episode_rewards, episode_lens
    
    def evaluate(self, actor, render = False, record = False, stochastic = False):
        obs = self.render_env.reset()
        episode_reward = 0.
        episode_len = 0
        done = False
        reward_details = {}
        while not done:
            if self.obs_rms:
                obs = self.obs_rms.normalize(obs)
            u = actor.act(obs, deterministic = not stochastic).squeeze(0)
            obs, reward, done, info = self.render_env.step(u)
            
            episode_reward += reward
        
            episode_len += 1

            for key in info:
                if key[0:6] == 'reward':
                    if key not in reward_details:
                        reward_details[key] = 0.
                    reward_details[key] += info[key].detach().cpu().item()
                elif key[0:5] == 'final':
                    if done:
                        reward_details[key] = info[key].detach().cpu().item()

        print_ok('[Evaluation] Reward = {:.2f}, len = {}, {}'.format(episode_reward, episode_len, reward_details))

        if render:
            if not record:
                self.render_env.render('loop')
            else:
                self.render_env.render('record')

        return episode_reward, episode_len, reward_details

    def play(self, cfg):
        
        # _, self.actor = torch.load(cfg['params']['general']['checkpoint'])
        self.actor, self.obs_rms = torch.load(cfg['params']['general']['checkpoint'])
        num_games = cfg['params']['general']['num_games']
        
        games_cnt = 0
        total_reward = 0.
        total_reward_details = {}

        while num_games != games_cnt:
            reward, episode_len, reward_details = self.evaluate(self.actor, render = cfg['params']['general']['render'], record = cfg['params']['general']['record'], stochastic = cfg['params']['general']['stochastic'])
            games_cnt += 1
            total_reward += reward
            for key in reward_details:
                if key not in total_reward_details:
                    total_reward_details[key] = 0.
                total_reward_details[key] += reward_details[key]
        
        for key in total_reward_details:
            total_reward_details[key] = total_reward_details[key] / num_games

        print_info('[Summary] Avg reward = {:.3f}, details = {}'.format(total_reward / games_cnt, total_reward_details))

        print_info('-------------------------------------------------------------------------------------------------------------')

    def save(self, filename = None):
        if filename is None:
            filename = 'best_model'
        torch.save([self.actor, self.obs_rms], os.path.join(self.log_dir, "models", "{}.pt".format(filename)))

    def test_gradient(self):
        self.actor_optimizer.zero_grad()
        def evaluate(actor, backward):
            self.env.seed(10)
            obs = self.env.reset()
            # if backward:
            #     self.env.render('loop')
            episode_reward = 0.
            total_reward = 0.
            gamma = 1
            num_steps = 50 * 1
            tactile_loss = 0.
            for i in range(num_steps):
                u = actor.act(obs, deterministic = True).squeeze(0)
                obs, reward, done, _ = self.env.step(u)
                # episode_reward += gamma * reward
                episode_reward += torch.sum(obs[0:3])
                gamma *= self.gamma
                # if backward:
                #     self.env.render('loop')
                if done or i == num_steps - 1:
                    if backward:
                        print(episode_reward)
                    if backward:
                        episode_reward.backward()
                        # self.env.render('loop')
                    total_reward += episode_reward.detach().cpu().item()
                    obs = self.env.reset()
                    episode_reward = 0.
                    gamma = 1.
            return total_reward

        print('start analytical gradient computation')

        total_reward = evaluate(self.actor, backward = True)
        # print(total_reward)
        grad = flatten_grad(self.actor)
        print(grad.norm())
        # import IPython
        # IPython.embed()
        params = flatten_params(self.actor)

        print('start finite gradient computation')
        actor_tmp = deepcopy(self.actor)
        grad_fd = torch.zeros_like(grad)
        num_params = 50
        params_indices = np.random.randint(low = 0, high = grad.shape[0], size = num_params)
        # eps = 1e-2
        eps = 1.

        for _ in range(10):
            # for i in range(num_params):
            for i in params_indices:
                params_pos = params.clone()
                params_pos[i] += eps
                fill_params(actor_tmp, params_pos)
                total_reward_pos = evaluate(actor_tmp, backward = False)
                grad_fd[i] = (total_reward_pos - total_reward) / eps
                # params_neg = params.clone()
                # params_neg[i] -= eps
                # fill_params(actor_tmp, params_neg)
                # total_reward_neg = evaluate(actor_tmp, backward = False)
                # grad_fd[i] = (total_reward_pos - total_reward_neg) / (2. * eps)
            
            abs_error = (grad_fd[params_indices] - grad[params_indices]).norm().detach().cpu().item()
            rel_error = abs_error / max(1e-7, min(grad_fd[params_indices].norm().detach().cpu().item(), grad[params_indices].norm().detach().cpu().item()))

            grad_fd_normalized = grad_fd[params_indices] / grad_fd[params_indices].norm()
            grad_normalized = grad[params_indices] / grad[params_indices].norm()
            dot_product = (grad_fd_normalized * grad_normalized).sum().detach().cpu().item()
            print('eps = {}, abs error = {}, rel error = {}, dot product = {}'.format(eps, abs_error, rel_error, dot_product))
            eps /= 10.

        print('exit')

    def test_simple_gradient(self):
        num_steps = 20

        def evaluate(us):
            self.env.seed(10)
            _ = self.env.reset()
            
            episode_reward = 0.
            for i in range(num_steps):                
                obs, reward, _, _ = self.env.step(us[i])

                # episode_reward += reward
                episode_reward += torch.sum(obs[0:3])
            
            return episode_reward

        print('start analytical gradient computation')

        self.env.seed(10)
        obs = self.env.reset()
        
        us = []
        with torch.no_grad():
            for i in range(num_steps):
                u = self.actor.act(obs.unsqueeze(0), deterministic = True).squeeze(0)
                us.append(u)
                obs, _, _, _ = self.env.step(u)
            us = torch.stack(us, dim = 0)
        
        us.requires_grad = True
        print('here')
        total_reward = evaluate(us)
        
        total_reward.backward()
        # print(total_reward)
        grad = us.grad.clone()

        print(grad.norm())

        print('start finite gradient computation')
        
        grad_fd = torch.zeros_like(grad)
        
        eps = 1e-2
        for _ in range(10):
            for i in range(len(us)):
                for j in range(len(us[0])):
                    u_pos = us.clone()
                    u_pos[i][j] += eps
                    total_reward_pos = evaluate(u_pos)
                    grad_fd[i][j] = (total_reward_pos - total_reward) / eps
            
            abs_error = (grad_fd - grad).norm().detach().cpu().item()
            rel_error = abs_error / max(1e-7, min(grad_fd.norm().detach().cpu().item(), grad.norm().detach().cpu().item()))

            grad_fd_normalized = grad_fd / grad_fd.norm()
            grad_normalized = grad / grad.norm()
            dot_product = (grad_fd_normalized * grad_normalized).sum().detach().cpu().item()
            print('eps = {}, abs error = {}, rel error = {}, dot product = {}'.format(eps, abs_error, rel_error, dot_product))
            eps /= 10.

        print('exit')

        


