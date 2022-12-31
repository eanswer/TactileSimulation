
import sys, os

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

import a2c_ppo_acktr
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.storage import RolloutStorage

import envs
from utils import model

class PPO:
    def __init__(self, cfg):
        torch.set_num_threads(1) # TODO: check the effect

        self.seed = cfg['params']['general']['seed']
        self.env_name = cfg['params']['env']['name']
        self.device = cfg['params']['general'].get('device', 'cpu')
        env_params = copy.deepcopy(cfg['params']['env'])
        env_params.pop('name', None)
        
        set_random_seed(self.seed)

        # create render env
        self.render_env = gym.make(self.env_name, verbose = cfg['params']['general']['render'], **env_params)
        self.render_env.seed(self.seed)

        # create policy
        actor_critic_fn = getattr(model, cfg['params']['network']['actor_critic'])
        self.actor_critic = actor_critic_fn(self.render_env.observation_space.shape, self.render_env.action_space.shape[0], cfg['params']['network'])
        self.actor_critic.to(self.device)

        if cfg['params']['general']['train']:
            # total num steps
            self.total_env_steps = cfg['params']['config']['num_env_steps']
            self.num_steps = cfg['params']['config']['num_steps']
            self.num_processes = cfg['params']['config']['num_processes']
            self.use_linear_lr_decay = cfg['params']['config'].get('use_linear_lr_decay', True)
            self.use_proper_time_limits = cfg['params']['config'].get('use_proper_time_limits', True)
            self.lr = cfg['params']['config'].get('lr', 3e-4)
            self.use_gae = cfg['params']['config'].get('use_gae', False)
            self.gae_lambda = cfg['params']['config'].get('gae_lambda', 0.95)
            self.gamma = cfg['params']['config'].get('gamma', 0.99)
            
            # create envs
            self.envs = make_vec_envs(
                env_name = self.env_name,
                seed = self.seed,
                num_processes = self.num_processes,
                gamma = self.gamma,
                log_dir = None,
                device = self.device,
                allow_early_resets = False,
                norm_reward = cfg['params']['config'].get('norm_reward', True),
                norm_obs = cfg['params']['config'].get('norm_obs', True),
                clip_obs = cfg['params']['config'].get('clip_obs', 10.),
                clip_reward = cfg['params']['config'].get('clip_reward', 10.),
                **env_params)

            # create PPO agent
            self.agent = algo.PPO(
                actor_critic = self.actor_critic,
                clip_param = cfg['params']['config'].get('clip_param', 0.2),
                ppo_epoch = cfg['params']['config'].get('ppo_epoch', 10),
                num_mini_batch = cfg['params']['config'].get('num_mini_batch', 32),
                value_loss_coef = cfg['params']['config'].get('value_loss_coef', 0.5),
                entropy_coef = cfg['params']['config'].get('entropy_coef', 0.0),
                lr = self.lr,
                eps = cfg['params']['config'].get('eps', 1e-5),
                max_grad_norm = cfg['params']['config'].get('max_grad_norm', 0.5))

            # create rollout storage
            self.rollouts = RolloutStorage(
                num_steps = self.num_steps,
                num_processes = self.num_processes,
                obs_shape = self.envs.observation_space.shape, 
                action_space = self.envs.action_space,
                recurrent_hidden_state_size = self.actor_critic.recurrent_hidden_state_size)

            # interval-related arguments
            self.save_interval = cfg['params']['general'].get('save_interval', 50)
            self.log_interval = cfg['params']['general'].get('log_interval', 1)
            self.render_interval = cfg['params']['general'].get('render_interval', 20)
            
            # create logging folder
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)

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

    def train(self):
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)

        episode_rewards = deque(maxlen=200)
        episode_lens = deque(maxlen=200)

        start = time.time()
        num_updates = int(
            self.total_env_steps) // self.num_steps // self.num_processes
        
        last_eval_rew = 0.
        best_eval_rew = -1000.
        best_success_rate = -0.5
        best_len = np.inf

        for j in range(num_updates):

            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.agent.optimizer, j, num_updates,
                    self.lr)

            success_episode_cnt = 0
            total_episode_cnt = 0

            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])
                # Observe reward and next obs
                obs, reward, done, infos = self.envs.step(action)
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        episode_lens.append(info['episode']['l'])
                        total_episode_cnt += 1
                        if info['success'] == True:
                            success_episode_cnt += 1

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma,
                                    self.gae_lambda, self.use_proper_time_limits)
            
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

            self.rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % self.save_interval == 0 or j == num_updates - 1):
                model_save_dir = os.path.join(self.log_dir, 'models')
                os.makedirs(model_save_dir, exist_ok = True)
                torch.save([
                    self.actor_critic,
                    getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
                ], os.path.join(model_save_dir, 'model_iter{}'.format(j) + "_reward{:.1f}".format(np.mean(episode_rewards)) + ".pt"))

            # logging to console
            if j % self.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                end = time.time()
                # if np.mean(episode_rewards) > best_eval_rew:
                if success_episode_cnt / total_episode_cnt > best_success_rate + 0.002 or \
                    success_episode_cnt / total_episode_cnt > best_success_rate - 0.001 and np.mean(episode_lens) < best_len:
                    print_info(
                        "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:.1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                        "success rate {:.1f}% "
                        "dist_entropy {:.1f}, value_loss {:.4f}, action_loss {:.4f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                (end - start) / 60., 
                                len(episode_rewards), 
                                np.mean(episode_lens), np.median(episode_lens), 
                                np.min(episode_lens), np.max(episode_lens),
                                np.mean(episode_rewards), np.median(episode_rewards), 
                                np.min(episode_rewards), np.max(episode_rewards), 
                                (success_episode_cnt / total_episode_cnt) * 100.,
                                dist_entropy, value_loss,
                                action_loss))
                else:
                    print(
                        "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:.1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                        "success rate {:.1f}% "
                        "dist_entropy {:.1f}, value_loss {:.4f}, action_loss {:.4f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                (end - start) / 60., 
                                len(episode_rewards), 
                                np.mean(episode_lens), np.median(episode_lens), 
                                np.min(episode_lens), np.max(episode_lens),
                                np.mean(episode_rewards), np.median(episode_rewards), 
                                np.min(episode_rewards), np.max(episode_rewards), 
                                (success_episode_cnt / total_episode_cnt) * 100.,
                                dist_entropy, value_loss,
                                action_loss))

                # logging to writer
                time_elapse = end - start
                self.writer.add_scalar('rewards/step', np.mean(episode_rewards), total_num_steps)
                self.writer.add_scalar('rewards/time', np.mean(episode_rewards), time_elapse)
                self.writer.add_scalar('rewards/iter', np.mean(episode_rewards), j)
                self.writer.add_scalar('episode_lengths/step', np.mean(episode_lens), total_num_steps)
                self.writer.add_scalar('episode_lengths/time', np.mean(episode_lens), time_elapse)
                self.writer.add_scalar('episode_lengths/iter', np.mean(episode_lens), j)
                self.writer.add_scalar('success_rate/step', (success_episode_cnt / total_episode_cnt) * 100., total_num_steps)
                self.writer.add_scalar('success_rate/time', (success_episode_cnt / total_episode_cnt) * 100., time_elapse)
                self.writer.add_scalar('success_rate/iter', (success_episode_cnt / total_episode_cnt) * 100., j)
                self.writer.add_scalar('stats/value_loss/step', value_loss, total_num_steps)
                self.writer.add_scalar('stats/dist_entropy/step', dist_entropy, total_num_steps)
                self.writer.add_scalar('stats/action_loss/step', action_loss, total_num_steps)
                

                if success_episode_cnt / total_episode_cnt > best_success_rate + 0.002 or \
                    success_episode_cnt / total_episode_cnt > best_success_rate - 0.001 and np.mean(episode_lens) < best_len:
                    best_eval_rew = np.mean(episode_rewards)
                    best_success_rate = success_episode_cnt / total_episode_cnt
                    best_len = np.mean(episode_lens)
                    torch.save([
                        self.actor_critic,
                        getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
                    ], os.path.join(model_save_dir, 'best_model.pt'))
                    
            if (self.render_interval is not None and self.render_interval > 0 and (j + 1) % self.render_interval == 0):
                last_eval_rew, _, _, _, _ = self.play_once(self.actor_critic, getattr(self.envs, 'obs_rms', None))

            # save logs of every episode
            if (self.log_interval is not None and self.log_interval > 0 and (j + 1) % self.log_interval == 0):
                fp_log = open(self.training_log_path, 'a')
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                len_mean, len_min, len_max = np.mean(episode_lens), np.min(episode_lens), np.max(episode_lens)
                reward_mean, reward_min, reward_max = np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)
                fp_log.write('num_steps = {}, time = {}, mean(len) = {:.1f}, min(len) = {}, max(len) = {}, mean(reward) = {:.3f}, min(reward) = {:.3f}, max(reward) = {:.3f}, value_loss = {:.3f}, action_loss = {:.3f}, eval = {:.3f}, success_rate = {:.1f} \n'.format(
                    total_num_steps, end - start, len_mean, len_min, len_max, reward_mean, reward_min, reward_max, value_loss, action_loss, last_eval_rew, (success_episode_cnt / total_episode_cnt) * 100.))
                fp_log.close()
        
        self.writer.close()
        self.render_env.close()
        self.envs.close()

    def play_once(self, actor_critic, ob_rms = None, stochastic = False, mode = 'once', render = True, record = False):
        with torch.no_grad():
            ob = self.render_env.reset()
            if self.render_env.render_mode == 'step' and render:
                print('trial 0')
                if not record:
                    self.render_env.render(mode)
                else:
                    self.render_env.render('record')

            done = False
            total_reward = 0.
            success = False
            episode_len = 0
            improve_cnt = 0
            class_cnt = np.zeros((3, 3), dtype = int)
            class_improve_cnt = np.zeros((3, 3), dtype = int)
            class_success_cnt = np.zeros((3, 3), dtype = int)
            recurrent_hidden_states = torch.zeros((1, self.actor_critic.recurrent_hidden_state_size))
            points = []
            angles = []
            while not done:
                if ob_rms is not None:
                    ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10., 10.)
                ob = torch.Tensor(ob).unsqueeze(0).to(self.device)
                # recurrent_hidden_states = torch.zeros((1, self.actor_critic.recurrent_hidden_state_size))
                _, action, _, recurrent_hidden_states = actor_critic.act(ob, recurrent_hidden_states, torch.ones((1, 1)), deterministic = not stochastic)
                ob, reward, done, info = self.render_env.step(action.squeeze(0).detach().cpu().numpy())
                total_reward += reward
                episode_len += 1

                if self.render_env.render_mode == 'step' and render:
                    print('trial {}, reward = {}'.format(episode_len, reward))  
                    if not record:
                        self.render_env.render(mode)
                    else:
                        self.render_env.render('record')

                # determine the class
                class1, class2 = 0, 0
                if info['prev_object_pose'][0] < -0.00225:
                    class1 = 0
                elif info['prev_object_pose'][0] < 0.00225:
                    class1 = 1
                else:
                    class1 = 2
                if info['prev_object_pose'][1] < -0.00225:
                    class2 = 0
                elif info['prev_object_pose'][1] < 0.00225:
                    class2 = 1
                else:
                    class2 = 2
                class_cnt[class1][class2] += 1

                points.append(info['prev_object_pose'][0:2])
                angles.append(np.rad2deg(info['prev_object_pose'][2]))

                if info['success']:
                    success = True
                    class_success_cnt[class1][class2] += 1
                
                if info['improve']:
                    improve_cnt += 1
                    class_improve_cnt[class1][class2] += 1

            print('render: total reward = ', total_reward, ', success = ', info['success'])

            if self.render_env.render_mode == 'episode' and render:
                if mode != 'no_render':
                    if not record:
                        self.render_env.render(mode)
                    else:
                        self.render_env.render('record')
            
            extra_info = {}
            extra_info['class_cnt'] = class_cnt
            extra_info['class_improve_cnt'] = class_improve_cnt
            extra_info['class_success_cnt'] = class_success_cnt
            extra_info['points'] = points
            extra_info['angles'] = angles

            return total_reward, success, improve_cnt, episode_len, extra_info
    
    def play(self, cfg):
        self.actor_critic, ob_rms = torch.load(cfg['params']['general']['checkpoint'])
        num_games = cfg['params']['general']['num_games']
        
        print('policy std = ', self.actor_critic.actor_net.logstd.exp())
        games_cnt = 0
        success_cnt = 0
        improve_cnt = 0
        episode_len_sum = 0
        total_reward = 0.
        class_cnt = np.zeros((3, 3), dtype = int)
        class_improve_cnt = np.zeros((3, 3), dtype = int)
        class_success_cnt = np.zeros((3, 3), dtype = int)
        points = []
        angles = []
        while num_games != games_cnt:
            reward, success, improve_episode_cnt, episode_len, extra_info = self.play_once(self.actor_critic, ob_rms, cfg['params']['general']['stochastic'], \
                                            'loop', cfg['params']['general']['render'], cfg['params']['general']['record'])
            games_cnt += 1
            if success:
                success_cnt += 1
                episode_len_sum += episode_len
            total_reward += reward
            improve_cnt += improve_episode_cnt

            class_cnt = class_cnt + extra_info['class_cnt']
            class_improve_cnt = class_improve_cnt + extra_info['class_improve_cnt']
            class_success_cnt = class_success_cnt + extra_info['class_success_cnt']
            points = points + extra_info['points']
            angles = angles + extra_info['angles']
        
        print_info('[Summary] Avg reward = {:.3f}, Success rate = {:.2f}%, Avg success episode length = {:.2f}, Improve rate = {:.2f}%'.format(total_reward / games_cnt, success_cnt / games_cnt * 100., episode_len_sum / success_cnt, improve_cnt / np.sum(class_cnt) * 100.))

        print_info('-------------------------------------------------------------------------------------------------------------')

        for class1 in range(3):
            for class2 in range(3):
                class_success_rate = class_success_cnt[class1][class2] / max(class_cnt[class1][class2], 1) * 100.
                class_improve_rate = class_improve_cnt[class1][class2] / max(class_cnt[class1][class2], 1) * 100.
                print_info('Class [{}, {}], total cnt = {}, success rate = {:.3f}%, improve rate = {:.3f}%'.format(class1, class2, class_cnt[class1][class2], class_success_rate, class_improve_rate))
        
        points = np.array(points) * 1000.

        # draw grid
        fig, ax = plt.subplots(1, 2)
        color = 'black'
        # plt.plot([-10., -10.], [-10., 10.], c = color)
        # plt.plot([-2.25, -2.25], [-10., 10.], c = color)
        # plt.plot([2.25, 2.25], [-10., 10.], c = color)
        # plt.plot([10., 10.], [-10., 10.], c = color)
        # plt.plot([-10., 10.], [-10., -10.], c = color)
        # plt.plot([-10., 10.], [-2.25, -2.25], c = color)
        # plt.plot([-10., 10.], [2.25, 2.25], c = color)
        # plt.plot([-10., 10.], [10., 10.], c = color)
        ax[0].plot([-10., -10.], [-10., 10.], c = color)
        ax[0].plot([-2.25, -2.25], [-10., 10.], c = color)
        ax[0].plot([2.25, 2.25], [-10., 10.], c = color)
        ax[0].plot([10., 10.], [-10., 10.], c = color)
        ax[0].plot([-10., 10.], [-10., -10.], c = color)
        ax[0].plot([-10., 10.], [-2.25, -2.25], c = color)
        ax[0].plot([-10., 10.], [2.25, 2.25], c = color)
        ax[0].plot([-10., 10.], [10., 10.], c = color)

        # draw points
        ax[0].scatter(points[:, 0], points[:, 1])
        ax[0].set_title('misalignment distribution')

        # ax[1].scatter(list(np.arange(0, len(angles))), angles)
        ax[1].hist(angles, bins=20, edgecolor='black', facecolor='blue', alpha=0.7)
        ax[1].set_title('angle distribution')

        plt.show()