
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
        self.render_env = gym.make(self.env_name, verbose = cfg['params']['general']['render'], render_tactile = True, **env_params)
        self.render_env.seed(self.seed)

        # create policy
        actor_fn = getattr(model, cfg['params']['network'].get('actor', 'DiagGaussianPolicy'))
        critic_fn = getattr(model, cfg['params']['network'].get('critic', 'MLPCritic'))
        
        actor = actor_fn(self.render_env.observation_space.shape, self.render_env.action_space.shape[0], cfg['params']['network'])
        critic = critic_fn(self.render_env.observation_space.shape, cfg['params']['network'])

        self.actor_critic = model.ActorCritic(actor, critic)
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
        best_eval_rew = -np.inf
        best_len = np.inf

        for j in range(num_updates):
            cur_episode_rewards = []
            cur_episode_lens = []
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    self.agent.optimizer, j, num_updates,
                    self.lr)

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
                        cur_episode_rewards.append(info['episode']['r'])
                        cur_episode_lens.append(info['episode']['l'])

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

            # mean_rewards = np.mean(cur_episode_rewards)
            mean_rewards = np.mean(episode_rewards)
            
            # save for every interval-th episode or for the last epoch
            if (j % self.save_interval == 0 or j == num_updates - 1):
                model_save_dir = os.path.join(self.log_dir, 'models')
                os.makedirs(model_save_dir, exist_ok = True)
                torch.save([
                    self.actor_critic,
                    getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
                ], os.path.join(model_save_dir, 'model_iter{}'.format(j) + "_reward{:.1f}".format(mean_rewards) + ".pt"))

            # logging to console
            if j % self.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                end = time.time()
                if mean_rewards > best_eval_rew:
                    print_info(
                        "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:.1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                        "dist_entropy {:.1f}, value_loss {:.4f}, action_loss {:.4f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                (end - start) / 60., 
                                len(episode_rewards), 
                                np.mean(episode_lens), np.median(episode_lens), 
                                np.min(episode_lens), np.max(episode_lens),
                                mean_rewards, np.median(episode_rewards), 
                                np.min(episode_rewards), np.max(episode_rewards),
                                dist_entropy, value_loss,
                                action_loss))
                else:
                    print(
                        "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:.1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                        "dist_entropy {:.1f}, value_loss {:.4f}, action_loss {:.4f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                (end - start) / 60., 
                                len(episode_rewards), 
                                np.mean(episode_lens), np.median(episode_lens), 
                                np.min(episode_lens), np.max(episode_lens),
                                mean_rewards, np.median(episode_rewards), 
                                np.min(episode_rewards), np.max(episode_rewards),
                                dist_entropy, value_loss,
                                action_loss))

                # logging to writer
                time_elapse = end - start
                self.writer.add_scalar('rewards/step', mean_rewards, total_num_steps)
                self.writer.add_scalar('rewards/time', mean_rewards, time_elapse)
                self.writer.add_scalar('rewards/iter', mean_rewards, j)
                self.writer.add_scalar('episode_lengths/step', np.mean(episode_lens), total_num_steps)
                self.writer.add_scalar('episode_lengths/time', np.mean(episode_lens), time_elapse)
                self.writer.add_scalar('episode_lengths/iter', np.mean(episode_lens), j)
                self.writer.add_scalar('stats/value_loss/step', value_loss, total_num_steps)
                self.writer.add_scalar('stats/dist_entropy/step', dist_entropy, total_num_steps)
                self.writer.add_scalar('stats/action_loss/step', action_loss, total_num_steps)
                

                if mean_rewards > best_eval_rew:
                    best_eval_rew = mean_rewards
                    torch.save([
                        self.actor_critic,
                        getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
                    ], os.path.join(model_save_dir, 'best_model.pt'))
                    
            if (self.render_interval is not None and self.render_interval > 0 and (j + 1) % self.render_interval == 0):
                last_eval_rew, _, _ = self.play_once(self.actor_critic, getattr(self.envs, 'obs_rms', None))

            # save logs of every episode
            if (self.log_interval is not None and self.log_interval > 0 and (j + 1) % self.log_interval == 0):
                fp_log = open(self.training_log_path, 'a')
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                len_mean, len_min, len_max = np.mean(episode_lens), np.min(episode_lens), np.max(episode_lens)
                reward_mean, reward_min, reward_max = mean_rewards, np.min(episode_rewards), np.max(episode_rewards)
                fp_log.write('num_steps = {}, time = {}, mean(len) = {:.1f}, min(len) = {}, max(len) = {}, mean(reward) = {:.3f}, min(reward) = {:.3f}, max(reward) = {:.3f}, value_loss = {:.3f}, action_loss = {:.3f}, eval = {:.3f} \n'.format(
                    total_num_steps, end - start, len_mean, len_min, len_max, reward_mean, reward_min, reward_max, value_loss, action_loss, last_eval_rew))
                fp_log.close()
        
        self.writer.close()
        self.render_env.close()
        self.envs.close()

    def play_once(self, actor_critic, ob_rms = None, stochastic = False, mode = 'once', render = True, record = False):
        with torch.no_grad():
            ob = self.render_env.reset()

            done = False
            total_reward = 0.
            episode_len = 0
            reward_details = {}
            while not done:
                if ob_rms is not None:
                    ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10., 10.)
                ob = torch.Tensor(ob).unsqueeze(0).to(self.device)
                _, action, _, _ = actor_critic.act(ob, None, None, deterministic = not stochastic)
                ob, reward, done, info = self.render_env.step(action.squeeze(0).detach().cpu().numpy())
                total_reward += reward

                for key in info:
                    if key[0:6] == 'reward':
                        if key not in reward_details:
                            reward_details[key] = 0.
                        reward_details[key] += info[key].detach().cpu().item()
                    elif key[0:5] == 'final':
                        if done:
                            reward_details[key] = info[key].detach().cpu().item()

                episode_len += 1

            print_ok('render: total reward = ', total_reward, ', len = ', episode_len, reward_details)

            if render:
                if mode != 'no_render':
                    if not record:
                        self.render_env.render(mode)
                    else:
                        self.render_env.render('record')

            extra_info = {}
            extra_info['reward_details'] = reward_details

            return total_reward, episode_len, extra_info
    
    def play(self, cfg):
        self.actor_critic, ob_rms = torch.load(cfg['params']['general']['checkpoint'])
        num_games = cfg['params']['general']['num_games']
        
        print('policy std = ', self.actor_critic.actor_net.logstd.exp())
        games_cnt = 0
        total_reward = 0.
        total_reward_details = {}
        while num_games != games_cnt:
            reward, episode_len, extra_info = self.play_once(self.actor_critic, ob_rms, cfg['params']['general']['stochastic'], \
                                            'loop', cfg['params']['general']['render'], cfg['params']['general']['record'])
            games_cnt += 1

            total_reward += reward
            reward_details = extra_info['reward_details']
            for key in reward_details:
                if key not in total_reward_details:
                    total_reward_details[key] = 0.
                total_reward_details[key] += reward_details[key]
        
        for key in total_reward_details:
            total_reward_details[key] = total_reward_details[key] / num_games
            
        print_info('[Summary] Avg reward = {:.3f}, details = {}'.format(total_reward / games_cnt, total_reward_details))

        print_info('-------------------------------------------------------------------------------------------------------------')