import torch
import torch.nn as nn
from utils import model_utils
from copy import deepcopy
import numpy as np

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean

class MLP(nn.Module):
    def __init__(self, in_dim, cfg_network):
        super(MLP, self).__init__()
        
        layer_sizes = cfg_network['layer_sizes']
        modules = []
        for i in range(len(layer_sizes)):
            modules.append(nn.Linear(in_dim, layer_sizes[i]))
            modules.append(model_utils.get_activation_func(cfg_network['activation']))
            if cfg_network.get('layernorm', False):
                modules.append(torch.nn.LayerNorm(layer_sizes[i]))
            in_dim = layer_sizes[i]

        self.body = nn.Sequential(*modules)
        self.out_features = layer_sizes[-1]

    def forward(self, inputs):
        return self.body(inputs)

class CNN(nn.Module):
    def __init__(self, input_shape, cfg_network):
        super(CNN, self).__init__()

        assert len(input_shape) == 3 # (feature_channels, rows, cols)

        in_channels = input_shape[0]

        kernel_sizes = cfg_network['kernel_sizes']
        layer_sizes = cfg_network['layer_sizes']
        stride_sizes = cfg_network['stride_sizes']
        hidden_size = cfg_network['hidden_size']
        image_shape = np.array(input_shape[1:])
        modules = []
        for i in range(len(kernel_sizes)):
            modules.append(nn.Conv2d(in_channels, layer_sizes[i], kernel_sizes[i], stride = stride_sizes[i]))
            modules.append(model_utils.get_activation_func(cfg_network['activation']))
            if cfg_network.get('layernorm', False):
                modules.append(nn.LayerNorm(layer_sizes[i]))
            in_channels = layer_sizes[i]
            image_shape = (image_shape - kernel_sizes[i]) // (stride_sizes[i]) + 1
        
        modules.append(nn.Flatten())
        modules.append(nn.Linear(image_shape[0] * image_shape[1] * in_channels, hidden_size))
        modules.append(model_utils.get_activation_func(cfg_network['activation']))

        self.body = nn.Sequential(*modules)
        self.out_features = hidden_size
    
    def forward(self, inputs):
        return self.body(inputs)

class CNNActor(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        self.feature_net = CNN(obs_shape, cfg_network['actor_cnn'])

        self.mean_net = nn.Linear(self.feature_net.out_features, action_dim)
        self.logstd = nn.Parameter(torch.ones(action_dim) * cfg_network.get('actor_logstd_init', -1.0))

    def forward(self, inputs):
        features = self.feature_net(inputs)
        mean = self.mean_net(features)
        std = self.logstd.expand_as(mean).exp()
        dist = FixedNormal(loc=mean, scale=std)
        
        return dist
    
    def act(self, inputs, deterministic = False):
        action_dist = self.forward(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.rsample()

        return action

class CNNCritic(nn.Module):
    def __init__(self,
                 obs_shape,
                 cfg_network):
        super().__init__()
        self.feature_net = CNN(obs_shape, cfg_network['critic_cnn'])

        self.value_net = nn.Linear(self.feature_net.out_features, 1)

    def forward(self, inputs):
        features = self.feature_net(inputs)
        values = self.value_net(features)
        
        return values
    
    def act(self, inputs, deterministic = False):
        action_dist = self.forward(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.rsample()

        return action

class DiagGaussianActor(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        
        self.feature_net = MLP(obs_shape[0], cfg_network['actor_mlp'])

        self.mean_net = nn.Linear(self.feature_net.out_features, action_dim)
        self.logstd = nn.Parameter(torch.ones(action_dim) * cfg_network.get('actor_logstd_init', -1.0))

    def forward(self, inputs):
        features = self.feature_net(inputs)
        mean = self.mean_net(features)
        std = self.logstd.expand_as(mean).exp()
        dist = FixedNormal(loc=mean, scale=std)
        
        return dist
    
    def act(self, inputs, deterministic = False):
        action_dist = self.forward(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.rsample()

        return action

class MLPCritic(nn.Module):
    def __init__(self,
                 obs_shape, 
                 cfg_network):
        super().__init__()
        self.feature_net = MLP(obs_shape[0], cfg_network['critic_mlp'])

        self.value_net = nn.Linear(self.feature_net.out_features, 1)

    def forward(self, inputs):
        features = self.feature_net(inputs)
        value = self.value_net(features)

        return value

class ActorCritic(nn.Module):
    def __init__(self, 
                 actor_net,
                 critic_net):

        super().__init__()
        self.actor_net = actor_net
        self.critic_net = critic_net
    
    def act(self, inputs, rnn_hxs = None, masks = None, deterministic = False):
        action_dist = self.actor_net(inputs)
        value = self.critic_net(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()
        
        action_log_probs = action_dist.log_probs(action)
        
        return value, action, action_log_probs, rnn_hxs
    
    def get_value(self, inputs, rnn_hxs = None, masks = None):
        value = self.critic_net(inputs)

        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        action_dist = self.actor_net(inputs)
        value = self.critic_net(inputs)

        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
        
    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1
    
class ActorCriticRNN(nn.Module):
    def __init__(self, 
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        self.feature_net = CNN(obs_shape, cfg_network['feature_cnn'])
        self.rnn_hidden_size = cfg_network['rnn_hidden_size']
        self.rnn_hidden_layers = cfg_network['rnn_hidden_layers']

        self.rnn = nn.GRU(self.feature_net.out_features, self.rnn_hidden_size, self.rnn_hidden_layers)
        self.actor_net = DiagGaussianActor((self.rnn_hidden_size, ), action_dim, cfg_network)
        self.critic_net = MLPCritic((self.rnn_hidden_size, ), cfg_network)

    def _forward_rnn(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            N = hxs.size(0)
            hxs = (hxs * masks).view(hxs.size(0), self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)
            x, hxs = self.rnn(x.unsqueeze(0), hxs)
            x = x.squeeze(0)
            hxs = hxs.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.unsqueeze(0)
            hxs = hxs.view(N, self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                
                rnn_scores, hxs = self.rnn(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.permute(1, 0, 2).contiguous().view(N, -1)

        return x, hxs

    def act(self, inputs, rnn_hxs, masks, deterministic = False):
        features = self.feature_net(inputs)
        rnn_output, rnn_hxs_new = self._forward_rnn(features, rnn_hxs, masks)

        action_dist = self.actor_net(rnn_output)
        value = self.critic_net(rnn_output)
        
        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()
        
        action_log_probs = action_dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs_new
    
    def get_value(self, inputs, rnn_hxs, masks):
        features = self.feature_net(inputs)
        rnn_output, _ = self._forward_rnn(features, rnn_hxs, masks)
        
        value = self.critic_net(rnn_output)
        
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        features = self.feature_net(inputs)
        rnn_output, rnn_hxs_new = self._forward_rnn(features, rnn_hxs, masks)

        action_dist = self.actor_net(rnn_output)
        value = self.critic_net(rnn_output)
        
        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs_new
    
    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.rnn_hidden_size * self.rnn_hidden_layers

class ActorCriticMLPRNN(nn.Module):
    def __init__(self, 
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        self.feature_net = MLP(obs_shape[0], cfg_network['feature_mlp'])
        self.rnn_hidden_size = cfg_network['rnn_hidden_size']
        self.rnn_hidden_layers = cfg_network['rnn_hidden_layers']

        self.rnn = nn.GRU(self.feature_net.out_features, self.rnn_hidden_size, self.rnn_hidden_layers)
        self.actor_net = DiagGaussianActor((self.rnn_hidden_size, ), action_dim, cfg_network)
        self.critic_net = MLPCritic((self.rnn_hidden_size, ), cfg_network)

    def _forward_rnn(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            N = hxs.size(0)
            hxs = (hxs * masks).view(hxs.size(0), self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)
            x, hxs = self.rnn(x.unsqueeze(0), hxs)
            x = x.squeeze(0)
            hxs = hxs.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.unsqueeze(0)
            hxs = hxs.view(N, self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                
                rnn_scores, hxs = self.rnn(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.permute(1, 0, 2).contiguous().view(N, -1)

        return x, hxs

    def act(self, inputs, rnn_hxs, masks, deterministic = False):
        features = self.feature_net(inputs)
        rnn_output, rnn_hxs_new = self._forward_rnn(features, rnn_hxs, masks)

        action_dist = self.actor_net(rnn_output)
        value = self.critic_net(rnn_output)
        
        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()
        
        action_log_probs = action_dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs_new
    
    def get_value(self, inputs, rnn_hxs, masks):
        features = self.feature_net(inputs)
        rnn_output, _ = self._forward_rnn(features, rnn_hxs, masks)
        
        value = self.critic_net(rnn_output)
        
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        features = self.feature_net(inputs)
        rnn_output, rnn_hxs_new = self._forward_rnn(features, rnn_hxs, masks)

        action_dist = self.actor_net(rnn_output)
        value = self.critic_net(rnn_output)
        
        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs_new
    
    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.rnn_hidden_size * self.rnn_hidden_layers