# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: In-Hand Object Rotation via Rapid Motor Adaptation
# Copyright (c) 2022 Haozhi Qi
# Licensed under MIT License
# https://github.com/HaozhiQi/hora/
# --------------------------------------------------------
import random
import numpy as np
import torch
import torch.nn as nn
from .block import TemporalConv


class MLP(nn.Module):
    def __init__(self, units, input_size, with_last_activation=True):
        super(MLP, self).__init__()
        # use with_last_activation=False when we need the network to output raw values before activation
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        if not with_last_activation:
            layers.pop()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
        
class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        policy_input_dim = kwargs.get('input_shape')[0]
        actions_num = kwargs.get('actions_num')
        self.units = kwargs.get('actor_units')
        self.priv_mlp = kwargs.get('priv_mlp_units')
        self.priv_dim = self.priv_mlp[-1]
        self.use_point_cloud_info = kwargs.get('use_point_cloud_info')
        self.point_mlp_units = kwargs.get('point_mlp_units')
        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        self.proprio_len = kwargs.get('proprio_len', 30)

        if self.priv_info:
            policy_input_dim += self.priv_mlp[-1]
            # the output of env_mlp and proprioceptive regression should both be before activation
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'], with_last_activation=False)
            if self.priv_info_stage2:
                # 스텝당 고유수용감각 = numActions(관절위치) + numActions(목표).
                # XHand 는 12+12=24 라 원본이 24 로 박아뒀지만, 20-DoF 손에서는 40 이어야 한다.
                # 안 맞추면 stage2(증류) 에서 에러 없이 관측이 잘린다.
                temporal_fusing_input_dim = kwargs.get('proprio_dim', 24)
                temporal_fusing_output_dim = 8
                if self.use_point_cloud_info:
                    temporal_fusing_output_dim += 32
                self.adapt_tconv = TemporalConv(temporal_fusing_input_dim, temporal_fusing_output_dim)

        if self.use_point_cloud_info:
            policy_input_dim += self.point_mlp_units[-1]
            self.point_mlp = MLP(units=self.point_mlp_units, input_size=3)

        self.actor_mlp = MLP(units=self.units, input_size=policy_input_dim)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        return mu, extrin, extrin_gt

    def _privileged_pred(self, joint_x, visual_x, tactile_x):
        # three part: modality specific transform, cross modality fusion, and temporal fusion
        # the order of the last two can be changed
        batch_dim, t_dim = joint_x.shape[:2]
        # ---- modality specific transform [*_x to *_t]
        joint_t = joint_x
        info_list = [joint_t]
        merge_t_t = torch.cat(info_list, dim=-1)
        extrin_pred = self.adapt_tconv(merge_t_t)

        return extrin_pred

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2: 
                extrin = self.adapt_tconv(obs_dict['proprio_hist']) # student's latent from hist
                # during supervised training, extrin has gt label
                if 'priv_info' in obs_dict:
                    extrin_gt = self.env_mlp(obs_dict['priv_info']) # teacher's latent from priv
                else:
                    extrin_gt = extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs_input = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin_gt = extrin

        if self.use_point_cloud_info:
            pcs = self.point_mlp(obs_dict['point_cloud_info'])
            pcs = torch.max(pcs, 1)[0]
            if self.priv_info_stage2:
                extrin_gt = torch.cat([extrin_gt, pcs], dim=-1)
            else:
                extrin = torch.cat([extrin, pcs], dim=-1)
                extrin_gt = extrin
        
        if not self.priv_info_stage2:
            extrin = torch.tanh(extrin)
            obs_input = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs_input)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        if prev_actions is not None:
            prev_neglogp = -distr.log_prob(prev_actions).sum(1)
            prev_neglogp = torch.squeeze(prev_neglogp)
        else:
            prev_neglogp = None
        result = {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result
