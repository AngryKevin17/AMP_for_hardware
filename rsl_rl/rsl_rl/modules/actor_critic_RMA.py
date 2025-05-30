# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticRMA(nn.Module):
    def __init__(self,  num_actor_obs, 
                        num_actions, 
                        num_privileged_obs, 
                        num_embedding, 
                        num_obs_stacking, 
                        activation = "elu", 
                        enc_activation = "elu",
                        init_noise_std=1.0,
                        fixed_std=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()
        
        self.activation = get_activation(activation)
        self.activation_encoder = get_activation(enc_activation)
        self.actor_out_dim = num_actions
        self.tsteps = num_obs_stacking

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_actor_obs + num_embedding, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_actions),
        )
        self.privileged_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )
        self.adaptation_module = torch.nn.Sequential(
            torch.nn.Linear(num_actor_obs * num_obs_stacking, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )
        self.privileged_decoder = torch.nn.Sequential(
            torch.nn.Linear(num_embedding, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_privileged_obs),
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Adaptation MLP: {self.adaptation_module}")
        print(f"Decoder MLP: {self.privileged_decoder}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, encoder_observations, **kwargs):
        priv_embedding = self.adaptation_module(encoder_observations)
        observations = torch.cat((observations, priv_embedding), dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, encoder_observations):
        priv_embedding = self.adaptation_module(encoder_observations)
        observations = torch.cat((observations, priv_embedding), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None