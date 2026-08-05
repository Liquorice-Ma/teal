import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial

from .FlowGNN import FlowGNN
from .TemporalEncoder import TemporalEncoder
from .utils import weight_initialization, print_


class TealActor(nn.Module):

    def __init__(
            self, teal_env, num_layer, model_dir, model_save, device,
            mask_mode='embed', gate=True, mask_init=57.0,
            std=1, log_std_min=-10.0, log_std_max=10.0):
        """Initialize teal actor.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
            model_dir: model save directory
            model_save: whether to save the model
            device: device id
            mask_mode: fill unobserved demands with learnable mask
                embedding ('embed'), zeros ('zero'), or the mean of
                observed demands ('mean', two-stage baseline)
            gate: whether to enable mask-aware gating in flowGNN
            std: std value, -1 if apply neuro networks for std
            log_std_min: lower bound for log std
            log_std_max: upper bound for log std
        """

        super(TealActor, self).__init__()

        # teal environment
        self.env = teal_env
        self.num_path = self.env.num_path

        # init FlowGNN
        self.device = device
        self.mask_mode = mask_mode
        self.FlowGNN = FlowGNN(self.env, num_layer, gate=gate).to(self.device)

        # learnable mask embedding for unobserved demands
        # replace masked-out entries instead of filling them with 0.
        # Must start away from zero: with a zeros init the embed branch is
        # numerically identical to mask_mode='zero', which silently turns
        # the ablation into a no-op. The scale matters too --- an init far
        # below the demand distribution is indistinguishable from zeros in
        # the model input. mask_init defaults to the median nonzero demand
        # of the trace (~57 Mbps on Starlink), which sits well below the
        # arithmetic mean (~159, used by the mean-interpolation baseline)
        # because the demand distribution is heavy-tailed.
        self.mask_embedding = nn.Parameter(
            mask_init*(0.75 + 0.5*torch.rand(
                self.num_path, device=self.device)))
        self.mask_init = mask_init
        # learnable scale for the neighbor-aware fill (mask_mode='nbr').
        # The neighbor mean sits at the arithmetic-mean scale (~159 Mbps on
        # Starlink), which the init sweep showed to over-estimate; the best
        # fill magnitude was the median (~57). Starting at 57/159 ~= 0.36
        # keeps the per-demand relative differences --- the reason to use
        # neighbors at all --- while placing the absolute level at the
        # measured optimum.
        self.mask_scale = nn.Parameter(
            torch.full((1,), 0.36, device=self.device))

        # temporal module enabled when hist_len > 1:
        # transformer over historical sparse traffic matrices, fused with
        # spatial embeddings through per-demand cross attention
        self.hist_len = self.env.hist_len
        self.d_spatial = self.num_path*(self.FlowGNN.num_layer+1)
        if self.hist_len > 1:
            self.d_fuse = 8
            self.TemporalEncoder = TemporalEncoder(
                self.hist_len).to(self.device)
            self.fuse_q = nn.Linear(
                self.TemporalEncoder.d_model, self.d_fuse).to(self.device)
            self.fuse_k = nn.Linear(
                self.FlowGNN.num_layer+1, self.d_fuse).to(self.device)
            self.fuse_v = nn.Linear(
                self.FlowGNN.num_layer+1, self.d_fuse).to(self.device)
            self.d_feature = self.d_spatial + self.num_path*self.d_fuse
        else:
            self.d_feature = self.d_spatial

        # init COMA policy
        self.std = std
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.mean_linear = nn.Linear(
            self.d_feature,
            self.num_path).to(self.device)
        # apply neuro networks for std
        if std < 0:
            self.log_std_linear = nn.Linear(
                self.d_feature,
                self.num_path).to(self.device)

        # get model fname
        self.model_fname = self.model_full_fname(
            model_dir, self.env.topo, num_layer, std)
        self.model_save = model_save
        # load model
        self.load_model()

    def model_full_fname(self, model_dir, topo, num_layer, std):
        """Return full name of the ML model."""

        return os.path.join(
            model_dir,
            "{}_flowGNN-{}_std-{}_obs-{}-{}_hist-{}_mask-{}_gate-{}.pt".format(
                topo, num_layer, std < 0,
                self.env.obs_type, self.env.obs_ratio, self.env.hist_len,
                self.mask_mode, self.FlowGNN.gate))

    def reinit_parameters(self):
        """Re-initialize every learnable parameter.

        Used for multi-restart initialization selection: on this problem a
        substantial fraction of random inits get stuck in a bad region
        (observed validation objectives of 2.4-3.2 against a normal 1.5),
        and such a run is unusable. reset_parameters() covers the standard
        torch modules (Linear, LayerNorm, MultiheadAttention), while the
        mask embedding is redrawn explicitly since it is a bare Parameter.
        """

        for module in self.modules():
            if module is not self and hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.apply(weight_initialization)
        with torch.no_grad():
            self.mask_embedding.copy_(
                self.mask_init*(0.75 + 0.5*torch.rand_like(
                    self.mask_embedding)))
            self.mask_scale.fill_(0.36)

    def load_model(self):
        """Load from model fname."""

        if os.path.exists(self.model_fname):
            print_(f'Loading Teal model from {self.model_fname}')
            if self.device.type == 'cpu':
                self.load_state_dict(
                    torch.load(
                        self.model_fname, map_location=torch.device('cpu')))
            else:
                self.load_state_dict(torch.load(self.model_fname))
        else:
            print_(f'Creating model {self.model_fname}')
            # weight initialization for neural networks
            self.apply(weight_initialization)

    def save_model(self):
        """Save from model fname."""

        if self.model_save:
            print_(f'Saving Teal model from {self.model_fname}')
            torch.save(self.state_dict(), self.model_fname)

    def forward(self, feature, tm_seq=None):
        """Return mean of normal distribution after forward propagation

        Args:
            feature: input features including capacity and demands
            tm_seq: historical sparse traffic matrices when hist_len > 1
        """
        x = self.FlowGNN(feature)

        # fuse temporal and spatial embeddings via cross attention:
        # temporal queries attend to spatial embeddings within each demand
        if self.hist_len > 1:
            h_temporal = self.TemporalEncoder(tm_seq)
            q = self.fuse_q(h_temporal).reshape(
                -1, self.num_path, self.d_fuse)
            k = self.fuse_k(x).reshape(-1, self.num_path, self.d_fuse)
            v = self.fuse_v(x).reshape(-1, self.num_path, self.d_fuse)
            attention = torch.softmax(
                q @ k.transpose(-2, -1) / math.sqrt(self.d_fuse), dim=-1)
            h_fuse = (attention @ v).reshape(-1, self.d_fuse)
            x = torch.cat([x, h_fuse], dim=-1)

        # num_path_node is read from env: it changes when the demand set
        # is rebuilt (demand_split) while weights remain shared
        x = x.reshape(
            self.env.num_path_node//self.num_path,
            self.d_feature)
        mean = self.mean_linear(x)

        # to apply neuro networks for std
        if self.std < 0:
            log_std = self.log_std_linear(x)
            log_std_clamped = torch.clamp(
                log_std,
                min=self.log_std_min,
                max=self.log_std_max)
            nn_std = torch.exp(log_std_clamped)
            return mean, nn_std
        # deterministic std
        else:
            return mean, self.std

    def evaluate(self, obs, deterministic=False):
        """Return raw action before softmax split ratio.

        Args:
            obs: sparse observation dict with capacity, tm_seq and mask
            deterministic: whether to have deterministic action
        """

        # latest sparse traffic matrix in the history window
        tm = obs['tm_seq'][-1]
        # demand-level mask expanded to path level
        path_mask = obs['mask'].repeat_interleave(self.num_path)
        # unobserved entries take the learnable mask embedding,
        # zeros, or the mean of observed demands
        tm = tm * path_mask
        if self.mask_mode == 'embed':
            tm = tm + \
                self.mask_embedding.repeat(
                    self.env.num_path_node//self.num_path) * (1 - path_mask)
        elif self.mask_mode == 'nbr':
            # neighbor-aware fill: per-demand estimate from observed demands
            # sharing the same source, scaled by a learnable factor. Unlike
            # 'embed', the fill differs across demands and therefore carries
            # information the policy can act on.
            est = self.env.neighbor_estimate(tm[::self.num_path])
            tm = tm + (self.mask_scale*est).repeat_interleave(
                self.num_path) * (1 - path_mask)
        elif self.mask_mode == 'mean':
            # mean interpolation: two-stage complete-then-optimize baseline
            tm = tm + tm.sum()/path_mask.sum() * (1 - path_mask)
        feature = torch.concat([obs['capacity'], tm]).reshape(-1, 1)
        mean, std = self.forward(feature, obs['tm_seq'])

        # test mode
        if deterministic:
            distribution = None
            raw_action = mean.detach()
            log_probability = None
        # train mode
        else:
            # use normal distribution for action
            distribution = Normal(mean, std)
            sample = distribution.rsample()
            raw_action = sample.detach()
            log_probability = distribution.log_prob(raw_action).sum(axis=-1)

        return raw_action, log_probability

    def act(self, obs):
        """Return split ratio as action with disabled gradient calculation.

        Args:
            obs: sparse observation dict with capacity, tm_seq and mask
        """

        with torch.no_grad():
            # deterministic action
            raw_action, _ = self.evaluate(obs, deterministic=True)
            return raw_action
