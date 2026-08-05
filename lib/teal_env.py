import pickle
import json
import os
import math
import time
import random
from itertools import product

from networkx.readwrite import json_graph

import torch
import torch_scatter
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from .config import TOPOLOGIES_DIR
from .ADMM import ADMM
from .path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles


class TealEnv(object):

    def __init__(
            self, obj, topo, problems,
            num_path, edge_disjoint, dist_metric, rho,
            train_size, val_size, test_size, num_failure, device,
            obs_ratio=1.0, obs_type='flow', obs_sample='uniform',
            hist_len=1, prune_demands=False,
            demand_split=False, test_topo=None,
            num_reward_edge=1, reward_temperature=0.01,
            raw_action_min=-10.0, raw_action_max=10.0):
        """Initialize Teal environment.

        Args:
            obj: objective
            topo: topology name
            problems: problem list
            num_path: number of paths per demand
            edge_disjoint: whether edge-disjoint paths
            dist_metric: distance metric for shortest paths
            rho: hyperparameter for the augumented Lagranian
            train size: train start index, stop index
            val size: val start index, stop index
            test size: test start index, stop index
            num_failure: number of edge failures
            device: device id
            obs_ratio: ratio of observed node pairs in sparse observation
            obs_type: sampling granularity, 'flow' samples demand pairs,
                'node' samples source nodes with all outgoing demands observed
            obs_sample: 'uniform' random sampling, or 'top' which meters
                the largest flows (80\% by training-slice mean volume,
                20\% random), following TEST's protocol
            hist_len: number of historical traffic matrices as model input
            prune_demands: only keep node pairs with nonzero demand in any
                traffic matrix (for sparse satellite traffic)
            demand_split: build demand set from training-slice TMs and
                rebuild from test-slice TMs at test time (zero-retraining
                generalization), requires prune_demands
            test_topo: alternative topology json used at test time only;
                the graph, paths and ADMM are rebuilt while model weights
                stay unchanged (topology-drift zero-retraining)
            num_reward_edge: number of most-congested links that receive
                credit in the MLU reward (1 recovers the original argmax)
            reward_temperature: softmax temperature over the top-k link
                utilizations when num_reward_edge > 1
            raw_action_min: min value when clamp raw action
            raw_action_max: max value when clamp raw action
        """

        self.obj = obj
        self.topo = topo
        self.problems = problems
        self.num_path = num_path
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

        self.train_start, self.train_stop = train_size
        self.val_start, self.val_stop = val_size
        self.test_start, self.test_stop = test_size
        self.num_failure = num_failure
        self.device = device

        # sparse observation: observed ratio and history window length
        self.obs_ratio = obs_ratio
        self.obs_type = obs_type
        self.obs_sample = obs_sample
        self.hist_len = hist_len

        # softened-max reward for the MLU objective: number of congested
        # links that receive credit, and the softmax temperature over their
        # utilizations (temperature -> 0 recovers the plain argmax)
        self.num_reward_edge = num_reward_edge
        self.reward_temperature = reward_temperature

        # init matrices related to topology; topo_active tracks the graph
        # currently loaded (training topo, or test_topo during testing)
        self.test_topo = test_topo
        self.topo_active = topo
        self.G = self._read_graph_json(topo)
        self.capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)
        # demand pairs: all node pairs, or pairs with nonzero demand
        # in any traffic matrix when prune_demands is enabled;
        # when demand_split is enabled, scan only the current slice so that
        # the test-time demand set is rebuilt from test TMs
        self.prune_demands = prune_demands
        self.demand_split = demand_split
        self.rho = rho
        self.pair_range = (self.train_start, self.train_stop) \
            if demand_split else None
        self._build_graph(
            self._get_demand_pairs(prune_demands, self.pair_range))

        # min/max value when clamp raw action
        self.raw_action_min = raw_action_min
        self.raw_action_max = raw_action_max

        self.reset('train')

    def _build_graph(self, demand_pairs):
        """(Re)build all structures derived from the demand-pair set:
        index tensors, path graph, observation mask and ADMM.
        Model weights are independent of the demand-set size, so the graph
        can be rebuilt at test time without retraining.
        """

        self.demand_pairs = demand_pairs
        self.num_demand = len(demand_pairs)
        self.num_path_node = self.num_path * self.num_demand
        self.demand_src = torch.LongTensor(
            [s for s, t in self.demand_pairs])
        self.demand_dst = torch.LongTensor(
            [t for s, t in self.demand_pairs])
        self.edge_index, self.edge_index_values, self.p2e = \
            self.get_topo_matrix(
                self.topo_active, self.num_path, self.edge_disjoint,
                self.dist_metric)

        # binary mask over demands: 1 = observed, 0 = unobserved
        # fixed across train/val/test for reproducibility
        self.obs_mask = self._init_obs_mask()

        # init ADMM
        self.ADMM = ADMM(
            self.p2e, self.num_path, self.num_path_node,
            self.num_edge_node, self.rho, self.device)

    def reset(self, mode='test'):
        """Reset the initial conditions in the beginning."""

        if mode == 'train':
            self.idx_start, self.idx_stop = self.train_start, self.train_stop
        elif mode == 'test':
            self.idx_start, self.idx_stop = self.test_start, self.test_stop
        else:
            self.idx_start, self.idx_stop = self.val_start, self.val_stop
        # rebuild demand set from the TMs of the current slice when
        # demand_split is enabled; val reuses the training demand set
        rebuild = False
        if self.demand_split:
            pair_range = (self.test_start, self.test_stop) if mode == 'test' \
                else (self.train_start, self.train_stop)
            if pair_range != self.pair_range:
                self.pair_range = pair_range
                rebuild = True
        # switch to the perturbed test topology at test time and back to
        # the training topology otherwise (topology-drift zero-retraining)
        if self.test_topo is not None:
            target = self.test_topo if mode == 'test' else self.topo
            if target != self.topo_active:
                self.topo_active = target
                self.G = self._read_graph_json(target)
                self.capacity = torch.FloatTensor(
                    [float(c_e) for u, v, c_e
                        in self.G.edges.data('capacity')])
                self.num_edge_node = len(self.G.edges)
                rebuild = True
        if rebuild:
            self._build_graph(self._get_demand_pairs(
                self.prune_demands, self.pair_range))
        self.idx = self.idx_start
        self.obs = self._read_obs()

    def _get_demand_pairs(self, prune_demands, pair_range=None):
        """Return list of demand pairs in source-major order.
        When prune_demands is enabled, only keep node pairs with nonzero
        demand in any traffic matrix within pair_range (all when None).
        """

        num_node = self.G.number_of_nodes()
        if not prune_demands:
            return [(s, t) for s in range(num_node) for t in range(num_node)
                    if s != t]
        # union of nonzero demands across traffic matrices
        problems = self.problems if pair_range is None \
            else self.problems[pair_range[0]:pair_range[1]]
        nonzero = None
        for _, _, tm_fname in problems:
            with open(tm_fname, 'rb') as f:
                tm = pickle.load(f)
            nonzero = (tm > 0) if nonzero is None else nonzero | (tm > 0)
        return [(s, t) for s in range(num_node) for t in range(num_node)
                if s != t and nonzero[s][t]]

    def _init_obs_mask(self):
        """Return binary mask over demands from sampling with obs_ratio.
        Mask is generated once with a fixed seed so that the same node
        pairs are observed in training and testing.
        flow-level: sample demand pairs directly;
        node-level: sample source nodes that meter all outgoing demands.
        """

        if self.obs_ratio >= 1.0:
            return torch.ones(self.num_demand).to(self.device)
        generator = torch.Generator().manual_seed(0)
        if self.obs_type == 'node':
            num_node = self.G.number_of_nodes()
            num_observed = max(1, int(round(num_node * self.obs_ratio)))
            idx_observed = torch.randperm(
                num_node, generator=generator)[:num_observed]
            node_observed = torch.zeros(num_node, dtype=torch.bool)
            node_observed[idx_observed] = True
            mask = torch.FloatTensor(
                [1. if node_observed[s] else 0.
                    for s, t in self.demand_pairs])
        elif self.obs_sample == 'top':
            # TEST-style: 80% of the budget meters the largest flows by
            # training-slice mean volume, 20% is random exploration
            num_observed = max(1, int(round(self.num_demand * self.obs_ratio)))
            num_top = int(round(num_observed * 0.8))
            mean_demand = torch.zeros(self.num_demand)
            for _, _, tm_fname in self.problems[
                    self.train_start:self.train_stop]:
                with open(tm_fname, 'rb') as f:
                    tm = pickle.load(f)
                mean_demand += torch.FloatTensor(tm)[
                    self.demand_src, self.demand_dst]
            top_idx = torch.argsort(mean_demand, descending=True)[:num_top]
            mask = torch.zeros(self.num_demand)
            mask[top_idx] = 1
            rest = torch.nonzero(mask == 0).flatten()
            rand_idx = rest[torch.randperm(
                rest.numel(), generator=generator)[:num_observed - num_top]]
            mask[rand_idx] = 1
        else:
            num_observed = max(1, int(round(self.num_demand * self.obs_ratio)))
            idx_observed = torch.randperm(
                self.num_demand, generator=generator)[:num_observed]
            mask = torch.zeros(self.num_demand)
            mask[idx_observed] = 1
        return mask.to(self.device)

    def neighbor_estimate(self, tm_demand):
        """Return a per-demand fill value derived from observed neighbors.

        A single learnable placeholder shared by every unobserved demand
        carries no discriminative information: the policy applies a softmax
        over each demand's paths, so adding the same offset to all of them
        leaves the split ratios essentially unchanged (measured: 0.05%
        difference against plain zero-filling). Instead, an unobserved
        demand (s,t) is filled with the mean of the observed demands that
        share its source s --- satellites serve a common geographic
        footprint, so their egress volumes correlate. Sources without any
        observed demand fall back to the global mean of observed traffic.

        Args:
            tm_demand: demand-level traffic of the current snapshot
        """

        num_node = self.G.number_of_nodes()
        observed = tm_demand * self.obs_mask
        src_sum = torch_scatter.scatter(
            observed, self.demand_src, dim_size=num_node)
        src_cnt = torch_scatter.scatter(
            self.obs_mask, self.demand_src, dim_size=num_node)
        src_mean = src_sum/src_cnt.clamp(min=1)
        est = src_mean[self.demand_src]
        global_mean = observed.sum()/self.obs_mask.sum().clamp(min=1)
        return torch.where(
            src_cnt[self.demand_src] > 0, est, global_mean)

    def get_obs(self):
        """Return sparse observation for the model:
        (capacity + historical sparse traffic matrices + mask).
        Unobserved demands are NOT filled with 0 here; they are marked by
        the mask and replaced with a learnable mask embedding in the actor.
        Full traffic matrix in self.obs is reserved for reward and ADMM.
        """

        return {
            'capacity': self.obs[:-self.num_path_node],
            'tm_seq': torch.stack(self.hist_tms),
            'mask': self.obs_mask,
        }

    def _read_obs(self):
        """Return observation (capacity + traffic matrix) from files."""

        topo, topo_fname, tm_fname = self.problems[self.idx]
        tm = self._read_tm(tm_fname)
        obs = torch.concat([self.capacity, tm]).to(self.device)
        # simulate link failures in testing
        if self.num_failure > 0 and self.idx_start == self.test_start:
            idx_failure = torch.tensor(
                random.sample(range(self.num_edge_node),
                self.num_failure)).to(self.device)
            obs[idx_failure] = 0
        # sparse historical traffic matrices for the model input
        self.hist_tms = self._read_hist_tms()
        return obs

    def _read_tm(self, tm_fname):
        """Return flattened traffic matrix on demand pairs from file."""

        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)
        # select demand pairs and repeat for each path
        tm = torch.FloatTensor(tm)[self.demand_src, self.demand_dst]
        return tm.repeat_interleave(self.num_path)

    def _read_hist_tms(self):
        """Return sparse historical traffic matrices ending at current idx.
        Traffic matrices are indexed in time order; take hist_len steps
        backwards within the current slice and mask unobserved demands.
        Masked entries are zeroed only as placeholders: the actor should
        replace them with a learnable mask embedding based on the mask.
        """

        hist_tms = []
        # demand-level mask expanded to path level
        path_mask = self.obs_mask.repeat_interleave(self.num_path)
        for step_back in range(self.hist_len - 1, -1, -1):
            idx = self.idx - step_back
            # pad with the earliest tm in the slice at the beginning
            if idx < self.idx_start:
                idx = self.idx_start
            _, _, tm_fname = self.problems[idx]
            tm = self._read_tm(tm_fname).to(self.device)
            hist_tms.append(tm * path_mask)
        return hist_tms

    def _next_obs(self):
        """Return next observation (capacity + traffic matrix)."""

        self.idx += 1
        if self.idx == self.idx_stop:
            self.idx = self.idx_start
        self.obs = self._read_obs()
        return self.obs

    def render(self):
        """Return a dictionary for the details of the current problem"""

        topo, topo_fname, tm_fname = self.problems[self.idx]
        problem_dict = {
            'problem_name': topo,
            'obj': self.obj,
            'tm_fname': tm_fname.split('/')[-1],
            'num_node': self.G.number_of_nodes(),
            'num_edge': self.G.number_of_edges(),
            'num_path': self.num_path,
            'edge_disjoint': self.edge_disjoint,
            'dist_metric': self.dist_metric,
            'traffic_model': tm_fname.split('/')[-2],
            'traffic_seed': int(tm_fname.split('_')[-3]),
            'scale_factor': float(tm_fname.split('_')[-2]),
            'total_demand': self.obs[
                -self.num_path_node::self.num_path].sum().item(),
        }
        return problem_dict

    def step(self, raw_action, num_sample=0, num_admm_step=0):
        """Return the reward of current action.

        Args:
            raw_action: raw action from actor
            num_sample: number of samples for reward during training
            num_admm_step: number of ADMM steps during testing
        """

        info = {}
        if self.idx_start == self.train_start:
            reward = self.take_action(raw_action, num_sample)
        else:
            start_time = time.time()
            action = self.transform_raw_action(raw_action)
            if self.obj == 'total_flow':
                # total flow require no constraint violation
                action = self.ADMM.tune_action(self.obs, action, num_admm_step)
                action = self.round_action(action)
            elif self.obj == 'min_max_link_util' and num_admm_step > 0:
                # conservation-preserving rebalancing (cf. rebalance_action)
                action = self.rebalance_action(
                    action, num_iter=10*num_admm_step)
            info['runtime'] = time.time() - start_time
            info['sol_mat'] = self.extract_sol_mat(action)
            reward = self.get_obj(action)

        # next observation
        self._next_obs()
        return reward, info

    def rebalance_action(self, action, num_iter=20, step=0.5):
        """Return rebalanced action for the MLU objective.

        Conservation-preserving repair: iteratively moves a fraction of
        flow from each demand's most congested path (by bottleneck link
        utilization) to its least congested one. Per-demand totals stay
        unchanged, so MLU cannot be lowered by under-serving traffic ---
        unlike ADMM, whose slack on the demand constraint drops flow.

        Args:
            action: path flows (num_path_node)
            num_iter: number of rebalancing iterations
            step: fraction of the donor path's flow moved per iteration
        """

        capacity = self.obs[:-self.num_path_node]
        num_demand = self.num_path_node//self.num_path
        idx = torch.arange(num_demand).to(self.device)
        for _ in range(num_iter):
            edge_flow = torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1],
                dim_size=self.num_edge_node)
            util = edge_flow/capacity
            # bottleneck utilization of each path
            path_util = torch_scatter.scatter_max(
                util[self.p2e[1]], self.p2e[0],
                dim_size=self.num_path_node)[0].reshape(-1, self.num_path)
            donor = path_util.argmax(-1)
            receiver = path_util.argmin(-1)
            # move only when it strictly improves the demand's bottleneck
            gain = path_util[idx, donor] - path_util[idx, receiver] > 1e-6
            donor = idx*self.num_path + donor
            receiver = idx*self.num_path + receiver
            move = step*action[donor]*gain
            action = action.clone()
            action[donor] -= move
            action[receiver] += move
        return action

    def get_obj(self, action):
        """Return objective."""

        if self.obj == 'total_flow':
            return action.sum(axis=-1)
        elif self.obj == 'min_max_link_util':
            return (torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1]
                )/self.obs[:-self.num_path_node]).max()

    def transform_raw_action(self, raw_action):
        """Return network flow allocation as action.

        Args:
            raw_action: raw action directly from ML output
        """
        # clamp raw action between raw_action_min and raw_action_max
        raw_action = torch.clamp(
            raw_action, min=self.raw_action_min, max=self.raw_action_max)

        # translate ML output to split ratio through softmax
        raw_action = raw_action.exp()
        if self.obj == 'min_max_link_util':
            # flow conservation: ratios sum to one (no unserved residual),
            # preventing artificially low MLU via under-serving demands
            raw_action = raw_action/raw_action.sum(axis=-1)[:, None]
        else:
            # 1 in softmax represents unallocated traffic
            raw_action = raw_action/(1+raw_action.sum(axis=-1)[:, None])

        # translate split ratio to flow
        raw_action = raw_action.flatten() * self.obs[-self.num_path_node:]

        return raw_action

    def round_action(
            self, action, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """Return rounded action.
        Action can still violate constraints even after ADMM fine-tuning.
        This function rounds the action through cutting flow.

        Args:
            action: input action
            round_demand: whether to round action for demand constraints
            round_capacity: whether to round action for capacity constraints
            num_round_iter: number of rounds when iteratively cutting flow
        """

        demand = self.obs[-self.num_path_node::self.num_path]
        capacity = self.obs[:-self.num_path_node]

        # reduce action proportionally if action exceed demand
        if round_demand:
            action = action.reshape(-1, self.num_path)
            ratio = action.sum(-1) / demand
            action[ratio > 1, :] /= ratio[ratio > 1, None]
            action = action.flatten()

        # iteratively reduce action proportionally if action exceed capacity
        if round_capacity:
            path_flow = action
            path_flow_allocated_total = torch.zeros(path_flow.shape)\
                .to(self.device)
            for round_iter in range(num_round_iter):
                # flow on each edge
                edge_flow = torch_scatter.scatter(
                    path_flow[self.p2e[0]], self.p2e[1])
                # util of each edge
                util = 1 + (edge_flow/capacity-1).relu()
                # propotionally cut path flow by max util
                util = torch_scatter.scatter(
                    util[self.p2e[1]], self.p2e[0], reduce="max")
                path_flow_allocated = path_flow/util
                # update total allocation, residual capacity, residual flow
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[self.p2e[0]], self.p2e[1])).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total

        return action

    def take_action(self, raw_action, num_sample):
        '''Return an approximate reward for action for each node pair.
        To make function fast and scalable on GPU, we only calculate delta.
        We assume when changing action in one node pair:
        (1) The change in edge utilization is very small;
        (2) The bottleneck edge in a path does not change due to (1).
        For evary path after change:
            path_flow/max(util, 1) =>
            (path_flow+delta_path_flow)/max(util+delta_util, 1)
            if util < 1:
                reward = - delta_path_flow
            if util > 1:
                reward = - delta_path_flow/(util+delta_util)
                    + path_flow*delta_util/(util+delta_util)/util
                    approx delta_path_flow/util - path_flow/util^2*delta_util

        Args:
            raw_action: raw action from policy network
            num_sample: number of samples in estimating reward
        '''

        path_flow = self.transform_raw_action(raw_action)
        edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])
        util = edge_flow/self.obs[:-self.num_path_node]

        # sample from uniform distribution [mean_min, min_max]
        distribution = Uniform(
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_min,
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_max)
        reward = torch.zeros(self.num_path_node//self.num_path).to(self.device)

        if self.obj == 'total_flow':

            # find bottlenack edge for each path
            util, path_bottleneck = torch_scatter.scatter_max(
                util[self.p2e[1]], self.p2e[0])
            path_bottleneck = self.p2e[1][path_bottleneck]

            # prepare -path_flow/util^2 for reward
            coef = path_flow/util**2
            coef[util < 1] = 0
            coef = torch_scatter.scatter(
                coef, path_bottleneck).reshape(-1, 1)

            # prepare path_util to bottleneck edge_util
            bottleneck_p2e = torch.sparse_coo_tensor(
                self.p2e, (1/self.obs[:-self.num_path_node])[self.p2e[1]],
                [self.num_path_node, self.num_edge_node])

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                # add -delta_path_flow if util < 1 else -delta_path_flow/util
                delta_path_flow = self.transform_raw_action(sample) - path_flow
                reward += -(delta_path_flow/(1+(util-1).relu()))\
                    .reshape(-1, self.num_path).sum(-1)

                # add path_flow/util^2*delta_util for each path
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                # get utilization changes on edge
                # do not use torch_sparse.spspmm()
                # "an illegal memory access was encountered" in large topology
                delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
                reward += torch.sparse.mm(delta_util, coef).flatten()

        elif self.obj == 'min_max_link_util':

            # softened max: spread credit over the top-k most congested links
            # instead of the single argmax. With one link only, the gradient
            # reaches a handful of demands and the bottleneck jumps between
            # steps, making the signal extremely noisy; weighting the top-k
            # links by a softmax over their utilization keeps the objective
            # aligned with MLU while giving many more demands a useful
            # gradient direction.
            k = min(self.num_reward_edge, util.numel())
            top_util, top_edge = torch.topk(util, k)
            weight = torch.softmax(top_util/self.reward_temperature, dim=0)

            # per-path coefficient: sum over the top-k links it traverses,
            # each scaled by its weight and inverse capacity
            edge_coef = torch.zeros(self.num_edge_node).to(self.device)
            edge_coef[top_edge] = weight/self.obs[:-self.num_path_node][
                top_edge]
            max_util_paths = torch_scatter.scatter(
                edge_coef[self.p2e[1]], self.p2e[0],
                dim_size=self.num_path_node)

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                delta_path_flow = self.transform_raw_action(sample) - path_flow
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                reward += torch.sparse.mm(
                    delta_path_flow, max_util_paths.reshape(-1, 1)).flatten()

        return reward/num_sample

    def _read_graph_json(self, topo):
        """Return network topo from json file."""

        assert topo.endswith(".json")
        with open(os.path.join(TOPOLOGIES_DIR, topo)) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)

    def path_full_fname(self, topo, num_path, edge_disjoint, dist_metric):
        """Return full name of the topology path."""

        # pruned path dict only contains demand pairs with nonzero demand;
        # demand_split builds per-slice path dicts, so include the range
        pruned_str = "_pruned-{}".format(self.num_demand) \
            if self.prune_demands else ""
        if self.pair_range is not None:
            pruned_str += "_slice-{}-{}".format(*self.pair_range)
        return os.path.join(
            TOPOLOGIES_DIR, "paths", "path-form",
            "{}-{}-paths_edge-disjoint-{}_dist-metric-{}{}-dict.pkl".format(
                topo, num_path, edge_disjoint, dist_metric, pruned_str))

    def get_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary."""

        self.path_fname = self.path_full_fname(
            topo, num_path, edge_disjoint, dist_metric)
        print("Loading paths from pickle file", self.path_fname)
        try:
            with open(self.path_fname, 'rb') as f:
                path_dict = pickle.load(f)
                print("path_dict size:", len(path_dict))
                return path_dict
        except FileNotFoundError:
            print("Creating paths {}".format(self.path_fname))
            path_dict = self.compute_path(
                topo, num_path, edge_disjoint, dist_metric)
            print("Saving paths to pickle file")
            with open(self.path_fname, "wb") as w:
                pickle.dump(path_dict, w)
        return path_dict

    def compute_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary through computation."""

        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, dist_metric)
        for s_k, t_k in self.demand_pairs:
            paths = find_paths(G, s_k, t_k, num_path, edge_disjoint)
            paths_no_cycles = [remove_cycles(path) for path in paths]
            path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def get_regular_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary with the same number of paths per demand.
        Fill with the first path when number of paths is not enough.
        """

        path_dict = self.get_path(topo, num_path, edge_disjoint, dist_metric)
        for (s_k, t_k) in path_dict:
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [
                    path_dict[(s_k, t_k)][0] for _
                    in range(self.num_path - len(path_dict[(s_k, t_k)]))]\
                    + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    def get_topo_matrix(self, topo, num_path, edge_disjoint, dist_metric):
        """
        Return matrices related to topology.
        edge_index, edge_index_values: index and value for matrix
        D^(-0.5)*(adjacent)*D^(-0.5) without self-loop
        p2e: [path_node_idx, edge_nodes_inx]
        """

        # get regular path dict
        path_dict = self.get_regular_path(
            topo, num_path, edge_disjoint, dist_metric)

        # edge nodes' degree, index lookup
        edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
        node2degree_dict = {}
        edge_num = len(self.G.edges)

        # build edge_index
        src, dst, path_i = [], [], 0
        for s, t in self.demand_pairs:
            for path in path_dict[(s, t)]:
                for (u, v) in zip(path[:-1], path[1:]):
                    src.append(edge_num+path_i)
                    dst.append(edge2idx_dict[(u, v)])

                    if src[-1] not in node2degree_dict:
                        node2degree_dict[src[-1]] = 0
                    node2degree_dict[src[-1]] += 1
                    if dst[-1] not in node2degree_dict:
                        node2degree_dict[dst[-1]] = 0
                    node2degree_dict[dst[-1]] += 1
                path_i += 1

        # edge_index is D^(-0.5)*(adj)*D^(-0.5) without self-loop
        edge_index_values = torch.tensor(
            [1/math.sqrt(node2degree_dict[u]*node2degree_dict[v])
                for u, v in zip(src+dst, dst+src)]).to(self.device)
        edge_index = torch.tensor(
            [src+dst, dst+src], dtype=torch.long).to(self.device)
        p2e = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        p2e[0] -= len(self.G.edges)

        return edge_index, edge_index_values, p2e

    def extract_sol_mat(self, action):
        """return sparse solution matrix.
        Solution matrix is of dimension num_of_demand x num_of_edge.
        The i, j entry represents the traffic flow from demand i on edge j.
        """

        # 3D sparse matrix to represent which path, which demand, which edge
        sol_mat_index = torch.stack([
            self.p2e[0] % self.num_path,
            torch.div(self.p2e[0], self.num_path, rounding_mode='floor'),
            self.p2e[1]])

        # merge allocation from different paths of the same demand
        sol_mat = torch.sparse_coo_tensor(
            sol_mat_index,
            action[self.p2e[0]],
            (self.num_path,
                self.num_path_node//self.num_path,
                self.num_edge_node))
        sol_mat = torch.sparse.sum(sol_mat, [0])

        return sol_mat
