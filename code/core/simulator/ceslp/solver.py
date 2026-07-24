from joblib import Parallel, delayed
from .cluster import Cluster
from simulator import util
from copy import deepcopy

import numpy as np
import itertools
import torch
import time
import os

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # save graph
        self.topo = util.load_topo(args)

    #######################################################
    # Caching the global path list and evaluate for
    # mice/elephant flows
    #######################################################

    def get_cache_path_list(self):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        # load the canonical path list
        elephant_canonical_pls = util.load_elephant_flow_canonical_pls(args)
        mice_canonical_pls     = util.load_mice_flow_canonical_pls(args)
        # prepare map from node 2d coordinate to id for shifting
        location2node = util.get_location2node(G)
        # shifting path list for every flow i, j
        cache_mice_pls, cache_elephant_pls = {}, {}
        canonical2shifted_node_ids = {}
        f = 0
        flow_index2flow = {}
        for i in range(V):
            canonical2shifted_node_id     = util.get_canonical2shifted_node_id(G, i, location2node, args)
            canonical2shifted_node_ids[i] = canonical2shifted_node_id
            for j in range(V):
                cache_mice_pls[(i, j)]    = util.get_shifted_pl(G, i, j,
                                                                canonical2shifted_node_id,
                                                                location2node,
                                                                mice_canonical_pls, args)
                flow_index2flow[f] = (i, j)
                f += 1
        # store data
        self.cache_mice_pls             = cache_mice_pls
        self.cache_elephant_pls         = cache_elephant_pls
        self.location2node              = location2node
        self.canonical2shifted_node_ids = canonical2shifted_node_ids
        self.elephant_canonical_pls     = elephant_canonical_pls
        # self.pls                        = deepcopy(self.cache_mice_pls)
        self.flow_index2flow            = flow_index2flow

    def get_mice_evaluation_matrix(self):
        # extract args
        args = self.args
        G    = self.topo
        pls  = self.cache_mice_pls
        # get P and path2edge
        P = util.get_n_path(G, pls)
        path2edge = util.get_path2edge_matrix(G, P, pls, args)
        # save data
        self.mice_P = P
        self.mice_path2edge = path2edge

    def compute_mice_edge_load(self, tm, elephant_indices):
        mice_tm = tm.clone().detach()
        mice_tm[elephant_indices] = 0
        edge_load = mice_tm @ self.mice_path2edge
        return edge_load

    def compute_edge_load(self, x):
        # extract args
        self.get_elephant_evaluation_matrix()
        P              = self.elephant_P
        flow2path      = self.elephant_flow2path
        path2edge      = self.elephant_path2edge
        edge_load_mice = self.edge_load_mice
        tm             = self.tm
        edge_load      = (flow2path.T @ tm) * x[:P] @ path2edge + edge_load_mice
        return edge_load

    def prepare(self):
        print('[+] loading cache path list')
        tic = time.time()
        self.get_cache_path_list()
        print(f'    - t={time.time() - tic:0.2f}s')
        print('[+] getting mice evaluation matrix')
        tic = time.time()
        self.get_mice_evaluation_matrix()
        print(f'    - t={time.time() - tic:0.2f}s')

    #######################################################
    # How to divide into cluster
    #######################################################

    def get_elephant_pls(self, elephant_indices):
        # extract args
        args = self.args
        G    = self.topo
        self.elephant_indices = elephant_indices
        #
        self.pls = {}
        for f in elephant_indices:
            i, j = self.flow_index2flow[int(f.item())]
            if (i, j) in self.cache_elephant_pls:
                # load from cache
                self.pls[(i, j)] = self.cache_elephant_pls[(i, j)]
            else:
                # find new path list by shifting
                pl = util.get_shifted_pl(G, i, j,
                                         self.canonical2shifted_node_ids[i],
                                         self.location2node,
                                         self.elephant_canonical_pls,
                                         args)
                self.cache_elephant_pls[(i, j)] = pl
                self.pls[(i, j)] = pl

    def get_elephant_evaluation_matrix(self):
        # extract args
        args = self.args
        G    = self.topo
        pls  = self.pls
        # get matrices
        P = util.get_n_path(G, pls)
        flow2path = util.get_flow2path_matrix(G, P, pls, args)
        path2edge = util.get_path2edge_matrix(G, P, pls, args)
        # save data
        self.elephant_P = P
        self.elephant_flow2path = flow2path
        self.elephant_path2edge = path2edge

    def get_clusters(self):
        # extract args
        args         = self.args
        G            = self.topo
        V            = G.number_of_nodes()
        pls          = self.pls

        # map clusters <-> edges
        cluster2edge, edge2cluster, C = util.get_edge_cluster_map(G, args)
        # initialize clusters
        clusters = []
        for c in range(C):
            edges   = cluster2edge[c]
            cluster = Cluster(c, edges, self)
            clusters.append(cluster)
        # map cluster -> path
        cluster2pls = util.get_path_cluster_map(self.flow_index2flow,
                                                self.elephant_indices,
                                                clusters, pls, args)
        # print(f'[+] total {self.elephant_P} paths')
        total = 0
        for c in range(C):
            # print(f'    - cluster {c} with {len(cluster2pls[c])} paths')
            total += len(cluster2pls[c])
            # print(cluster2pls[c])
            clusters[c].set_path_indices(cluster2pls[c])
        # print(f'    + total={total}')
        self.clusters = clusters

        # generate map from edge index to edge
        index2edge = util.get_index2edge(G)
        self.index2edge = index2edge
        self.edge2cluster   = edge2cluster

    #######################################################
    # Choose and update optimization result
    #######################################################

    def find_max_cluster(self, edge_load):
        # finding max edge
        e = torch.argmax(edge_load)
        u, v = self.index2edge[e.item()]
        c = self.edge2cluster[(u, v)]
        # print(f'    + max edge {e}:{u}->{v} max load {edge_load.max().item():0.2f}, from cluster {c}')
        return self.clusters[c]

    def merge_solution(self, x, cluster_x, cluster):
        P = len(cluster.path_indices)
        p = 0
        for f1, f in enumerate(cluster.flow_indices):
            (i, j) = cluster.index2flow[f]
            Q = len(cluster.pls[(i, j)])
            idx = cluster.path_indices[p:p+Q]
            x[idx] = cluster_x[p:p+Q]
            p += Q
        return x

    def check_constraint(self, x):
        pls = self.pls
        p = 0

        for f, (i, j) in enumerate(pls):
            Q = len(pls[(i, j)])
            y = torch.sum(x[p:p+Q])
            if torch.abs(y - 1) > 1e-3:
                print('[+] global constraint fail', f, i, j, y, x[p:p+Q])
            else:
                print('[+] global constraint ok')
            p += Q

    def solve(self, tm):
        best_mll = 9999
        for i in range(1):
            print(f'[+] solving round {i}/1')
            tic = time.time()
            initial_mll, mll, t, success = self.solve_one(tm)
            if mll < best_mll:
                best_mll = mll
            print(f'[+] initial_mll={initial_mll:0.4f} mll={mll:0.4f}, best_mll={best_mll:0.4f}, t={time.time() - tic:0.2f}s')
        return best_mll, t, success

    def solve_one(self, tm, patience=5):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        tm = torch.tensor(tm).to(torch.float32)
        # prepare
        # optimize
        tic = time.time()
        elephant_indices = util.get_elephant_flow_indices_torch(tm)
        self.get_elephant_pls(elephant_indices)
        self.get_elephant_evaluation_matrix()
        self.get_clusters()
        clusters = self.clusters
        # compute background traffic from mice flows
        edge_load_mice = self.compute_mice_edge_load(tm, elephant_indices)
        self.edge_load_mice = edge_load_mice
        #
        elephant_tm = torch.tensor(tm[elephant_indices])
        self.tm = elephant_tm # save for the cluster to read
        x, edge_load = util.get_elephant_x0(self.topo, self.pls, elephant_tm,
                                             self.elephant_P, self.elephant_flow2path,
                                             self.elephant_path2edge, edge_load_mice, args)
        #exit()
        initial_mll = edge_load.max().item()
        best_mll    = initial_mll
        n_no_improve = 0

        while 1:
            cluster = self.find_max_cluster(edge_load)
            x_c = cluster.solve(deepcopy(x), edge_load)
            x = self.merge_solution(deepcopy(x), deepcopy(x_c), cluster)
            # self.check_constraint(x)
            print(f'    - global_before={edge_load.max().item():0.4f}')
            edge_load = self.compute_edge_load(x)
            print(f'    - global_after={edge_load.max().item():0.4f}')
            cluster = self.find_max_cluster(edge_load)
            # print()
            # stopping condition
            mll = edge_load.max().item()
            if mll < best_mll:
                best_mll = mll
                n_no_improve = 0
            else:
                n_no_improve += 1
            if n_no_improve > patience:
                break
            toc = time.time()
            if toc - tic > args.timeout:
                break
        # extract result
        toc = time.time()
        mlu = edge_load.max()
        # print(f'[shortest path split elephant linear program clustering v2] MLL={initial_mll:0.4f} -> {best_mll:0.4f} time={toc-tic:0.1f}')
        return initial_mll, best_mll, toc - tic, True
