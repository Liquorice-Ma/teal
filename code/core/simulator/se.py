from joblib import Parallel, delayed
from pprint import pprint
from copy import deepcopy
from . import util

import itertools
import torch
import time
import os

class SplitEqual:

    def __init__(self, args):
        # save args
        self.args = args
        # save graph
        self.topo = util.load_topo(args)
        if args.plot:
            util.plot_topo(self.topo, args)

    def get_cache_path_list(self):
        '''
        not the final path list
        depend on the tm, path list change for flows
        when receive the tm, copy the cache_mice_pls to pls, then for each
        elephant flow, check if it in cache_elephant_pls or not,
        if not find the shifted path list, overwrite pls and cached it into cache_elephant_pls
        '''
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        # load the canonical path list
        elephant_canonical_pls = util.load_elephant_flow_canonical_pls(args)
        mice_canonical_pls     = util.load_mice_flow_canonical_pls(args)
        # prepare for shifting
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
        self.pls                        = deepcopy(self.cache_mice_pls)
        self.flow_index2flow            = flow_index2flow

    def get_x0(self):
        # extract args
        G    = self.topo
        V    = G.number_of_nodes()
        P    = self.P
        pls  = self.pls
        args = self.args
        #
        x0 = torch.zeros(P)
        p  = 0
        for i, j in itertools.product(range(V), range(V)):
            if (i, j) in pls:
                P = len(pls[(i, j)])
                x0[p:p+P] = torch.full([P], 1 / P)
                p += P
        return x0

    def prepare(self):
        # extract args
        args = self.args
        G    = self.topo

        print('[+] prepare the path list')
        tic = time.time()
        self.get_cache_path_list()
        toc = time.time()
        print(f'    - done, t={toc-tic:0.2f}s')

    def route(self, tm):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        # copy the pls from the cache mice pls
        # find elephant flows and update the pls
        elephant_indices = util.get_elephant_flow_indices_torch(tm)
        F = len(elephant_indices)
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
        # compute the equal split
        self.P = util.get_n_path(G, self.pls)
        self.x = self.get_x0()
        # store the elephant indices for reset self.pls
        self.elephant_indices = elephant_indices

    def reset(self):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        #
        for f in self.elephant_indices:
            i, j = self.flow_index2flow[int(f.item())]
            self.pls[(i, j)] = self.cache_mice_pls[(i, j)]

    def evaluate(self, tm):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        pls  = self.pls
        #
        # get number of path
        P = util.get_n_path(G, pls)
        # get flow2path matrix
        flow2path = util.get_flow2path_matrix(G, P, pls, args)
        # get path2edge matrix
        path2edge = util.get_path2edge_matrix(G, P, pls, args)
        # solution
        x = self.x
        # find mll
        mll = torch.max((tm @ flow2path * x) @ path2edge)
        return mll

    def solve(self, tm):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        tm = torch.tensor(tm).to(torch.float32)

        print(f'[+] routing')
        tic = time.time()
        self.route(tm)
        toc = time.time()
        print(f'    - done: t={toc - tic:0.2f}s')

        print(f'[+] evaluating')
        tic = time.time()
        mll = self.evaluate(tm)
        toc = time.time()
        print(f'    - done: t={toc - tic:0.2f}s')

        print(f'[+] reseting')
        tic = time.time()
        self.reset()
        toc = time.time()
        print(f'    - done: t={toc - tic:0.2f}s')

        # can further improve the speed by masking the tm to compute the background
        # link load from mice flow
        # then build a small matrix to compute the elephant flow load

        print(f'[split elephant equal] mll={mll.item():0.2f} time={toc-tic:0.1f}')
        return mll.item(), toc - tic, True
