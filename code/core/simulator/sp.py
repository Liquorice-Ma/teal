from joblib import Parallel, delayed
from pprint import pprint
from . import util
import itertools
import torch
import time
import os

class ShortestPath:

    def __init__(self, args):
        # save args
        self.args = args
        # save graph
        self.topo = util.load_topo(args)
        if args.plot:
            util.plot_topo(self.topo, args)

    def get_path_list(self):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        # load the canonical path list
        canonical_pls = util.load_mice_flow_canonical_pls(args)
        # prepare for shifting
        location2node = util.get_location2node(G)
        # shifting path list for every flow i, j
        pls = {}
        for i in range(V):
            canonical2shifted_node_id = util.get_canonical2shifted_node_id(G, i, location2node, args)
            for j in range(V):
                pls[(i, j)] = util.get_shifted_pl(G, i, j,
                                                  canonical2shifted_node_id,
                                                  location2node,
                                                  canonical_pls, args)
        return pls

    def prepare(self):
        # extract args
        args = self.args
        G    = self.topo

        print('[+] prepare the path list')
        tic = time.time()
        pls = self.get_path_list()
        toc = time.time()
        print(f'    - done, t={toc-tic:0.2f}s')

        print('[+] prepare the evaluation matrices')
        # get number of path
        tic = time.time()
        P = util.get_n_path(G, pls)
        # get flow2path matrix
        flow2path = util.get_flow2path_matrix(G, P, pls, args)
        # get path2edge matrix
        path2edge = util.get_path2edge_matrix(G, P, pls, args)
        toc = time.time()
        print(f'    - done, t={toc-tic:0.2f}s')

        # save data for later use
        self.pls = pls
        self.P   = P
        self.flow2path = flow2path
        self.path2edge = path2edge

    def solve(self, tm):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        tm = torch.tensor(tm).to(torch.float32)
        # start time counter
        tic = time.time()
        # get path list for every flow
        print(f'[+] routing ...')
        pls = self.pls
        # end time counter
        toc = time.time()
        print(f'- done: t={toc - tic:0.2f}s')
        # optimization variable
        P         = self.P
        x         = torch.ones(P)
        flow2path = self.flow2path
        path2edge = self.path2edge
        # compute mlu
        tm = tm.reshape(-1)
        print(f'[+] evaluating solution')
        tic = time.time()
        mlu = torch.max((tm @ flow2path * x) @ path2edge)
        toc = time.time()
        print(f'- done: t={toc - tic:0.2f}s')
        print(f'[shortest path] MLU={mlu.item():0.2f} time={toc-tic:0.1f}')
        return mlu.item(), toc - tic, True
