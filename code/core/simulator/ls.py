from joblib import Parallel, delayed
from pprint import pprint
from . import util
import itertools
import torch
import copy
import time
import os

class LocalSearch:

    def __init__(self, args):
        # save args
        self.args = args
        # save graph
        self.topo = util.load_topo(args)

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
        flow2path = util.get_flow2path_matrix(G, P, pls, args)
        path2edge = util.get_path2edge_matrix(G, P, pls, args)
        edge2flow = util.get_edge2flow(G, pls, args)
        toc = time.time()
        print(f'    - done, t={toc-tic:0.2f}s')

        # save data for later use
        self.pls = pls
        self.P   = P
        self.flow2path = flow2path
        self.path2edge = path2edge
        self.edge2flow = edge2flow

    def solve(self, tm):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        tm = torch.tensor(tm).to(torch.float32)
        # optimization variable
        P         = self.P
        x         = torch.ones(P)
        # start time counter
        tic = time.time()
        step = 0
        # get path list for every flow
        print(f'[+] deep copy path list ...')
        pls = copy.deepcopy(self.pls)
        # get edge2flow
        print('[+] getting edge2flow')
        edge2flow = copy.deepcopy(self.edge2flow)
        # get edge2index, index2edge, flow2index
        edge2index, index2edge = util.get_edge2index(G)
        flow2index, index2flow = util.get_flow2index(G)
        # get edge2index, index2edge, flow2index
        print(f'[+] initial evaluation ...')
        # initial evaluation
        flow2path = self.flow2path
        path2edge = self.path2edge
        edge_load = (tm @ flow2path * x) @ path2edge
        toc = time.time()
        print(f'    - done initial eval in {toc-tic:0.2f}s')
        print(f'[+] routing ...')
        # start local search here
        while toc - tic < args.timeout:
            # select the most congested link
            e = int(torch.argmax(edge_load).item())
            # print(step, toc - tic, torch.max(edge_load))
            (u, v) = index2edge[e]
            # list all flow through the edge
            flow_indices = [flow2index[flow] for flow in edge2flow[(u, v)]]
            sub_tm = tm[flow_indices]
            # randomly select one of the top flow in the most congested link
            prob = sub_tm / torch.sum(sub_tm)
            idx = int(torch.multinomial(prob, 1).item())
            f = flow_indices[idx]
            (i, j) = index2flow[f]
            # subtract the current load of the flow
            # remove the current flow from the edge2flow
            path = pls[(i, j)][0]
            for u, v in zip(path[:-1], path[1:]):
                e = edge2index[(u, v)]
                edge_load[e] -= tm[f]
                edge2flow[(u, v)] = [_ for _ in edge2flow[(u, v)] if _ != (i, j)]
            # reroute the flow
            pls[(i, j)] = util.get_path_list_mice_flow(G, i, j, args)
            # add the load of the flow back
            # add the current flow back to the edge2flow
            path = pls[(i, j)][0]
            for u, v in zip(path[:-1], path[1:]):
                e = edge2index[(u, v)]
                edge_load[e] += tm[f]
                edge2flow[(u, v)].append((i, j))
            # update time
            toc = time.time()
            step += 1
        # end
        # end time counter
        toc = time.time()
        print(f'- done: t={toc - tic:0.2f}s')
        # compute mlu
        tm = tm.reshape(-1)
        print(f'[+] evaluating solution')
        tic = time.time()
        # flow2path = util.get_flow2path_matrix(G, P, pls, args)
        # path2edge = util.get_path2edge_matrix(G, P, pls, args)
        mlu = torch.max(edge_load)
        toc = time.time()
        print(f'- done: t={toc - tic:0.2f}s')
        print(f'[local search] MLU={mlu.item():0.2f} time={toc-tic:0.1f}')
        return mlu.item(), toc - tic, True
