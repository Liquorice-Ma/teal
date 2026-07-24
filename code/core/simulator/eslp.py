from joblib import Parallel, delayed
from .se import SplitEqual
from . import util

import numpy as np
import itertools
import torch
import time
import os

class ElephantSplitLinearProgram(SplitEqual):

    def __init__(self, args):
        super().__init__(args)

    def get_path_list(self, tm):
        # extract args
        G       = self.topo
        V       = G.number_of_nodes()
        F       = V ** 2
        args    = self.args
        indices = util.get_elephant_flow_indices(tm)
        #
        print(f'[+] found {len(indices)} elephant flows among {F} flows')
        # generate path list for every flow
        def fn(f, i, j):
            if f in indices:
                return i, j, True, util.get_path_list_elephant_flow(G, i, j, args)
            else:
                return i, j, False, util.get_path_list_mice_flow(G, i, j, args)
        results = Parallel(n_jobs=os.cpu_count())(delayed(fn)(f, i, j) \
                        for f, (i, j) in enumerate(itertools.product(range(V), range(V))))
        elephant_pls = {}
        mice_pls = {}
        for i, j, is_elephant, pl in results:
            if is_elephant:
                elephant_pls[i, j] = pl
            else:
                mice_pls[i, j] = pl
        # generate mice indices
        elephant_indices = np.array(indices)
        mice_indices = np.setdiff1d(np.arange(F), elephant_indices)
        return elephant_pls, mice_pls, elephant_indices, mice_indices

    def get_x0_elephant(self, pls, tm, P, flow2path, path2edge, edge_load_mice):
        # extract args
        G = self.topo
        V = G.number_of_nodes()
        E = G.number_of_edges()
        args = self.args
        # pass
        x = torch.zeros(P + 1 + E)
        # compute initial x_ij^p similar to ECMP
        q = 0
        f = 0
        for (i, j) in itertools.product(range(V), range(V)):
            if (i, j) in pls:
                Q = len(pls[(i, j)])
                x[q:q+Q] = torch.full([Q], 1 / Q * tm[f])
                q += Q
                f += 1
        # compute theta (mlu)
        edge_load     = (flow2path.T @ tm) * x[:P] @ path2edge.T + edge_load_mice
        max_edge_load = torch.max(edge_load)
        x[q] = max_edge_load
        q += 1
        # compute slack variables
        x[q:] = max_edge_load - edge_load
        return x

    def solve(self, tm):
        # extract args
        G = self.topo
        args = self.args
        V = G.number_of_nodes()
        E = G.number_of_edges()
        F = V ** 2
        # start time counter
        tic = time.time()
        # get path list for every flow
        elephant_pls, mice_pls, elephant_indices, mice_indices = self.get_path_list(tm)
        # ==========================
        # Compute background traffic
        # ==========================
        # get number of path
        P_mice = util.get_n_path(G, mice_pls)
        # get path2edge matrix
        path2edge_mice = util.get_path2edge_matrix(G, P_mice, mice_pls, args)
        # tm to device
        tm_mice = torch.tensor(tm[mice_indices])
        # compute background traffic from mice flow
        edge_load_mice = tm_mice @ path2edge_mice
        # clean unuse variable
        path2edge_mice = None
        tm_mice = None
        torch.cuda.empty_cache()

        # ======================
        # Optimize elephant flow
        # ======================
        # tm to device
        tm_elephant = torch.tensor(tm[elephant_indices])
        # get number of path
        P_elephant = util.get_n_path(G, elephant_pls)
        F_elephant = len(elephant_indices)
        # building A
        flow2path_elephant = util.get_flow2path_matrix(G, P_elephant, elephant_pls, args).to_dense()
        A1                 = torch.cat([flow2path_elephant, torch.zeros(F_elephant, 1 + E)], dim=1)
        path2edge_elephant = util.get_path2edge_matrix(G, P_elephant, elephant_pls, args).to_dense().T
        A2                 = torch.cat([path2edge_elephant * (tm_elephant @ flow2path_elephant), - torch.ones(E, 1), torch.eye(E)], dim=1)
        A = torch.cat([A1, A2], dim=0)
        # building B
        b = torch.cat([tm_elephant, - edge_load_mice])
        # building c
        c = torch.cat([torch.zeros(P_elephant), torch.ones(1), torch.zeros(E)])
        # building x0
        x0 = self.get_x0_elephant(elephant_pls, tm_elephant, P_elephant, flow2path_elephant, path2edge_elephant, edge_load_mice)
        # solve using linprog torch
        result = util.linprog(A, b, c, x0=x0, timeout=args.timeout)
        # end time counter
        toc = time.time()
        x_opt = result.x
        # re evaluate the edge load
        edge_load = (flow2path_elephant.T @ tm_elephant) * torch.tensor(x_opt[:P_elephant]) @ path2edge_elephant.T + edge_load_mice
        mlu = edge_load.max()
        allocated = torch.cuda.memory_allocated(args.device) / (1024 ** 3)
        cached = torch.cuda.memory_cached(args.device) / (1024 ** 3)
        print(f'[shortest path split elephant linear program ver 2] MLU={mlu:0.2f} time={toc - tic:0.1f} allocated={allocated} cached={cached}')
        return float(mlu.item()), toc - tic, result.success
