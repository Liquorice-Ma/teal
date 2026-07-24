from joblib import Parallel, delayed
import itertools
import torch
import os

from simulator import util

def get_elephant_flow_path_list(G, elephant_indices, args):
    # extract args
    V       = G.number_of_nodes()
    F       = V ** 2
    # generate path list for every flow
    def fn(f, i, j):
        if f in elephant_indices:
            return i, j, True, util.get_path_list_elephant_flow(G, i, j, args)
        else:
            return i, j, False, []
    results = Parallel(n_jobs=os.cpu_count())(delayed(fn)(f, i, j) \
                    for f, (i, j) in enumerate(itertools.product(range(V), range(V))))
    pls = {}
    for i, j, is_elephant, pl in results:
        if is_elephant:
            pls[i, j] = pl
    return pls

def get_elephant_x0(G, pls, tm, P, flow2path, path2edge, edge_load_mice, args, E=None):
    # extract args
    V = G.number_of_nodes()
    if E is None:
        E = G.number_of_edges()
    # pass
    x = torch.zeros(P + 1 + E)
    # compute initial x_ij^p similar to ECMP
    q = 0
    for (i, j) in pls.keys():
        Q = len(pls[(i, j)])
        x[q:q+Q] = torch.full([Q], 1 / Q)
        q += Q
    # compute theta (mlu)
    # print(f'{flow2path.shape} {tm.shape} {P} {path2edge.shape} {edge_load_mice.shape}')
    # print((flow2path.T @ tm) * x[:P])
    edge_load     = (flow2path.T @ tm) * x[:P] @ path2edge + edge_load_mice
    max_edge_load = torch.max(edge_load)
    x[P] = max_edge_load
    # compute slack variables
    x[P+1:] = max_edge_load - edge_load
    return x, edge_load

def get_elephant_x0_v2(flows, G, pls, tm, P, flow2path, path2edge, edge_load_mice, args, max_ratio, E=None):
    # extract args
    V = G.number_of_nodes()
    if E is None:
        E = G.number_of_edges()
    # pass
    x = torch.zeros(P + 1 + E)
    # compute initial x_ij^p similar to ECMP
    q = 0

    for f, (i, j) in enumerate(flows):
        Q = len(pls[(i, j)])
        x[q:q+Q] = torch.full([Q], max_ratio[f] / Q)
        q += Q
    # compute theta (mlu)
    # print(f'{flow2path.shape} {tm.shape} {P} {path2edge.shape} {edge_load_mice.shape}')
    # print((flow2path.T @ tm) * x[:P])
    edge_load     = (flow2path.T @ tm) * x[:P] @ path2edge + edge_load_mice
    max_edge_load = torch.max(edge_load)
    x[P] = max_edge_load
    # compute slack variables
    x[P+1:] = max_edge_load - edge_load
    return x, edge_load
    # ah, missing the traffic from other cluster
