from joblib import Parallel, delayed
from simulator import util
import itertools
import torch
import os

def get_mice_path_list(G, mice_indices, args):
    # extract args
    V       = G.number_of_nodes()
    F       = V ** 2
    #
    print(f'[+] generating path list for {len(mice_indices)} mice flows among {F} flows')
    # generate path list for every flow
    def fn(f, i, j):
        if f in mice_indices:
            return f, i, j, util.get_path_list_mice_flow(G, i, j, args)
        else:
            return f, i, j, []
    results = Parallel(n_jobs=os.cpu_count())(delayed(fn)(f, i, j) \
                    for f, (i, j) in enumerate(itertools.product(range(V), range(V))))
    mice_pls = {}
    for f, i, j, pl in results:
        if f in mice_indices:
            mice_pls[i, j] = pl
    return mice_pls

def get_edge_load_mice(G, tm, mice_indices, args):
    # extract args
    V       = G.number_of_nodes()
    F       = V ** 2
    # get path list for mice flow
    mice_pls = get_mice_path_list(G, mice_indices, args)
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
    return edge_load_mice
