from joblib import Parallel, delayed
from simulator import util
import itertools
import pickle
import time
import os

def prepare_mice_path_list_v2(args):
    # load topology
    tic = time.time()
    print(f'[+] loading {args.dataset} topology')
    G = util.load_topo(args)
    V = G.number_of_nodes()
    E = G.number_of_edges()
    toc = time.time()
    print(f'    - completed, t={toc - tic:0.2f}')
    print(f'    - V={V} E={E}')

    # generate mice flow path from node i with coordinate 0, 0
    tic = time.time()
    print(f'[+] generating canonical path list of mice flow')
    i = 0
    def job_fn(j):
        pl = util.get_path_list_mice_flow(G, i, j, args)
        if j % 10 == 0:
            toc = time.time()
            print(f'    + flow (0, {j}) completed, time remain {(toc - tic) / (j + 1) * (V - j):0.2f}s')
        return i, j, pl
    # job_fn(350)
    results = Parallel(n_jobs=os.cpu_count())(delayed(job_fn)(j) for j in range(V))
    canonical_pls = {}
    for i, j, pl in results:
        canonical_pls[(i, j)] = pl
    toc = time.time()
    print(f'    - completed, t={toc - tic:0.2f}')

    # save the canonical path list of mice flow
    path = os.path.join(args.routing_policy_dir, f'mice_{args.size_x}_{args.size_y}.pkl')
    print(f'[+] saving the canonical path list to {path}')
    tic = time.time()
    with open(path, 'wb') as fp:
        pickle.dump(canonical_pls, fp, protocol=pickle.HIGHEST_PROTOCOL)
    toc = time.time()
    print(f'    - completed, t={toc - tic:0.2f}')
