#!/usr/bin/env python
"""LP oracle: optimal MLU per test snapshot for PR normalization.

Solves the path-form MCF (min theta s.t. per-link load <= theta*c_e,
split ratios sum to 1) with scipy HiGHS on the ground-truth TM, i.e.,
the denominator U_opt of TEST's Performance Ratio PR = U / U_opt.
Reuses TealEnv for demand pairs, candidate paths and p2e mapping.

Example:
    python lp_oracle.py --topo Starlink2272.json --tm-model starlink \
        --prune-demands --slice-test-start 90 --slice-test-stop 101
Writes lp-oracle-{topo}.csv with per-snapshot optimal MLU.
"""

import sys

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, eye

from teal_helper import get_args_and_problems, print_, PATH_FORM_HYPERPARAMS

sys.path.append('..')

from lib.teal_env import TealEnv


def solve_mlu(env, demand):
    """Return optimal MLU for one snapshot via path-form LP.

    Variables: split ratios r (num_path_node) followed by theta.
    minimize theta
    s.t.  sum_{p on e} r_p * d_p <= theta * c_e     (per link)
          sum_{p in demand i} r_{i,p} = 1           (per demand)
          r >= 0
    """

    n_r = env.num_path_node
    p2e = env.p2e.cpu().numpy()
    cap = env.capacity.cpu().numpy()
    d_path = demand.cpu().numpy()                    # demand per path node

    # capacity rows: load(e) - theta*c_e <= 0
    a_cap = coo_matrix(
        (d_path[p2e[0]], (p2e[1], p2e[0])),
        shape=(env.num_edge_node, n_r))
    a_ub = hstack([a_cap, coo_matrix(
        (-cap, (range(env.num_edge_node), [0]*env.num_edge_node)),
        shape=(env.num_edge_node, 1))]).tocsc()
    b_ub = np.zeros(env.num_edge_node)

    # conservation rows: sum of ratios per demand == 1
    rows = np.arange(n_r) // env.num_path
    a_eq = hstack([
        coo_matrix((np.ones(n_r), (rows, np.arange(n_r))),
                   shape=(env.num_demand, n_r)),
        coo_matrix((env.num_demand, 1))]).tocsc()
    b_eq = np.ones(env.num_demand)

    c = np.zeros(n_r + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                  bounds=[(0, None)]*n_r + [(0, None)], method='highs')
    assert res.status == 0, res.message
    return res.x[-1]


if __name__ == '__main__':
    args, _, problems = get_args_and_problems('lp-oracle-{}-{}.csv')
    num_path, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
    num_path = args.num_path
    edge_disjoint = not args.shared_paths

    env = TealEnv(
        obj='min_max_link_util', topo=args.topo, problems=problems,
        num_path=num_path, edge_disjoint=edge_disjoint,
        dist_metric=dist_metric, rho=args.rho,
        train_size=[args.slice_train_start, args.slice_train_stop],
        val_size=[args.slice_val_start, args.slice_val_stop],
        test_size=[args.slice_test_start, args.slice_test_stop],
        num_failure=0, device=torch.device('cpu'),
        prune_demands=args.prune_demands)

    env.reset('test')
    out_fname = 'lp-oracle-{}.csv'.format(args.topo)
    with open(out_fname, 'w') as f:
        print_('snapshot,opt_mlu', file=f)
        for idx in range(env.idx_start, env.idx_stop):
            demand = env.obs[-env.num_path_node:]
            opt = solve_mlu(env, demand)
            print_('{},{:.6f}'.format(idx, opt), file=f)
            print_('snapshot {}: optimal MLU = {:.4f}'.format(idx, opt))
            env._next_obs()
    print_('saved ' + out_fname)
