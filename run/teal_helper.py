from collections import defaultdict
from glob import iglob

import argparse
import os
import sys

sys.path.append("..")

from lib.config import TOPOLOGIES_DIR, TM_DIR

PROBLEM_NAMES = [
    'B4.json',
    'UsCarrier.json',
    'Kdl.json',
    'ASN2k.json',
    'Starlink2224.json',
    'Starlink2272.json',
]
TM_MODELS = [
    "real",
    "toy",
    "starlink",
]
SCALE_FACTORS = [1.0]
OBJ_STRS = ["total_flow", "min_max_link_util"]

PATH_FORM_HYPERPARAMS = (4, True, "min-hop")

PROBLEM_NAMES_AND_TM_MODELS = [
    (prob_name, tm_model) for prob_name in PROBLEM_NAMES
    for tm_model in TM_MODELS
]

PROBLEMS = []
GROUPED_BY_PROBLEMS = defaultdict(list)
HOLDOUT_PROBLEMS = []
GROUPED_BY_HOLDOUT_PROBLEMS = defaultdict(list)

for problem_name in PROBLEM_NAMES:
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join(TOPOLOGIES_DIR, "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join(TOPOLOGIES_DIR, problem_name)
    for model in TM_MODELS:
        for tm_fname in iglob(
            "{}/{}/{}*_traffic-matrix.pkl".format(TM_DIR, model, problem_name)
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]),\
                float(vals[3])
            GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            PROBLEMS.append((problem_name, topo_fname, tm_fname))
        for tm_fname in iglob(
            "{}/holdout/{}/{}*_traffic-matrix.pkl".format(
                TM_DIR, model, problem_name
            )
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]),\
                float(vals[3])
            GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model, scale_factor)]\
                .append(
                    (topo_fname, tm_fname)
            )
            HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))

GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
for key, vals in GROUPED_BY_PROBLEMS.items():
    GROUPED_BY_PROBLEMS[key] = sorted(
        vals, key=lambda x: int(x[-1].split('_')[-3]))

GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
    GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(
        vals, key=lambda x: int(x[-1].split('_')[-3]))


def get_problems(args):
    if (args.topo, args.tm_model, args.scale_factor) not in GROUPED_BY_PROBLEMS:
        raise Exception('Traffic matrices not found')
    problems = []
    for topo_fname, tm_fname in GROUPED_BY_PROBLEMS[
            (args.topo, args.tm_model, args.scale_factor)]:
        problems.append((args.topo, topo_fname, tm_fname))
    return problems


def get_args_and_problems(formatted_fname_template, additional_args=[]):
    parser = argparse.ArgumentParser()

    # Problems arguments
    parser.add_argument(
        "--dry-run", dest="dry_run", default=False, action="store_true",
        help="list problems to run")
    parser.add_argument(
        "--obj", type=str, default='total_flow', choices=OBJ_STRS,
        help="objective function")
    parser.add_argument(
        "--tm-model", type=str, default='real', choices=TM_MODELS,
        help="traffic matrix model")
    parser.add_argument(
        "--topo", type=str, required=True, choices=PROBLEM_NAMES,
        help="network topology")
    parser.add_argument(
        "--scale-factor", type=float, default=1.0, choices=SCALE_FACTORS,
        help="traffic matrix scale factor")
    parser.add_argument(
        '--devid', type=int, default=0,
        help='GPU device id')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed for reproducibility')
    parser.add_argument(
        '--model-save', type=bool, default=False,
        help='whether to save model')

    # env hyper-parameters
    parser.add_argument(
        '--slice-train-start', type=int, default=0,
        help="start index of training")
    parser.add_argument(
        '--slice-train-stop', type=int, default=20,
        help="end index of training")
    parser.add_argument(
        '--slice-val-start', type=int, default=20,
        help="start index of validation")
    parser.add_argument(
        '--slice-val-stop', type=int, default=28,
        help="end index of validation")
    parser.add_argument(
        '--slice-test-start', type=int, default=28,
        help="start index of testing")
    parser.add_argument(
        '--slice-test-stop', type=int, default=36,
        help="end index of testing")

    # sparse observation hyper-parameters
    parser.add_argument(
        '--obs-ratio', type=float, default=1.0,
        help='ratio of observed node pairs in sparse traffic matrices')
    parser.add_argument(
        '--obs-type', type=str, default='flow', choices=['flow', 'node'],
        help='sparse sampling granularity: flow-level samples demand '
             'pairs; node-level samples source nodes whose outgoing '
             'demands are all observed')
    parser.add_argument(
        '--mask-init', type=float, default=57.0,
        help='initial scale of the learnable mask embedding; default is '
             'the median nonzero demand of the Starlink trace. Must stay '
             'within the demand distribution: values near zero make the '
             'embed mode identical to zero-filling, while the arithmetic '
             'mean over-estimates on heavy-tailed traffic')
    parser.add_argument(
        '--deterministic', action='store_true',
        help='force deterministic GPU algorithms to remove run-to-run '
             'variance (slower, but needed when the method gap is smaller '
             'than the seed noise)')
    parser.add_argument(
        '--num-path', type=int, default=4,
        help='number of candidate paths per demand')
    parser.add_argument(
        '--shared-paths', action='store_true',
        help='use k-shortest paths that may share links instead of '
             'edge-disjoint ones. Edge-disjoint path sets leave nothing '
             'for the policy to decide (their paths never contend for the '
             'same link), so uniform splitting is already near-optimal')
    parser.add_argument(
        '--obs-sample', type=str, default='uniform',
        choices=['uniform', 'top'],
        help='flow-level sampling strategy: uniform random, or top '
             '(80%% largest flows by training mean + 20%% random, '
             'following TEST)')
    parser.add_argument(
        '--hist-len', type=int, default=1,
        help='number of historical traffic matrices as model input')
    parser.add_argument(
        '--prune-demands', dest='prune_demands', default=False,
        action='store_true',
        help='only keep node pairs with nonzero demand in any '
             'traffic matrix (for sparse satellite traffic)')
    parser.add_argument(
        '--mask-mode', type=str, default='embed',
        choices=['embed', 'nbr', 'zero', 'mean'],
        help='how to fill unobserved demands: a learnable placeholder '
             'shared by all of them (embed), a per-demand estimate from '
             'observed demands sharing the same source scaled by a '
             'learnable factor (nbr), zero filling, or mean interpolation '
             'over observed demands (two-stage complete-then-optimize '
             'baseline)')
    parser.add_argument(
        '--no-gate', dest='no_gate', default=False, action='store_true',
        help='disable mask-aware gating in FlowGNN (ablation baseline)')
    parser.add_argument(
        '--demand-split', dest='demand_split', default=False,
        action='store_true',
        help='build demand set from training-slice TMs only, and rebuild '
             'from test-slice TMs at test time with the same weights '
             '(zero-retraining generalization, requires --prune-demands)')
    parser.add_argument(
        '--test-topo', type=str, default=None,
        help='alternative topology json used at test time only, e.g. a '
             'perturbed constellation (topology-drift zero-retraining)')

    # actor hyper-parameters
    parser.add_argument(
        '--layers', type=int, default=6,
        help='number of flowGNN layers')
    parser.add_argument(
        '--rho', type=float, default=1.0,
        help='rho in ADMM')

    # training hyper-parameters
    parser.add_argument(
        '--lr', type=float, default=0.0001,
        help='learning rate')
    parser.add_argument(
        '--epochs', type=int, default=0,
        help='number of training epochs')
    parser.add_argument(
        '--bsz', type=int, default=20,
        help='batch size')
    parser.add_argument(
        '--samples', type=int, default=5,
        help='number of COMA samples')
    parser.add_argument(
        '--reward-edges', type=int, default=1,
        help='number of most-congested links credited in the MLU reward; '
             '1 is the original single-bottleneck reward, larger values '
             'soften the max and reduce gradient variance')
    parser.add_argument(
        '--reward-temperature', type=float, default=0.01,
        help='softmax temperature over top-k link utilizations when '
             '--reward-edges > 1')
    parser.add_argument(
        '--num-restart', type=int, default=1,
        help='number of candidate initializations screened on the '
             'validation slice before full training; a sizable fraction of '
             'random inits never escape a bad region on this problem')
    parser.add_argument(
        '--warmup-epochs', type=int, default=5,
        help='epochs per candidate during initialization screening')
    parser.add_argument(
        '--admm-steps', type=int, default=5,
        help='number of ADMM steps')
    parser.add_argument(
        '--early-stop', type=bool, default=False,
        help='whether to stop early')

    # testing hyper-parameters
    parser.add_argument(
        '--failures', type=int, default=0, help='number of edge failures')

    for add_arg in additional_args:
        name_or_flags, kwargs = add_arg[0], add_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    args = parser.parse_args()

    slice_str = "all"  # "slice_" + "_".join(str(i) for i in args.slices)
    formatted_fname_substr = formatted_fname_template.format(
        args.obj, slice_str)
    return args, formatted_fname_substr, get_problems(args)


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
