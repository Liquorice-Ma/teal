#!/usr/bin/env python
"""Check that the mask embedding actually reaches the model input.

The embed/zero ablation produced bit-identical results, so this verifies,
on a real env, that (a) the mask marks some demands as unobserved, and
(b) switching mask_mode changes the tensor fed to the GNN.
"""

import sys

import torch

sys.path.append('..')

from teal_helper import get_args_and_problems, PATH_FORM_HYPERPARAMS  # noqa
from lib.teal_env import TealEnv                                      # noqa
from lib.teal_actor import TealActor                                  # noqa

args, _, problems = get_args_and_problems('diag-{}-{}.csv')
num_path, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS
num_path = args.num_path
edge_disjoint = not args.shared_paths

env = TealEnv(
    obj=args.obj, topo=args.topo, problems=problems,
    num_path=num_path, edge_disjoint=edge_disjoint,
    dist_metric=dist_metric, rho=args.rho,
    train_size=[args.slice_train_start, args.slice_train_stop],
    val_size=[args.slice_val_start, args.slice_val_stop],
    test_size=[args.slice_test_start, args.slice_test_stop],
    num_failure=0, device=torch.device('cpu'),
    obs_ratio=args.obs_ratio, hist_len=args.hist_len,
    prune_demands=args.prune_demands)
env.reset('test')
obs = env.get_obs()

print('num_demand      :', env.num_demand)
print('num_path_node   :', env.num_path_node)
print('mask shape      :', tuple(obs['mask'].shape))
print('mask observed   : %d / %d (%.1f%%)'
      % (obs['mask'].sum().item(), obs['mask'].numel(),
         100*obs['mask'].mean().item()))
print('tm_seq shape    :', tuple(obs['tm_seq'].shape))
print('tm_seq[-1] len  :', obs['tm_seq'][-1].numel())

for mode in ['embed', 'zero']:
    actor = TealActor(
        teal_env=env, num_layer=6, model_dir='./models',
        model_save=False, device=torch.device('cpu'),
        mask_mode=mode)
    tm = obs['tm_seq'][-1]
    path_mask = obs['mask'].repeat_interleave(actor.num_path)
    masked = tm * path_mask
    if mode == 'embed':
        out = masked + actor.mask_embedding.repeat(
            env.num_path_node//actor.num_path) * (1 - path_mask)
    else:
        out = masked
    print('%-6s: embedding[:3]=%s  input sum=%.4f  nonzero=%d'
          % (mode,
             [round(v, 4) for v in actor.mask_embedding[:3].tolist()],
             out.sum().item(), (out != 0).sum().item()))
