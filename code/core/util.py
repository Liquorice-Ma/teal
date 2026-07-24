import argparse
import torch
import time
import os

base_folder = os.path.dirname(os.path.dirname(__file__))

def create_dirs(args):
    for d in [args.csv_dir,
              args.topo_dir,
              args.figure_dir,
              args.routing_policy_dir,
              args.traffic_matrix_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

def set_default_device(args):
    torch.set_default_device(args.device)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    # scenario
    parser.add_argument('--scenario', type=str, default='main')
    # solver
    parser.add_argument('--solver', type=str, default='ceslp')
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--cluster_timeout', type=float, default=10)
    # data
    parser.add_argument('--size_x', type=int, default=22) # 22
    parser.add_argument('--size_y', type=int, default=72) # 72
    parser.add_argument('--cluster_size_x', type=int, default=11) # 22
    parser.add_argument('--cluster_size_y', type=int, default=12) # 72
    # I/O
    parser.add_argument('--traffic_type', type=str, default=f'pop')
    parser.add_argument('--traffic_matrix_dir', type=str, default=f'{base_folder}/data/traffic_matrix')
    parser.add_argument('--routing_policy_dir', type=str, default=f'{base_folder}/data/routing_policy/')
    parser.add_argument('--raw_dir', type=str, default=f'{base_folder}/data/raw')
    parser.add_argument('--figure_dir', type=str, default=f'{base_folder}/data/figure/')
    parser.add_argument('--topo_dir', type=str, default=f'{base_folder}/data/topo')
    parser.add_argument('--csv_dir', type=str, default=f'{base_folder}/data/csv')
    parser.add_argument('--dataset', type=str, default='starlink_synthetic')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--csv', action='store_true')
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    # parse args
    args = parser.parse_args()
    # create dirs if not existed
    create_dirs(args)
    # set default device
    set_default_device(args)
    return args

def print_args(args):
    pass
