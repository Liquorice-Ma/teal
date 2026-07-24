import numpy as np
import torch
import os

def load_tm(args):
    path = os.path.join(args.traffic_matrix_dir, f'{args.dataset}.npz')
    tm = np.load(path)['tm'].astype(np.float32)
    return tm

def load_tm_v2(args):
    path = os.path.join(args.traffic_matrix_dir,
                        f'{args.dataset}_{args.size_x}_{args.size_y}.npz')
    tm = np.load(path)['tm'].astype(np.float32)
    return tm

def save_tm(tm, dataset, args):
    path = os.path.join(args.traffic_matrix_dir, f'{dataset}.npz')
    np.savez_compressed(path, tm=tm)
