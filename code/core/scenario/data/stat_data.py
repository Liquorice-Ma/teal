import pandas as pd
import numpy as np
import simulator
import torch
import os

def stat_data(args):
    # load original tm
    tm = simulator.util.load_tm(args)
    tm = tm.reshape(-1)
    tm = tm[tm != 0]
    threshold = np.mean(tm) + 3 * np.std(tm)
    n_non_zero = len(tm)
    n_elephant = len(np.where(tm > threshold)[0])
    print('min', float(np.min(tm)))
    print('max', float(np.max(tm)))
    print('mean', float(np.mean(tm)))
    print('std', float(np.std(tm)))
    print('var', float(np.var(tm)))
    print('percent non zero', len(tm) / (101 * 1584 * 1584) * 100)
    print(n_elephant, n_non_zero)
    print('percent elephant', n_elephant / n_non_zero * 100)
