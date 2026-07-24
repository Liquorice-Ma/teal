import matplotlib.pyplot as plt
import numpy as np
import simulator
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

def sum_data(args):
    # load data
    datasets = ['starlink_gdp', 'starlink_pop']
    for dataset in datasets:
        args.dataset = dataset
        # load original tm
        tm = simulator.util.load_tm(args)
        tm = np.sum(tm, axis=1)
        tm = np.sum(tm, axis=1)
        print(tm.shape)
        print(dataset, tm)
