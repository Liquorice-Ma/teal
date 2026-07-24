import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import simulator
import torch
import os

def boxplot_data(args):
    datasets = ['starlink_gdp', 'starlink_pop']
    Y = []
    for dataset in datasets:
        args.dataset = dataset
        # load original tm
        tm = simulator.util.load_tm(args)
        Y.append(tm.reshape(-1))
    # plot
    fig, ax = plt.subplots()
    ax.boxplot(Y, showfliers=False)
    ax.set_xticklabels(datasets)
    # save
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)
