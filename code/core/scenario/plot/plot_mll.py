import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def load(args):
    path = os.path.join(args.csv_dir, f'{args.dataset}_{args.solver}.csv')
    return pd.read_csv(path)['mll'].to_numpy()

def plot_mll(args):
    # load
    datasets = ['starlink_pop', 'starlink_gdp']
    for dataset in datasets:
        ub = None
        args.dataset = dataset
        solvers = ['sp', 'se', 'ls', 'ceslp']
        labels  = ['sp', 'se', 'ls', 'ceslp']
        data = []
        L = 9999
        for solver in solvers:
            args.solver = solver
            y = load(args)
            if ub is None:
                ub = np.max(y)
            y = y / ub
            data.append(y)
            if len(y) < L:
                L = len(y)


        data1 = []
        for y in data:
            data1.append(y[:L])

        data2 = []
        for y in data:
            data2.append(y / data[0].max())

        # plot
        fig, ax = plt.subplots()
        ax.boxplot(data1)
        ax.set_xticklabels(solvers)

        # save
        plt.tight_layout()
        path = os.path.join(args.figure_dir, f'{args.scenario}_{args.dataset}.pdf')
        plt.savefig(path)
