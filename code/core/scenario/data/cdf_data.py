import matplotlib.pyplot as plt
import numpy as np
import simulator
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

def cdf_data(args):
    # load data
    datasets = ['starlink_gdp', 'starlink_pop']
    for dataset in datasets:
        args.dataset = dataset
        # load original tm
        tm = simulator.util.load_tm(args)
        T, N, _ = tm.shape
        tm = tm.reshape(T, -1)
        # get top tm
        top_tm = []
        for t in range(T):
            idx = np.argsort(tm[t, :])[-N:]
            top_tm.append(tm[t, idx])
        top_tm = np.array(top_tm).reshape(-1)
        # plot CDF
        top_tm = np.sort(top_tm)
        print(top_tm)
        exit()
        cumulative = np.linspace(0, 1, len(top_tm))
        plt.plot(top_tm, cumulative)
    # decorate
    plt.xlabel('Data')
    plt.ylabel('Cumulative Probability')
    # save
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)
