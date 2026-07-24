import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import simulator
import os

def plot_flow_cdf(args):
    # load tm
    tm = simulator.util.load_tm(args).detach().cpu().numpy()

    # crop for subproblem
    tm = tm[:40, :40]

    # plot cdf
    tm = tm.reshape(-1)
    print('mean', np.mean(tm))
    print('max', np.max(tm))
    print('min', np.min(tm))
    print('std', np.std(tm))
    print('var', np.var(tm))

    # identify outlier flow
    idx = np.where(tm > np.mean(tm) + 3 * np.std(tm))[0]
    print('threshold', np.mean(tm) + 3 * np.std(tm))
    print(idx)
    print(tm[idx])
    print(len(idx))
    print(len(tm))
    print(len(idx) / len(tm))

    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)
