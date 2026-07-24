import pandas as pd
import numpy as np
import simulator
import os

def crop_data(args):
    # set reference dataset name
    # new dataset name
    dataset = f'{args.dataset}_{args.size_x}_{args.size_y}'
    # load original tm
    tm = simulator.util.load_tm(args)
    # determine number of flows of the crop matrix
    F = args.size_x * args.size_y
    # crop
    tm = tm[:, :F, :F].astype(np.float32)
    # save
    print('[+] saving tm', tm.shape)
    tm = simulator.util.save_tm(tm, dataset, args)

    # generate cropped edge
    i = 0
    data = {
        'u': [],
        'v': [],
        'capacity': [],
        'delay'   : [],
        'p_error' : [],
    }
    for y in range(args.size_y):
        for x in range(args.size_x):
            u = x + y * args.size_x
            # up
            x1 = x
            y1 = (y - 1) % args.size_y
            v = x1 + y1 * args.size_x
            data['u'].append(u)
            data['v'].append(v)
            data['capacity'].append(1)
            data['delay'].append(1)
            data['p_error'].append(0.01)

            # down
            x1 = x
            y1 = (y + 1) % args.size_y
            v = x1 + y1 * args.size_x
            data['u'].append(u)
            data['v'].append(v)
            data['capacity'].append(1)
            data['delay'].append(1)
            data['p_error'].append(0.01)

            # left
            x1 = (x - 1) % args.size_x
            y1 = y
            v = x1 + y1 * args.size_x
            data['u'].append(u)
            data['v'].append(v)
            data['capacity'].append(1)
            data['delay'].append(1)
            data['p_error'].append(0.01)

            # right
            x1 = (x + 1) % args.size_x
            y1 = y
            v = x1 + y1 * args.size_x
            data['u'].append(u)
            data['v'].append(v)
            data['capacity'].append(1)
            data['delay'].append(1)
            data['p_error'].append(0.01)
    # save
    df = pd.DataFrame(data)
    path = os.path.join(args.topo_dir, f'edge_{dataset}.csv')
    df.to_csv(path, index=None)
    print('[+] saving edge')
    print(df)

    # generate cropped vertex
    i = 0
    data = {
        'i': [],
        'x': [],
        'y': [],
    }
    for y in range(args.size_y):
        for x in range(args.size_x):
            i = x + y * args.size_x
            data['i'].append(i)
            data['x'].append(x)
            data['y'].append(y)
    df = pd.DataFrame(data)
    path = os.path.join(args.topo_dir, f'vertex_{dataset}.csv')
    df.to_csv(path, index=None)
    print('[+] saving vertex')
    print(df)
