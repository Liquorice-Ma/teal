import numpy as np
import os

def prepare_starlink_tm(args):
    TM = []
    for i in range(101):
        try:
            path = os.path.join(args.raw_dir, args.traffic_type, f'tmMatrix{i}.npy')
            tm = np.load(path)
            TM.append(tm)
        except:
            pass
    TM = np.array(TM)
    print(TM.shape)
    path = os.path.join(args.traffic_matrix_dir, f'starlink_{args.traffic_type}.npz')
    np.savez_compressed(path, tm=TM)
