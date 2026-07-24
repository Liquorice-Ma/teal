import numpy as np
import pickle
import os

def prepare_synthetic_tm(args):
    # small1010.json_small_0_1.0_traffic-matrix.pkl
    TM = []
    for i in range(37):
        try:
            path = os.path.join(args.raw_dir, args.traffic_type,
                                f'small{args.size_x}{args.size_y}.json_small_{i}_1.0_traffic-matrix.pkl')
            # tm = np.load(path)
            with open(path, 'rb') as fp:
                tm = pickle.load(fp)
            TM.append(tm)
        except:
            pass
    TM = np.array(TM)
    print(TM.shape)
    path = os.path.join(args.traffic_matrix_dir,
                        f'starlink_{args.traffic_type}.npz')
    np.savez_compressed(path, tm=TM)
