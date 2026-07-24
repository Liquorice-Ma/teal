import pandas as pd
import simulator
import torch
import os

def main(args):
    with torch.no_grad():
        # csv data
        csv_data = {
            'i'      : [],
            'mll'    : [],
            'time'   : [],
            'success': [],
        }
#         path = os.path.join(args.csv_dir, f'{args.dataset}_{args.solver}_{args.size_x}_{args.size_y}.csv')
#         if os.path.exists(path):
#             df = pd.read_csv(path)
#             csv_data = df.to_dict()
#             for key in csv_data:
#                 csv_data[key] = list(csv_data[key].values())
#         start_index = len(csv_data['i']) + 1
        start_index = 0
#         print(f'[+] restart from matrix {start_index}')
        # initialize solver
        solver = simulator.create_solver(args)
        # prepare the solver
        solver.prepare()
        # load traffic matrix
        TM = simulator.util.load_tm_v2(args)
        # solve many tm
        for i in range(start_index, TM.shape[0]):
            tm = TM[i, ...]
            tm = tm.reshape(-1)
            # normalize tm
            tm_max = tm.max()
            tm = tm / tm_max
            # solve
            mll, t, success = solver.solve(tm)
            print(i, mll * tm_max / 1000, t, success)
            csv_data['i'].append(i)
            csv_data['mll'].append(mll * tm_max / 1000)
            csv_data['time'].append(t)
            csv_data['success'].append(success)
            # save csv every time
            df = pd.DataFrame(csv_data)
            path = os.path.join(args.csv_dir, f'{args.dataset}_{args.solver}_{args.size_x}_{args.size_y}.csv')
            df.to_csv(path, index=None)
            # break
