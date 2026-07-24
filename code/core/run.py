import itertools
import os

size_xs = [5, 10]
size_ys = [5, 10]
solvers = ['eslp']
# size_xs = [5, 10, 15]
# size_ys = [5, 10, 15]
# solvers = ['se', 'sp', 'ls', 'eslp']

# cmds = [
#     'python3 main.py --scenario=prepare_synthetic_tm --traffic_type=synthetic',
#     'python3 main.py --scenario=crop_data --traffic_type=synthetic',
# ]
# for cmd in cmds:
#     for size_x, size_y in zip(size_xs, size_ys):
#         cmd_ = f'{cmd} --size_x={size_x} --size_y={size_y}'
#         print(cmd_)
#         os.system(cmd_)

cmds = [
    'python3 main.py --scenario=prepare_elephant_path_list --traffic_type=synthetic',
    'python3 main.py --scenario=prepare_mice_path_list --traffic_type=synthetic',
    'python3 main.py --scenario=prepare_mice_path_list_v2 --traffic_type=synthetic',
]

for size_x, size_y in zip(size_xs, size_ys):
    for cmd in cmds:
        cmd_ = f'{cmd} --size_x={size_x} --size_y={size_y}'
        print(cmd_)
        os.system(cmd_)
    for solver in solvers:
        cmd_ = f'python3 main.py --size_x={size_x} --size_y={size_y} --solver={solver}'
        print(cmd_)
        os.system(cmd_)
