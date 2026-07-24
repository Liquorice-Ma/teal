import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import torch
import time
import os

from torch.autograd import Variable
from pprint import pprint
from . import util

class MultiPathShortestPathSolver:

    def __init__(self, args):
        # save args
        self.args = args
        # load topo
        self.load_topo()
        # initialize self.load_topo()
        # self.plot_topo(self.topo)

    def load_topo(self):
        # extract args
        args = self.args
        # initialize graph
        self.topo = nx.DiGraph()
        # load vertex
        path = os.path.join(args.topo_dir, f'vertex_{args.dataset}.csv')
        df = pd.read_csv(path)
        # add each vertex in the topo
        for i, row in df.iterrows():
            self.topo.add_node(int(row['i']), x=row['x'], y=row['y'])
        # load edge
        path = os.path.join(args.topo_dir, f'edge_{args.dataset}.csv')
        df = pd.read_csv(path)
        # add each edge in the topo
        for i, row in df.iterrows():
            self.topo.add_edge(int(row['u']), int(row['v']),
                                   capacity=row['capacity'],
                                   delay=row['delay'],
                                   p_error=row['p_error'])

    def plot_topo(self, G, label='topo'):
        # extract args
        args = self.args
        # get vertex positions
        pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
        # draw nodes
        nx.draw(G, pos, node_color='skyblue',
                        node_size=1000,
                        with_labels=True,
                        connectionstyle='arc3, rad=0.2')
        # save the plot
        path = os.path.join(args.figure_dir, f'{label}_{args.dataset}.pdf')
        plt.tight_layout()
        plt.savefig(path)
        plt.cla()
        plt.clf()

    def compute_one_shortest_path_list(self, i, j):
        # extract args
        args = self.args
        G    = self.topo
        D    = int(np.sqrt(G.number_of_nodes()))
        # debug
        sub_G = util.crop_graph(G, D, i, j)
        # self.plot_topo(sub_G, label=f'sub_topo_{i}_{j}')
        paths = []
        for path in util.a_star(sub_G, D, i, j, []):
            paths.append(path)
        # pprint(paths)
        return paths

    def compute_shortest_path_list(self):
        # extract args
        args = self.args
        G    = self.topo
        V    = G.number_of_nodes()
        D    = int(np.sqrt(V))
        spl = {} # shortest path list

        # compute the canonical shortest path from (0, 0) to any place
        i = 0
        for j in range(V):
            print(f'[+] compute_shortest_path_list for {i} -> {j}')
            spl[(i, j)] = self.compute_one_shortest_path_list(i, j)

        # precompute the map from canonical coordinate (x, y) -> node id i
        coord2node = {}
        for i, (x, y) in enumerate(itertools.product(range(D), range(D))):
            coord2node[(x, y)] = i

        # shift the canonical shortest path for non canonical one
        for i in range(1, V):
            x_i, y_i = G.nodes[i]['x'], G.nodes[i]['y']
            # precompute the map from i1 (non canonical frame of reference) -> i
            shifted_node = {}
            for idx, (x, y) in enumerate(itertools.product(range(D), range(D))):
                # compute location on shifted space
                x1 = (x + D - x_i) % D
                y1 = (y + D - y_i) % D
                # compute index on shifted space
                shifted_idx = coord2node[(x1, y1)]
                # shifted space -> original space
                shifted_node[shifted_idx] = idx

            for j in range(V):
                print(f'[+] shift the canonical path for {i} -> {j}')
                # check what j1 will look like when map i->i1=0
                x_j, y_j = G.nodes[j]['x'], G.nodes[j]['y']
                x_j1 = (x_j + D - x_i) % D
                y_j1 = (y_j + D - y_i) % D
                j1 = coord2node[(x_j1, y_j1)]
                # look up the canonical sp
                canonical_sp = spl[(0, j1)]
                shifted_sp = []
                for path in canonical_sp:
                    shifted_path = []
                    for node in path:
                        shifted_path.append(shifted_node[node])
                    shifted_sp.append(shifted_path)
                # store the shifted sp
                spl[(i, j)] = shifted_sp
                # pprint(shifted_sp)
        return spl

    def compute_optimal_split_ratio(self, tm):
        # extract args
        G   = self.topo
        V   = G.number_of_nodes()
        E   = G.number_of_edges()
        spl = self.compute_shortest_path_list()
        args = self.args

        # reshape tm
        tm = tm.reshape(-1).to(device='cuda:0')

        # compute PV2
        print('[+] compute PV2')
        tic = time.time()
        PV2 = 0
        for i, j in itertools.product(range(V), range(V)):
            PV2 += len(spl[(i, j)])
        print(f'    - done:{time.time() - tic:0.2f}')

        # building flow2path sparse matrix
        print('[+] building flow2path')
        indices = []
        f = 0
        p = 0
        for i, j in itertools.product(range(V), range(V)):
            for _ in range(len(spl[i, j])):
                indices.append([f, p])
                p += 1
            f += 1
        indices = torch.tensor(indices).T
        values  = torch.ones(indices.shape[1])
        flow2path = torch.sparse_coo_tensor(indices, values, (V ** 2, PV2, ),
                                            dtype=torch.float32)\
                         .to(device='cuda:0')
        print(f'    - done:{time.time() - tic:0.2f}')

        # precompute mapping froom u, v -> e
        edge2idx = {}
        for e, (u, v) in enumerate(G.edges):
            edge2idx[(u, v)] = e

        # building the path2edge matrix
        print('[+] building path2edge')
        indices = []
        p = 0
        for i, j in itertools.product(range(V), range(V)):
            for path in spl[(i, j)]:
                for u, v in zip(path[:-1], path[1:]):
                    e = edge2idx[(u, v)]
                    indices.append([p, e])
                p += 1
        indices   = torch.tensor(indices).T
        values    = torch.ones(indices.shape[1])
        path2edge = torch.sparse_coo_tensor(indices, values, (PV2, E, ),
                                            dtype=torch.float32)\
                         .to(device='cuda:0')
        print(f'    - done:{time.time() - tic:0.2f}')

        # define the objective function
        def objective(x):
            # initialize
            load = {}
            for u, v in G.edges:
                load[(u, v)] = 0
            # compute load
            util = (tm @ flow2path * x) @ path2edge
            # compute utilization
            # TODO: remove same capacity hardcode
            for u, v, data in G.edges(data=True):
                util = util / data['capacity']
                break
            # compute mlu
            mlu = torch.max(util)
            return mlu

        # define the contraints
        def constraint(x):
            constraint_values = []
            p = 0
            for i, j in itertools.product(range(V), range(V)):
                P = len(spl[(i, j)])
                constraint_values.append(torch.abs(torch.sum(x[p:p+P]) - 1))
                p += P
            constraint_values.append(torch.sum(torch.clamp(-x, min=0)))
            constraint_values = [_.view(1) if _.dim() == 0 else _ for _ in constraint_values]
            constraint_values = torch.cat(constraint_values)
            return constraint_values

        # create a variable from optimization
        x = Variable(torch.zeros(PV2, device='cuda:0'), requires_grad=True)
        with torch.no_grad():
            p = 0
            for i, j in itertools.product(range(V), range(V)):
                P = len(spl[(i, j)])
                x[p:p+P] = torch.full([P], 1 / P)
                p += P

        # define the optimizer
        optimizer = torch.optim.Adam([x], lr=args.learning_rate)

        # define the number of optimization steps
        num_steps = 1000
        initial_obj = None

        # optimization loop
        tic = time.time()
        for step in range(num_steps):
            optimizer.zero_grad()
            obj_value = objective(x)
            constraint_value = torch.sum(constraint(x) ** 2)
            loss = obj_value + constraint_value
            loss.backward()
            optimizer.step()
            toc = time.time()
            if initial_obj == None:
                initial_obj = obj_value.item()
            obj_value = obj_value.item() / initial_obj
            print(f'step={step} obj={obj_value} constraint={constraint_value.item()} time={toc - tic:0.2f}')
            if toc - tic > args.timeout:
                break



        # Print the optimized result
        print("Optimal objective value:", obj_value.item())

    def solve(self, tm):
        self.compute_optimal_split_ratio(tm)
