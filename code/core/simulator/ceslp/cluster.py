from simulator import util
import torch
import time

class Cluster:

    def __init__(self, cluster_id, edges, solver):
        self.cluster_id = cluster_id
        self.edges = edges
        self.args = solver.args
        self.solver = solver

    def set_path_indices(self, path_indices):
        self.path_indices = path_indices

    def prepare(self):
        # extract args
        path_indices = self.path_indices
        solver       = self.solver
        global_pls   = solver.pls
        edges        = self.edges
        G            = solver.topo
        args         = self.args
        V = G.number_of_nodes()
        E = len(edges)
        # background traffic of this cluster only
        edge_indices = []
        for e, (u, v) in enumerate(G.edges()):
            if (u, v) in edges:
                edge_indices.append(e)
        self.edge_indices = edge_indices
        # compute the reduced path list
        # filtered by the self.path_indices
        flow_indices  = []
        flows         = []
        flow2index    = {}
        index2flow    = {}
        p = 0
        pls = {}
        for f, (i, j) in enumerate(global_pls):
            flow2index[(i, j)] = f
            index2flow[f] = (i, j)
            for path in global_pls[(i, j)]:
                if p in path_indices:
                    if (i, j) not in flows:
                        flows.append((i, j))
                    if f not in flow_indices:
                        flow_indices.append(f)
                    if (i, j) not in pls:
                        pls[(i, j)] = []
                    pls[(i, j)].append(path)
                p += 1
        self.flow_indices = list(set(flow_indices))
        self.flows        = list(set(flows))

        # extract final matrix form
        P = util.get_n_path(G, pls)
        F = len(self.flow_indices)
        flow2path = util.get_flow2path_matrix_v2(G, P, pls, global_pls, args).to_dense()
        path2edge = util.get_path2edge_matrix_v2(G, P, pls, edges, args).to_dense().T
        # store attributes
        self.P = P
        self.F = F
        # print(f'[+] cluster {self.cluster_id} with {P} paths and {F} flows')
        self.E = E
        self.flow2path      = flow2path
        self.path2edge      = path2edge
        self.pls            = pls
        self.index2flow     = index2flow

        # extract sum ratio
        max_ratio = []
        for f in self.flow_indices:
            (i, j) = index2flow[f]
            max_ratio.append(len(pls[(i, j)]) / len(global_pls[(i, j)]))
            # print(i, j, max_ratio[-1])
        self.max_ratio = torch.tensor(max_ratio)

    def check_constraint(self, x):
        p = 0
        for f1, f in enumerate(self.flow_indices):
            (i, j) = self.index2flow[f]
            Q = len(self.pls[(i, j)])
            y = torch.sum(x[p:p+Q])
            if torch.abs(y - self.max_ratio[f1]) > 1e-3:
                print('cluster constraint failed', i, j, y)
            else:
                print('cluster constraint', f, i, j, x[p:p+Q], self.path_indices[p:p+Q], 'satisfied')
            p += Q

    def solve(self, x, edge_load):
        # extract args
        solver = self.solver
        G      = solver.topo
        #
        self.prepare()
        # extract the data from prepare
        # need the following data in prepare
        F = self.F
        E = self.E
        P = self.P
        tm = solver.tm[self.flow_indices]
        flow2path = self.flow2path
        path2edge = self.path2edge
        pls       = self.pls
        args      = self.args
        max_ratio = self.max_ratio
        flows     = self.flows

        # extract initial solution of cluster from global solution
        x0_ij_p = x[self.path_indices].clone().detach()
        edge_load_cluster = (flow2path.T @ tm) * x0_ij_p @ path2edge.T
        edge_load = edge_load[self.edge_indices]
        x0 = torch.cat([x0_ij_p, torch.tensor([edge_load.max()]), edge_load.max() - edge_load])

        # compute background traffic
        edge_load_mice = edge_load - edge_load_cluster
        self.edge_load_mice = edge_load_mice

        # compute fun init
        fun_init = edge_load.max()
        fun_init = (edge_load_cluster + edge_load_mice).max()

        # build A
        A1 = torch.cat([flow2path, torch.zeros(F, 1+E)], dim=1)
        A2 = torch.cat([path2edge * (tm @ flow2path), - torch.ones(E, 1), torch.eye(E)], dim=1)
        A = torch.cat([A1, A2], dim=0)
        # build B
        b = torch.cat([max_ratio, - edge_load_mice])
        # build C
        c = torch.cat([torch.zeros(P), torch.ones(1), torch.zeros(E)])
        # solve using linprog torch
        result = util.linprog(A, b, c, x0=x0, timeout=args.cluster_timeout)
        x_opt = result.x
        fun   = (c @ x_opt).item()
        print(f'    + cluster {self.cluster_id} before={fun_init:0.4f}')
        print(f'    + cluster {self.cluster_id} after={fun:0.4f}')
        return x_opt
