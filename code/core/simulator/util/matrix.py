import itertools
import torch

def get_n_path(G, pls):
    # extract args
    V = G.number_of_nodes()
    # pass
    n_path = 0
    for i, j in itertools.product(range(V), range(V)):
        if i in G.nodes():
            if (i, j) in pls:
                n_path += len(pls[(i, j)])
    return n_path

def get_flow2path_matrix(G, n_path, pls, args):
    # extract args
    V = G.number_of_nodes()
    #
    indices = []
    f = 0
    p = 0
    debug = 0
    for i, j in itertools.product(range(V), range(V)):
        if i in G.nodes():
            if (i, j) in pls:
                for _ in range(len(pls[i, j])):
                    indices.append([f, p])
                    p += 1
                f += 1
    indices = torch.tensor(indices).T
    values  = torch.ones(indices.shape[1])
    flow2path = torch.sparse_coo_tensor(indices, values, (f, p, ),
                                        dtype=torch.float32)\
                     .to(device=args.device)
    return flow2path

def get_flow2path_matrix_v2(G, n_path, pls, global_pls, args):
    # extract args
    V = G.number_of_nodes()
    #
    indices = []
    f = 0
    p = 0
    debug = 0
    for i, j in global_pls.keys():
        if (i, j) in pls:
            for _ in range(len(pls[i, j])):
                indices.append([f, p])
                p += 1
            f += 1
    indices = torch.tensor(indices).T
    values  = torch.ones(indices.shape[1])
    flow2path = torch.sparse_coo_tensor(indices, values, (f, p, ),
                                        dtype=torch.float32)\
                     .to(device=args.device)
    return flow2path

def get_path2edge_matrix(G, n_path, pls, args, edges=None):
    # extract args
    V = G.number_of_nodes()
    E = G.number_of_edges()
    # precompute mapping from u, v -> e
    edge2idx = {}
    if edges is None:
        for e, (u, v) in enumerate(G.edges()):
            edge2idx[(u, v)] = e
    else:
        for e, (u, v) in enumerate(G.edges()):
            if (u, v) in edges:
                edge2idx[(u, v)] = e
    #
    indices = []
    p = 0
    for i, j in itertools.product(range(V), range(V)):
        if i in G.nodes():
            if (i, j) in pls:
                for path in pls[(i, j)]:
                    for u, v in zip(path[:-1], path[1:]):
                        if (u, v) in edge2idx:
                            e = edge2idx[(u, v)]
                            indices.append([p, e])
                    p += 1
    indices   = torch.tensor(indices).T
    values    = torch.ones(indices.shape[1])
    path2edge = torch.sparse_coo_tensor(indices, values, (n_path, E, ),
                                        dtype=torch.float32)\
                     .to(device=args.device)
    return path2edge

def get_path2edge_matrix_v2(G, n_path, pls, edges, args):
    # extract args
    V = G.number_of_nodes()
    E = len(edges)
    # precompute mapping from u, v -> e
    edge2idx = {}
    for e, (u, v) in enumerate(edges):
        edge2idx[(u, v)] = e
    #
    indices = []
    p = 0
    for i, j in pls:
        for path in pls[(i, j)]:
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in edge2idx:
                    e = edge2idx[(u, v)]
                    indices.append([p, e])
            p += 1
    indices   = torch.tensor(indices).T
    values    = torch.ones(indices.shape[1])
    path2edge = torch.sparse_coo_tensor(indices, values, (n_path, E, ),
                                        dtype=torch.float32)\
                     .to(device=args.device)
    return path2edge
