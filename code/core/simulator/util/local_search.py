from simulator import util
import itertools

def get_edge2flow(G, pls, args):
    # initialize
    edge2flow = {}
    for u, v in G.edges():
        edge2flow[(u, v)] = []
    # for each flow associate them to edge
    for i, j in pls.keys():
        for path in pls[(i, j)]:
            for u, v in zip(path[:-1], path[1:]):
                edge2flow[(u, v)].append((i, j))
    return edge2flow

def get_edge2index(G):
    edge2index = {}
    index2edge = {}
    for i, (u, v) in enumerate(G.edges()):
        edge2index[(u, v)] = i
        index2edge[i] = (u, v)
    return edge2index, index2edge

def get_flow2index(G):
    V = G.number_of_nodes()
    flow2index = {}
    index2flow = {}
    for idx, (i, j) in enumerate(itertools.product(range(V), range(V))):
        flow2index[(i, j)] = idx
        index2flow[idx] = (i, j)
    return flow2index, index2flow
