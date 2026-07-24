import itertools
import pickle
import os

def load_mice_flow_canonical_pls(args):
    path = os.path.join(args.routing_policy_dir, 'mice.pkl')
    with open(path, 'rb') as fp:
        canonical_pls = pickle.load(fp)
    return canonical_pls

def load_mice_flow_canonical_pls_v2(args):
    path = os.path.join(args.routing_policy_dir, f'mice_{args.size_x}_{args.size_y}.pkl')
    with open(path, 'rb') as fp:
        canonical_pls = pickle.load(fp)
    return canonical_pls

def load_elephant_flow_canonical_pls(args):
    path = os.path.join(args.routing_policy_dir, 'elephant.pkl')
    with open(path, 'rb') as fp:
        canonical_pls = pickle.load(fp)
    return canonical_pls

def get_location2node(G):
    location2node = {}
    for i in G.nodes():
        x, y = G.nodes[i]['x'], G.nodes[i]['y']
        location2node[(x, y)] = i
    return location2node

def get_canonical2shifted_node_id(G, i, location2node, args):
    # extract location of node i
    x_i, y_i = G.nodes[i]['x'], G.nodes[i]['y']
    # shift the space from node 0 to node i
    canonical2shifted_node_id = {}
    for idx, (y, x) in enumerate(itertools.product(range(args.size_y), range(args.size_x))):
        # compute location on shifted space
        x1 = (x + args.size_x - x_i) % args.size_x
        y1 = (y + args.size_y - y_i) % args.size_y
        # map back to location on canonical space
        canonical2shifted_node_id[location2node[(x1, y1)]] = idx
    return canonical2shifted_node_id

def get_shifted_pl(G, i, j, canonical2shifted_node_id, location2node, canonical_pls, args):
    # extract location of i and j
    x_i, y_i = G.nodes[i]['x'], G.nodes[i]['y']
    x_j, y_j = G.nodes[j]['x'], G.nodes[j]['y']
    # check what j1 will look like when map i->i1=0
    x_j1 = (x_j + args.size_x - x_i) % args.size_x
    y_j1 = (y_j + args.size_y - y_i) % args.size_y
    j1   = location2node[(x_j1, y_j1)]
    # look up the canonical_pls
    canonical_pl = canonical_pls[(0, j1)]
    pl           = []
    for canonical_path in canonical_pl:
        path = list(map(lambda node: canonical2shifted_node_id[node], canonical_path))
        pl.append(path)
    return pl
