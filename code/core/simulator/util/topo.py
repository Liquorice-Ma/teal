import networkx as nx
import pandas as pd
import os

def load_topo(args):
    # initialize graph
    G = nx.DiGraph()
    # load vertex
    path = os.path.join(args.topo_dir, f'vertex_{args.dataset}_{args.size_x}_{args.size_y}.csv')
    df = pd.read_csv(path)
    # add each vertex in the topo
    for i, row in df.iterrows():
        G.add_node(int(row['i']), x=row['x'], y=row['y'])
    # load edge
    path = os.path.join(args.topo_dir, f'edge_{args.dataset}_{args.size_x}_{args.size_y}.csv')
    df = pd.read_csv(path)
    # add each edge in the topo
    for i, row in df.iterrows():
        G.add_edge(int(row['u']), int(row['v']),
                   capacity=row['capacity'],
                   delay=row['delay'],
                   p_error=row['p_error'])
    return G

def plot_topo(G, args, label='topo'):
    # get vertex positions
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    # draw nodes
    nx.draw(G, pos, node_color='skyblue',
                    node_size=10,
                    width=0.5,
                    with_labels=True,
                    font_size=2,
                    connectionstyle='arc3, rad=0.2')
    # save the plot
    path = os.path.join(args.figure_dir, f'{label}_{args.dataset}.pdf')
    plt.tight_layout()
    plt.savefig(path)
    plt.cla()
    plt.clf()

def get_index2edge(G):
    index2edge = {}
    for e, (u, v) in enumerate(G.edges):
        index2edge[e] = (u, v)
    return index2edge
