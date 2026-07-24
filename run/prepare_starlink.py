#! /usr/bin/env python
"""Convert satellite topology and traffic matrices in ./code to Teal format.

Topology: code/data/topo/{vertex,edge}_starlink_{X}_{Y}.csv
    -> topologies/Starlink{X}{Y}.json (networkx node-link json)
Traffic:  code/data/traffic_matrix/starlink_{X}_{Y}.npz
    -> traffic-matrices/starlink/
       Starlink{X}{Y}.json_starlink_{i}_1.0_traffic-matrix.pkl

Note: topology name contains no underscore, since tm fname is parsed by
splitting on underscores in teal_helper.py.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

sys.path.append('..')

from lib.config import TOPOLOGIES_DIR, TM_DIR


def convert(size_x, size_y, dataset, capacity, code_dir):
    topo_name = 'Starlink{}{}'.format(size_x, size_y)

    # ========== convert topology csv to node-link json
    G = nx.DiGraph()
    vertex_fname = os.path.join(
        code_dir, 'data', 'topo',
        'vertex_{}_{}_{}.csv'.format(dataset, size_x, size_y))
    for _, row in pd.read_csv(vertex_fname).iterrows():
        G.add_node(int(row['i']), x=float(row['x']), y=float(row['y']))
    edge_fname = os.path.join(
        code_dir, 'data', 'topo',
        'edge_{}_{}_{}.csv'.format(dataset, size_x, size_y))
    for _, row in pd.read_csv(edge_fname).iterrows():
        # capacity in csv is normalized to 1 for the MLL objective;
        # reset to an absolute value comparable with traffic demands
        G.add_edge(int(row['u']), int(row['v']), capacity=float(capacity))

    topo_fname = os.path.join(TOPOLOGIES_DIR, topo_name + '.json')
    with open(topo_fname, 'w') as f:
        json.dump(json_graph.node_link_data(G), f)
    print('Saved topology {} ({} nodes, {} edges)'.format(
        topo_fname, G.number_of_nodes(), G.number_of_edges()))

    # ========== convert npz traffic matrices to per-step pickles
    tm_fname = os.path.join(
        code_dir, 'data', 'traffic_matrix',
        '{}_{}_{}.npz'.format(dataset, size_x, size_y))
    tms = np.load(tm_fname)['tm'].astype(np.float32)

    tm_dir = os.path.join(TM_DIR, 'starlink')
    if not os.path.exists(tm_dir):
        os.makedirs(tm_dir)
    for i in range(tms.shape[0]):
        out_fname = os.path.join(
            tm_dir, '{}.json_starlink_{}_1.0_traffic-matrix.pkl'.format(
                topo_name, i))
        with open(out_fname, 'wb') as f:
            pickle.dump(tms[i], f)
    print('Saved {} traffic matrices to {} '
          '(total demand mean {:.0f}, nonzero ratio {:.4f})'.format(
              tms.shape[0], tm_dir,
              tms.sum(axis=(1, 2)).mean(),
              np.count_nonzero(tms)/tms.size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size-x', type=int, default=22, help='number of orbits')
    parser.add_argument(
        '--size-y', type=int, default=72, help='satellites per orbit')
    parser.add_argument(
        '--dataset', type=str, default='starlink',
        help='dataset name prefix in code/data')
    parser.add_argument(
        '--capacity', type=float, default=5000.0,
        help='absolute ISL capacity to replace the normalized value')
    parser.add_argument(
        '--code-dir', type=str, default='../code',
        help='directory of the satellite code folder')
    args = parser.parse_args()

    convert(args.size_x, args.size_y, args.dataset,
            args.capacity, args.code_dir)
