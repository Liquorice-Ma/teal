#!/usr/bin/env python
"""Compare edge-disjoint vs k-shortest candidate paths on the constellation.

Diagnosis for why learning has no effect: with 4 edge-disjoint paths per
demand the paths share no links, so uniform splitting is already close to
optimal and the policy has nothing to decide. This script quantifies, on
a sample of the active demand pairs:
  - link sharing within a demand's own path set (intra-demand contention)
  - how many demands compete for the same link (inter-demand contention)
  - path length inflation (edge-disjoint detours cost hops)

Usage (in run/):  python diagnose_paths.py [num_sample]
"""

import json
import random
import sys
from collections import Counter

import networkx as nx
from networkx.readwrite import json_graph

sys.path.append('..')

from lib.path_utils import find_paths, path_to_edge_list          # noqa: E402

NUM_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NUM_PATH = 4

with open('../topologies/Starlink2272.json') as f:
    G = json_graph.node_link_graph(json.load(f))
for u, v in G.edges():
    G[u][v]['weight'] = 1

nodes = list(G.nodes())
random.seed(0)
pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(NUM_SAMPLE)]
pairs = [(s, t) for s, t in pairs if s != t]

for disjoint, label in [(True, 'edge-disjoint (current)'),
                        (False, 'k-shortest (proposed)')]:
    shared_within = 0        # demands whose own paths share >=1 link
    hops = []
    edge_usage = Counter()   # link -> number of demands using it
    for s, t in pairs:
        paths = find_paths(G, s, t, NUM_PATH, disjoint=disjoint)
        if not paths:
            continue
        seen, dup = set(), False
        for p in paths:
            hops.append(len(p) - 1)
            for e in path_to_edge_list(p):
                e = tuple(sorted(e))
                if e in seen:
                    dup = True
                seen.add(e)
        shared_within += dup
        for e in seen:
            edge_usage[e] += 1

    n = len(pairs)
    print('=== %s ===' % label)
    print('  demands with intra-set link sharing : %d/%d (%.0f%%)'
          % (shared_within, n, 100*shared_within/n))
    print('  mean path length                    : %.2f hops'
          % (sum(hops)/len(hops)))
    print('  links touched                       : %d' % len(edge_usage))
    print('  max demands sharing one link        : %d'
          % max(edge_usage.values()))
    print('  mean demands per touched link       : %.2f\n'
          % (sum(edge_usage.values())/len(edge_usage)))
