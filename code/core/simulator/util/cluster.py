import numpy as np

def get_cluster2edge(G, x_min, x_max, y_min, y_max):
    edges = []
    for u, v in G.edges():
        x_u, y_u, x_v, y_v = G.nodes[u]['x'], G.nodes[u]['y'], G.nodes[v]['x'], G.nodes[v]['y']
        if x_min <= x_u < x_max and y_min <= y_u < y_max and \
           x_min <= x_v < x_max and y_min <= y_v < y_max:
            edges.append((u, v))
    return edges

def get_edge_cluster_map(G, args):
    # initialize
    x_min, x_max, y_min, y_max = 0, args.cluster_size_x, 0, args.cluster_size_y
    #
    cluster_id = 0
    edge2cluster = {}
    cluster2edge = {}
    for x_min in range(0, args.size_x, args.cluster_size_x):
        for y_min in range(0, args.size_y, args.cluster_size_y):
            #
            x_max, y_max = x_min + args.cluster_size_x, y_min + args.cluster_size_y
            # args
            # n_edge = 0
            cluster2edge[cluster_id] = get_cluster2edge(G, x_min, x_max, y_min, y_max)
            for u, v in cluster2edge[cluster_id]:
                edge2cluster[(u, v)] = cluster_id
                # n_edge += 1
            # print(f'[+] cluster={cluster_id} {x_min} {x_max} {y_min} {y_max} n_edge={n_edge}')
            cluster_id += 1
    # intercluster
    # n_edge = 0
    cluster2edge[cluster_id] = []
    for u, v in G.edges():
        if (u, v) not in edge2cluster:
            edge2cluster[(u, v)] = cluster_id
            cluster2edge[cluster_id].append((u, v))
            if len(cluster2edge[cluster_id]) > len(cluster2edge[0]):
                cluster_id += 1
                cluster2edge[cluster_id] = []
            # n_edge += 1
    # print(f'[+] cluster={cluster_id} n_edge={n_edge}')
    C = cluster_id + 1 # number of cluster
    return cluster2edge, edge2cluster, C

def get_path_cluster_map(flow_index2flow, elephant_indices, clusters, global_pls, args):
    # map paths -> clusters
    p = 0
    C = len(clusters)
    path2cls = {}
    for f in elephant_indices:
        i, j = flow_index2flow[int(f.item())]
        for path in global_pls[(i, j)]:
            path2cls[p] = []
            for u, v in zip(path[:-1], path[1:]):
                for c, cluster in enumerate(clusters):
                    if (u, v) in cluster.edges:
                        if c not in path2cls[p]:
                            path2cls[p].append(c)
            p += 1

    # map cluster -> paths
    cluster2pls = {}
    for c in range(C):
        cluster2pls[c] = []
    for p in path2cls.keys():
        if len(path2cls[p]) > 0:
            c = np.random.choice(path2cls[p])
            cluster2pls[c].append(p)
    return cluster2pls

def get_flow2cluster(pl, edge2cluster):
    cluster_list = []
    for path in pl:
        for u, v in zip(path[:-1], path[1:]):
            c = edge2cluster[(u, v)]
            cluster_list.append(c)
    cluster_set = list(sorted(set(cluster_list)))
    return cluster_set
