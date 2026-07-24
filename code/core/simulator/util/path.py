import numpy as np
import torch
import copy

def compute_neighbor_distance(G, i, j, args):
    neighbors = []
    distances = []
    x_i, y_i = G.nodes[i]['x'], G.nodes[i]['y']
    x_j, y_j = G.nodes[j]['x'], G.nodes[j]['y']
    for k in G.neighbors(i):
        neighbors.append(k)
        x_k, y_k = G.nodes[k]['x'], G.nodes[k]['y']
        d = min(np.abs(x_k - x_j), np.abs(x_k + args.size_x - x_j)) + \
            min(np.abs(y_k - y_j), np.abs(y_k + args.size_y - y_j))
        distances.append(d)
    neighbors = np.array(neighbors)
    distances = np.array(distances)
    return neighbors, distances

def choose_random_nearest_neighbor(neighbors, distances, path):
    '''
    choose a random neighbor k of source i
    such that the distance from k to target j is minimized
    and k not in visited path
    '''
    indices = np.where(distances == np.min(distances))
    nearest_neighbors = neighbors[indices]
    k = np.random.choice(nearest_neighbors)
    count = 0
    while k in path:
        k = np.random.choice(nearest_neighbors)
        count += 1
        if count > 100:
            print('[+] error: get_path_list_elephant_flow loop forever')
            exit()
    return k

def get_path_list_mice_flow(G, i, j, args):
    # initialize current node
    k = i      # current visiting node
    path = [i] # visited path
    # find until destination
    while k != j:
        neighbors, distances = compute_neighbor_distance(G, k, j, args)
        k = choose_random_nearest_neighbor(neighbors, distances, path)
        path.append(k)
    path_list = [path]
    return path_list

def get_min_hop_count(G, i, j):
    x_i, y_i = G.nodes[i]['x'], G.nodes[i]['y']
    x_j, y_j = G.nodes[j]['x'], G.nodes[j]['y']
    min_hop_count = np.abs(x_i - x_j) + np.abs(y_i - y_j)
    return min_hop_count

def random_permutation_nearest_neighbor_list(neighbors, distances, path):
    '''
    get a randomly permutated list of neighbor k of source i
    such that the distance from k to target j is minimized
    and k not in visited path
    '''
    indices = np.where(distances == np.min(distances))
    nearest_neighbors = neighbors[indices]
    np.random.shuffle(nearest_neighbors)
    return nearest_neighbors

def get_path_list_elephant_flow(G, i, j, args):
    min_hop_count = get_min_hop_count(G, i, j)
    max_path = np.min([min_hop_count, 30])
    paths = []
    for path in randomized_a_star(G, i, j, [], args):
        paths.append(path)
        if len(paths) > max_path:
            break
    return paths

def randomized_a_star(G, i, j, path, args):
    ####################
    # STOPPING CONDITION
    # visit i and yield
    # if i==j
    ####################
    path.append(i)
    done = False
    if i == j:
        yield path
        done = True
    if not done:
        neighbors, distances = compute_neighbor_distance(G, i, j, args)
        nearest_neighbors = random_permutation_nearest_neighbor_list(neighbors, distances, path)
        for k in nearest_neighbors:
            if k not in path:
                for sub_path in randomized_a_star(G, k, j, copy.deepcopy(path), args):
                    yield sub_path

def get_elephant_flow_indices(tm):
    threshold = 0 # np.mean(tm) + 3 * np.std(tm)
    indices = np.where(tm > threshold)[0]
    return indices

def get_elephant_flow_indices_torch(tm):
    threshold = 0 # torch.mean(tm) + 3 * torch.std(tm)
    indices = torch.where(tm > threshold)[0]
    return indices

def get_elephant_mice_flow_indices(tm):
    # extract args
    F = len(tm)
    # classify flow to mice and elephant flows
    elephant_indices = np.array(get_elephant_flow_indices(tm))
    mice_indices = np.setdiff1d(np.arange(F), elephant_indices)
    return elephant_indices, mice_indices
