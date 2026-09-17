"""固定路由评估与边流 LP；不依赖具体学习方法，也不更新模型权重。"""
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from types import MappingProxyType

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def validate_graph(graph):
    if not graph.is_directed() or set(graph.nodes) != set(range(len(graph))):
        raise ValueError('图必须有向且节点为连续整数')
    if any(not np.isfinite(data['capacity']) or data['capacity'] <= 0
           for _, _, data in graph.edges(data=True)):
        raise ValueError('容量必须为正；故障边须真正移除')


def validate_traffic(tm, nodes):
    demand = sparse.coo_matrix(tm)
    if demand.shape != (nodes, nodes) or np.any(~np.isfinite(demand.data)) or np.any(demand.data < 0):
        raise ValueError('TM 形状、有限性或非负性不满足要求')
    demand.sum_duplicates()
    return [(int(s), int(t), float(value)) for s, t, value in zip(demand.row, demand.col, demand.data)
            if s != t and value > 0]


def failure_graph(graph, ratio, seed):
    """按物理 ISL 同时移除双向边；不为维持连通而重抽故障样本。"""
    validate_graph(graph)
    if not 0 <= ratio <= 1:
        raise ValueError('故障比例必须在[0,1]')
    if any(not graph.has_edge(v, u) for u, v in graph.edges):
        raise ValueError('物理ISL要求双向边成对存在')
    links = sorted({tuple(sorted((u, v))) for u, v in graph.edges})
    indices = np.random.default_rng(seed).permutation(len(links))[:int(len(links) * ratio)]
    removed = tuple(links[i] for i in indices)
    result = graph.copy()
    for u, v in removed:
        result.remove_edge(u, v)
        result.remove_edge(v, u)
    return result, removed


@dataclass(frozen=True)
class FrozenPaths:
    routes: object
    input_cutoff: int

    @classmethod
    def create(cls, routes, input_cutoff):
        normalized = {}
        for (s, t), choices in routes.items():
            paths = tuple((tuple(path), float(weight)) for path, weight in choices)
            if not paths or any(len(path) < 2 or path[0] != s or path[-1] != t
                                or len(set(path)) != len(path) for path, _ in paths):
                raise ValueError('候选路径必须是端点正确的简单路径')
            weights = np.asarray([weight for _, weight in paths])
            if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-6, rtol=0):
                raise ValueError('路由比例非负且每个需求之和必须为1')
            normalized[(int(s), int(t))] = paths
        return cls(MappingProxyType(normalized), int(input_cutoff))

    def digest(self):
        serial = [(pair, choices) for pair, choices in sorted(self.routes.items())]
        return hashlib.sha256(json.dumps([self.input_cutoff, serial]).encode()).hexdigest()


def score_frozen(graph, tm, policy, target, num_paths=4):
    """将同一比例应用于新 TM；新增需求及失效路径不得静默丢弃。"""
    validate_graph(graph)
    if target <= policy.input_cutoff or num_paths < 1:
        raise ValueError('目标必须晚于输入截止时间，候选路径数必须为正')
    before = policy.digest()
    edge_load = {edge: 0.0 for edge in graph.edges}
    total = unreachable = fallback = unseen = 0.0
    for s, t, value in validate_traffic(tm, len(graph)):
        total += value
        choices = policy.routes.get((s, t))
        if choices is None:
            unseen += value
            try:
                paths = list(islice(nx.shortest_simple_paths(graph, s, t), num_paths))
                choices = [(path, 1 / len(paths)) for path in paths]
            except nx.NetworkXNoPath:
                choices = []
            fallback += value
        else:
            choices = [(path, weight) for path, weight in choices
                       if all(graph.has_edge(u, v) for u, v in zip(path[:-1], path[1:])) and weight > 0]
            if not choices:
                fallback += value
                try:
                    choices = [(nx.shortest_path(graph, s, t), 1.0)]
                except nx.NetworkXNoPath:
                    choices = []
        if not choices:
            unreachable += value
            continue
        weight_sum = sum(weight for _, weight in choices)
        for path, weight in choices:
            for edge in zip(path[:-1], path[1:]):
                edge_load[edge] += value * weight / weight_sum
    served_mlu = max((load / graph[u][v]['capacity'] for (u, v), load in edge_load.items()), default=0.0)
    if before != policy.digest():
        raise RuntimeError('评分改变了冻结路由')
    return dict(mlu=served_mlu if unreachable == 0 else None, served_mlu=served_mlu,
                unreachable_fraction=unreachable / total if total else 0.0,
                fallback_fraction=fallback / total if total else 0.0,
                unseen_fraction=unseen / total if total else 0.0,
                edge_load=edge_load, route_sha256=before)


def edge_flow_oracle(graph, tm, time_limit=120):
    """完整信息边流 LP：按目的地合并商品，容量约束共用同一个 MLU。"""
    validate_graph(graph)
    traffic = validate_traffic(tm, len(graph))
    if not traffic:
        return 0.0
    for source, target, _ in traffic:
        if not nx.has_path(graph, source, target):
            return None
    nodes, edges = len(graph), sorted(graph.edges)
    destinations = sorted({t for _, t, _ in traffic})
    dest_index = {t: i for i, t in enumerate(destinations)}
    count, width = len(destinations), len(edges)
    columns = np.repeat(np.arange(width), 2)
    rows = np.asarray(edges).ravel()
    incidence = sparse.coo_matrix((np.tile([1.0, -1.0], width), (rows, columns)), shape=(nodes, width)).tocsr()
    conservation = sparse.kron(sparse.eye(count, format='csr'), incidence, format='csr')
    rhs = np.zeros((count, nodes))
    for s, t, amount in traffic:
        rhs[dest_index[t], s] += amount
        rhs[dest_index[t], t] -= amount
    capacities = np.asarray([graph[u][v]['capacity'] for u, v in edges])
    shared_capacity = sparse.hstack([sparse.eye(width, format='csr')] * count, format='csr')
    inequality = sparse.hstack([shared_capacity, sparse.csr_matrix(-capacities[:, None])], format='csr')
    equality = sparse.hstack([conservation, sparse.csr_matrix((count * nodes, 1))], format='csr')
    objective = np.zeros(count * width + 1)
    objective[-1] = 1
    result = linprog(objective, A_ub=inequality, b_ub=np.zeros(width), A_eq=equality,
                     b_eq=rhs.ravel(), bounds=(0, None), method='highs',
                     options={'time_limit': time_limit, 'primal_feasibility_tolerance': 1e-8})
    if not result.success:
        raise RuntimeError(f'LP 未求得最优解，不能用作定标: {result.message}')
    return float(result.fun)
