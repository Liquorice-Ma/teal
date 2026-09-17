#!/usr/bin/env python
"""归档并统计 Phase C/Phase D；不改写任一原始实验 CSV。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
from statistics import mean, stdev

import numpy as np


PHASE_C_METHODS = ('ours', 'test-style', 'zero-fill', 'mean-interp', 'nbr-fill', 'untrained')
PHASE_C_RHOS = ('0.02', '0.05', '0.1', '0.3', '0.5')
NODE_METHODS = ('ours', 'zero-fill', 'untrained')
NODE_RHOS = ('0.1', '0.3', '0.5')
NODE_SEEDS = ('0', '1', '2')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_rows(path, required=('config', 'rho', 'seed', 'mlu')):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not rows[0]:
        raise ValueError(f'CSV 无数据列: {path}')
    absent = set(required) - set(rows[0])
    if absent:
        raise ValueError(f'{path} 缺少列: {sorted(absent)}')
    return rows


def index_rows(rows, label):
    indexed = {}
    for row in rows:
        key = (row['config'], row['rho'], row['seed'])
        if key in indexed:
            raise ValueError(f'{label} 存在重复条件: {key}')
        value = float(row['mlu'])
        if not math.isfinite(value):
            raise ValueError(f'{label} 有非有限 MLU: {key}')
        indexed[key] = value
    return indexed


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=mean(values) if values else None,
                std=stdev(values) if len(values) > 1 else 0.0 if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def sign_p_two_sided(wins, n):
    if not n:
        return None
    extreme = max(wins, n - wins)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def phase_c_report(original, retry=None):
    indexed = index_rows(read_rows(original), 'Phase C 原始 CSV')
    retry_index = {}
    if retry and Path(retry).is_file():
        retry_index = index_rows(read_rows(retry), 'Phase C 重试 CSV')
        overlap = sorted(set(indexed) & set(retry_index))
        if overlap:
            raise ValueError(f'重试结果与原始结果重叠，拒绝选择其一: {overlap}')
        indexed.update(retry_index)
    cells, missing = [], []
    for config in PHASE_C_METHODS:
        for rho in PHASE_C_RHOS:
            values = {seed: indexed[(config, rho, seed)] for seed in NODE_SEEDS + ('3', '4')
                      if (config, rho, seed) in indexed}
            cells.append(dict(config=config, rho=float(rho), seeds=sorted(values), **stats(values.values())))
            for seed in ('0', '1', '2', '3', '4'):
                if (config, rho, seed) not in indexed:
                    missing.append(dict(config=config, rho=float(rho), seed=seed))
    pairwise = []
    for rho in PHASE_C_RHOS:
        ours = {seed: indexed[('ours', rho, seed)] for seed in ('0', '1', '2', '3', '4')
                if ('ours', rho, seed) in indexed}
        for baseline in PHASE_C_METHODS:
            if baseline == 'ours':
                continue
            base = {seed: indexed[(baseline, rho, seed)] for seed in ('0', '1', '2', '3', '4')
                    if (baseline, rho, seed) in indexed}
            shared = sorted(set(ours) & set(base), key=int)
            wins = sum(ours[seed] < base[seed] for seed in shared)
            pairwise.append(dict(rho=float(rho), method='ours', baseline=baseline,
                                 paired_n=len(shared), wins=wins, losses=len(shared) - wins,
                                 mean_delta_ours_minus_baseline=mean([ours[s] - base[s] for s in shared]) if shared else None,
                                 sign_test_two_sided=sign_p_two_sided(wins, len(shared))))
    return dict(source_rows=len(indexed), retry_rows=len(retry_index), cells=cells,
                missing=missing, paired=pairwise), indexed


def starlink_problems(repo):
    # 仅接受当前仓库内受版本控制的数据；pickle 不能用于外部输入。
    traffic_dir = (Path(repo) / 'traffic-matrices' / 'starlink').resolve()
    files = sorted(traffic_dir.glob('Starlink2272.json*_traffic-matrix.pkl'),
                   key=lambda path: int(path.stem.split('_')[-3]))
    if any(not path.resolve().is_relative_to(traffic_dir) for path in files):
        raise ValueError('拒绝加载星座流量目录以外的序列化数据')
    if len(files) < 101:
        raise ValueError(f'预期至少101个 Starlink TM，实际 {len(files)}')
    return files[:101]


def coverage(repo, rho, obs_type, obs_seed=0, test_start=90, test_stop=101):
    """重建旧协议固定掩码，并以测试切片真值计算实际配对/体积覆盖率。"""
    files = starlink_problems(repo)
    matrices = []
    union = None
    for path in files:
        with path.open('rb') as stream:
            matrix = np.asarray(pickle.load(stream), dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f'无效 TM 形状: {path}')
        matrices.append(matrix)
        union = matrix > 0 if union is None else union | (matrix > 0)
    nodes = union.shape[0]
    np.fill_diagonal(union, False)
    source, target = np.nonzero(union)
    if obs_type == 'node':
        import torch
        generator = torch.Generator().manual_seed(obs_seed)
        count = max(1, int(round(nodes * rho)))
        observed_sources = torch.randperm(nodes, generator=generator)[:count].numpy()
        observed = np.isin(source, observed_sources)
    elif obs_type == 'flow':
        import torch
        generator = torch.Generator().manual_seed(obs_seed)
        count = max(1, int(round(len(source) * rho)))
        observed_idx = torch.randperm(len(source), generator=generator)[:count].numpy()
        observed = np.zeros(len(source), dtype=bool)
        observed[observed_idx] = True
    else:
        raise ValueError(f'未知观测粒度: {obs_type}')
    total_volume = observed_volume = 0.0
    for matrix in matrices[test_start:test_stop]:
        demand = matrix[source, target]
        total_volume += float(demand.sum())
        observed_volume += float(demand[observed].sum())
    return dict(obs_type=obs_type, rho=rho, obs_seed=obs_seed, nodes=nodes,
                demand_pairs=int(len(source)), observed_pairs=int(observed.sum()),
                observed_pair_fraction=float(observed.mean()), total_test_volume=total_volume,
                observed_test_volume=observed_volume,
                observed_volume_fraction=observed_volume / total_volume if total_volume else None,
                test_range=[test_start, test_stop])


def node_flow_report(flow, node, repo, include_coverage=True):
    flow_index = index_rows(read_rows(flow), 'flow-level CSV')
    node_index = index_rows(read_rows(node), 'node-level CSV')
    pairs, missing = [], []
    for config in NODE_METHODS:
        for rho in NODE_RHOS:
            shared, node_values, flow_values = [], [], []
            for seed in NODE_SEEDS:
                key = (config, rho, seed)
                if key not in node_index or key not in flow_index:
                    missing.append(dict(config=config, rho=float(rho), seed=seed,
                                        node_present=key in node_index, flow_present=key in flow_index))
                    continue
                shared.append(seed)
                node_values.append(node_index[key])
                flow_values.append(flow_index[key])
            deltas = [a - b for a, b in zip(node_values, flow_values)]
            pairs.append(dict(config=config, rho=float(rho), paired_n=len(shared), shared_seeds=shared,
                              node=stats(node_values), flow=stats(flow_values), delta_node_minus_flow=stats(deltas),
                              node_lower_wins=sum(a < b for a, b in zip(node_values, flow_values))))
    coverage_rows = ([coverage(repo, float(rho), kind) for rho in NODE_RHOS
                      for kind in ('flow', 'node')] if include_coverage else [])
    return dict(pairs=pairs, missing=missing, coverage=coverage_rows)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--phase-c', type=Path, required=True)
    parser.add_argument('--phase-c-retry', type=Path)
    parser.add_argument('--nodeobs', type=Path, required=True)
    parser.add_argument('--flow', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'输出目录已存在，拒绝覆盖: {args.output}')
    args.output.mkdir(parents=True)
    phase_c, merged = phase_c_report(args.phase_c, args.phase_c_retry)
    node_flow = node_flow_report(args.flow, args.nodeobs, args.repo)
    provenance = dict(phase_c_sha256=digest(args.phase_c), nodeobs_sha256=digest(args.nodeobs),
                      flow_sha256=digest(args.flow), retry_sha256=digest(args.phase_c_retry)
                      if args.phase_c_retry and args.phase_c_retry.exists() else None)
    report = dict(schema_version=1, provenance=provenance, phase_c=phase_c, node_flow=node_flow,
                  note='所有配对均按相同 config、rho、seed 对齐；node/flow 仅比较同一部署协议的原始结果。')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    write_csv(args.output / 'phase_c_cells.csv', phase_c['cells'])
    write_csv(args.output / 'phase_c_missing.csv', phase_c['missing'])
    write_csv(args.output / 'phase_c_paired.csv', phase_c['paired'])
    write_csv(args.output / 'node_flow_pairs.csv', node_flow['pairs'])
    write_csv(args.output / 'observation_coverage.csv', node_flow['coverage'])
    print(json.dumps(dict(phase_c_missing=len(phase_c['missing']), node_flow_missing=len(node_flow['missing']),
                          output=str(args.output)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
