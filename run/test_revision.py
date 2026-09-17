"""补实验协议回归：纯数值单测及可选的真实 PyG/Teal CPU 集成测试。"""
import json
from pathlib import Path
import pickle
import sys

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))
from revision_data import (EARTH_KM, SHELLS, access_mapping, city_traffic,
                           satellite_traffic, segment_clearance, traffic_inventory)
from revision_protocol import check_checkpoint, history_indices, save_checkpoint, state_digest
from revision_routing import FrozenPaths, edge_flow_oracle, failure_graph, score_frozen


def diamond():
    graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)]).to_directed()
    for i, (u, v) in enumerate(sorted(graph.edges)):
        graph[u][v].update(capacity=1.0, edge_id=i)
    return graph


def test_history_no_future_and_legacy_padding():
    assert history_indices(0, 3, 1) == [None, None, None]
    assert history_indices(1, 3, 1) == [None, None, 0]
    assert history_indices(288, 3, 1, 288) == [285, 286, 287]
    assert history_indices(288, 3, 0, 288) == [288, 288, 288]
    for target in range(432):
        assert all(t is None or t < target for t in history_indices(target, 3, 1))


def test_shell_counts_and_access_geometry():
    assert [s.nodes for s in SHELLS] == [1584, 1156, 351]
    ground = np.array([[EARTH_KM, 0, 0], [-EARTH_KM, 0, 0]])
    satellites = np.array([[EARTH_KM + 550, 0, 0]])
    assert access_mapping(satellites, ground, 25).tolist() == [0, -1]
    assert segment_clearance(np.array([[7000., 0, 0]]), np.array([[-7000., 0, 0]]))[0] == -EARTH_KM


def test_two_traffic_models_and_aggregation():
    cities = [dict(population=p, latitude=i * 10, longitude=i * 20) for i, p in enumerate([10, 20, 30])]
    first = city_traffic(cities, seed=7, snapshots=5)
    repeat = city_traffic(cities, seed=7, snapshots=5)
    for model, trace in first.items():
        np.testing.assert_array_equal(trace, repeat[model])
        np.testing.assert_allclose(np.trace(trace, axis1=1, axis2=2), 0)
    np.testing.assert_allclose(first['population'].sum(axis=(1, 2)), first['gravity'].sum(axis=(1, 2)))
    assert not np.allclose(first['population'], first['gravity'])
    attachment = np.tile([0, 0, 1], (5, 1))
    matrix, local = satellite_traffic(first['population'], attachment, 2)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel() + local,
                               first['population'].sum(axis=(1, 2)))
    assert set(matrix.indices) == {1, 2}


def test_memory_bound_not_first_snapshot():
    matrix = sparse.csr_matrix(([1., 1.], ([0, 1], [1, 2])), shape=(432, 1584**2))
    report = traffic_inventory(matrix, 1584)
    assert report['full_pair_union'] == 2
    assert report['dote_fp32_adam_lower_bound_gib'] > 24


def test_frozen_ratio_scores_changed_volume_and_lp_bound():
    graph = diamond()
    policy = FrozenPaths.create({(0, 3): [([0, 1, 3], 0.5), ([0, 2, 3], 0.5)]}, 288)
    tm = np.zeros((4, 4))
    tm[0, 3] = 2
    before = policy.digest()
    for horizon in range(1, 10):
        result = score_frozen(graph, tm * horizon, policy, 288 + horizon)
        assert result['mlu'] == pytest.approx(horizon, abs=1e-6)
        assert result['route_sha256'] == before
    assert edge_flow_oracle(graph, tm) == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError):
        score_frozen(graph, tm, policy, 288)
    with pytest.raises(TypeError):
        policy.routes[(0, 3)] = ()


def test_failed_paths_fallback_and_unreachable():
    graph = diamond()
    policy = FrozenPaths.create({(0, 3): [([0, 1, 3], 1)]}, 0)
    tm = np.zeros((4, 4))
    tm[0, 3] = 2
    graph.remove_edge(0, 1)
    result = score_frozen(graph, tm, policy, 1)
    assert result['mlu'] == 2
    assert result['fallback_fraction'] == 1
    assert (0, 1) not in result['edge_load']
    graph.remove_edge(0, 2)
    result = score_frozen(graph, tm, policy, 1)
    assert result['mlu'] is None
    assert result['unreachable_fraction'] == 1
    assert edge_flow_oracle(graph, tm) is None


def test_unseen_demand_not_dropped_and_physical_failures():
    graph = diamond()
    tm = np.zeros((4, 4))
    tm[0, 3] = 2
    result = score_frozen(graph, tm, FrozenPaths.create({}, 0), 1)
    assert result['unseen_fraction'] == 1
    assert result['mlu'] == 1
    failed, removed = failure_graph(graph, 0.5, 42)
    assert len(removed) == 2 and len(failed.edges) == 4
    assert removed == failure_graph(graph, 0.5, 42)[1]
    for u, v in removed:
        assert not failed.has_edge(u, v) and not failed.has_edge(v, u)
    assert not nx.is_connected(failure_graph(graph, 1, 42)[0].to_undirected())


def test_checkpoint_config_integrity_and_exclusive_save(tmp_path):
    torch = pytest.importorskip('torch')
    path = tmp_path / 'seed0.pt'
    state = {'w': torch.tensor([1., 2.])}
    config = {'seed': 0, 'hist_len': 3}
    save_checkpoint(path, state, config, 1)
    payload = torch.load(path, weights_only=True)
    check_checkpoint(payload, config)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, state, config, 1)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / 'random.pt', state, config, 0)
    with pytest.raises(ValueError):
        check_checkpoint(payload, {'seed': 1, 'hist_len': 3})
    payload['state_dict']['w'][0] = 4
    with pytest.raises(ValueError, match='哈希'):
        check_checkpoint(payload, config)


@pytest.fixture
def real_env(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_scatter')
    pytest.importorskip('torch_sparse')
    import lib.teal_env as module
    monkeypatch.setattr(module, 'TOPOLOGIES_DIR', str(tmp_path))
    (tmp_path / 'paths/path-form').mkdir(parents=True)
    path = tmp_path / 'Tiny.json'
    path.write_text(json.dumps(nx.node_link_data(diamond())))
    problems = []
    for t in range(8):
        tm = np.full((4, 4), float(t + 1))
        np.fill_diagonal(tm, 0)
        filename = tmp_path / f'Tiny.json_toy_{t}_1.0_traffic-matrix.pkl'
        with filename.open('wb') as stream:
            pickle.dump(tm, stream)
        problems.append(('Tiny.json', str(path), str(filename)))
    return module.TealEnv(obj='min_max_link_util', topo='Tiny.json', problems=problems,
                          num_path=2, edge_disjoint=False, dist_metric='min-hop', rho=1,
                          train_size=(0, 3), val_size=(3, 5), test_size=(5, 8), num_failure=0,
                          device=torch.device('cpu'), obs_ratio=0.5, test_obs_ratio=0.25,
                          obs_seed=7, hist_len=3, input_lag=1, repair_input='zero')


def test_real_env_future_isolation_and_nested_masks(real_env):
    import torch
    env = real_env
    train_mask = env.obs_mask.clone()
    env.reset('test')
    assert torch.all(env.obs_mask <= train_mask)
    assert int(env.obs_mask.sum()) == 3
    before = env.get_obs()['tm_seq'].clone()
    filename = env.problems[env.idx][2]
    with open(filename, 'wb') as stream:
        pickle.dump(np.full((4, 4), 9999.), stream)
    env.obs = env._read_obs()
    assert torch.equal(env.get_obs()['tm_seq'], before)
    env.reset('train')
    assert torch.equal(env.obs_mask, train_mask)
    assert torch.count_nonzero(env.get_obs()['tm_seq']) == 0


def test_real_actor_checkpoint_gate_and_no_optimizer(real_env, tmp_path):
    from lib.teal_actor import TealActor
    from lib.teal_model import Teal
    actor = TealActor(real_env, 1, str(tmp_path), False, real_env.device, auto_load=False)
    model = Teal(real_env, actor, 0.001, False, eval_only=True)
    assert model.actor_optimizer is None
    with pytest.raises(RuntimeError):
        model.train(1, 1, 1)
    gate_before = actor.FlowGNN.gated_index_values.clone()
    real_env.reset('test')
    actor.FlowGNN.refresh_graph()
    assert not gate_before.equal(actor.FlowGNN.gated_index_values)
    actor.eval()
    checkpoint = tmp_path / 'trained.pt'
    # 仅测试序列化契约；不将此人工 fixture 当成实验训练结果。
    save_checkpoint(checkpoint, actor.state_dict(), {'test_fixture': True}, 1)
    loaded = TealActor(real_env, 1, str(tmp_path), False, real_env.device,
                       checkpoint=str(checkpoint), checkpoint_config={'test_fixture': True})
    loaded.eval()
    weights = state_digest(loaded.state_dict())
    assert actor.act(real_env.get_obs()).equal(loaded.act(real_env.get_obs()))
    assert state_digest(loaded.state_dict()) == weights


def test_cli_train_then_fixed_checkpoint_eval(real_env, tmp_path, monkeypatch):
    import teal
    import teal_helper
    import torch
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(teal_helper, 'PROBLEM_NAMES', ['Tiny.json'])
    monkeypatch.setattr(teal_helper, 'get_problems', lambda args: real_env.problems)
    checkpoint = str(tmp_path / 'trained_real.pt')
    common = ['teal.py', '--topo', 'Tiny.json', '--tm-model', 'toy',
              '--obj', 'min_max_link_util', '--input-lag', '1', '--hist-len', '3',
              '--slice-train-stop', '3', '--slice-val-start', '3', '--slice-val-stop', '5',
              '--slice-test-start', '5', '--slice-test-stop', '8', '--num-path', '2',
              '--shared-paths', '--layers', '1', '--admm-steps', '0',
              '--train-obs-ratio', '0.5', '--obs-seed', '7', '--checkpoint', checkpoint]
    monkeypatch.setattr(sys, 'argv', common + ['--run-id', 'train', '--epochs', '1', '--bsz', '1', '--samples', '2'])
    args, output, problems = teal_helper.get_args_and_problems(teal.OUTPUT_CSV_TEMPLATE)
    teal.benchmark(problems, output, args)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload['trained_epochs'] == 1
    before = state_digest(payload['state_dict'])
    monkeypatch.setattr(sys, 'argv', common + ['--run-id', 'eval', '--eval-only', '--test-obs-ratio', '0.25'])
    args, output, problems = teal_helper.get_args_and_problems(teal.OUTPUT_CSV_TEMPLATE)
    teal.benchmark(problems, output, args)
    completed = json.loads((tmp_path / 'teal-logs/eval/completed.json').read_text())
    assert completed['weights_sha256'] == before
    assert completed['test_obs_ratio'] == 0.25
    assert completed['input_cutoff'] == 'target-1'
    assert len(Path(output).read_text().splitlines()) == 4


@pytest.mark.parametrize('precision', ['float64', 'float32'])
@pytest.mark.parametrize('chunk_size', [5, 100])
def test_offload_outputs_gradients_and_adam(precision, chunk_size):
    torch = pytest.importorskip('torch')
    from precheck_offload import verify_equivalence
    result = verify_equivalence(dtype=getattr(torch, precision), chunk_size=chunk_size)
    assert result['status'] == 'passed'
    assert result['steps'] == 3
    assert result['max_absolute_errors']['mlu'] <= result['absolute_tolerance']


def test_offload_zero_demand_ties_and_path_mask():
    torch = pytest.importorskip('torch')
    from revision_offload import maxutil_objective, normalized_path_weights
    # 原实现全维 max 在并列处均分梯度，不能换成只选一个下标的 max(dim)。
    utilization = torch.tensor([[2., 2., 1.], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    loss, maximum = maxutil_objective(utilization)
    loss.backward()
    assert loss.item() == 1
    torch.testing.assert_close(maximum, torch.tensor([2., 0.], dtype=torch.float64))
    torch.testing.assert_close(utilization.grad, torch.tensor([[0.125, 0.125, 0.],
                                                              [-1/6, -1/6, -1/6]], dtype=torch.float64))
    logits = torch.full((1, 2, 4), -10000., dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    ratios = normalized_path_weights(logits, valid)
    torch.testing.assert_close(ratios, torch.tensor([[[0.5, 0.5, 0., 0.], [1., 0., 0., 0.]]], dtype=torch.float64))
    ratios.sum().backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match='没有有效路径'):
        normalized_path_weights(logits, torch.zeros_like(valid))


def test_offload_inventory_matches_full_parameter_domain():
    pytest.importorskip('torch')
    from revision_offload import OffloadedMLP, offload_inventory
    report = offload_inventory(4, width=8, chunk_size=5)
    network = OffloadedMLP(12, width=8, chunk_size=5)
    assert report['parameters'] == sum(p.numel() for p in network.parameters())
    large = offload_inventory(1584)
    assert large['parameters'] == 2256774464
    assert large['cpu_parameter_gradient_adam_bytes'] / 2**30 > 67
    assert large['kernel_device_working_estimate_bytes'] / 2**30 < 2
    assert large['scope'] == 'MLP_only_excludes_path_graph_and_full_training'


def test_offload_budget_gate_and_tiny_cpu_kernel(monkeypatch):
    pytest.importorskip('torch')
    from types import SimpleNamespace
    import precheck_offload as probe
    args = SimpleNamespace(dtype='float64', nodes=4, paths=4, history=3, width=8,
                           chunk_size=5, batch=1, device='cpu', seed=0, steps=2)
    # 人工预算只用于验证小图 gate 分支，不是机器资源实测或大图放行。
    blocked = probe.benchmark(args, {'host_available_bytes': 1})
    assert blocked['status'] == 'blocked_host_budget'
    result = probe.benchmark(args, {'host_available_bytes': 4 * 2**30})
    assert result['status'] == 'kernel_passed_full_training_gate_still_closed'
    assert len(result['steps']) == 2
    assert all(row['total_seconds'] > 0 for row in result['steps'])
    assert 'cuda_peak_allocated_bytes' not in result


def test_offload_cli_verification_and_exclusive_report(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    import precheck_offload as probe
    # set_num_interop_threads 在同一进程不能重复设置；CLI 实测另行以独立进程执行。
    monkeypatch.setattr(probe.torch, 'set_num_interop_threads', lambda _: None)
    output = tmp_path / 'probe.json'
    monkeypatch.setattr(sys, 'argv', ['precheck_offload.py', '--verify-only', '--output', str(output)])
    assert probe.main() == 0
    report = json.loads(output.read_text())
    assert not report['full_training_gate_passed']
    assert report['status'] == 'equivalence_only_not_resource_gate'
    assert set(report['implementation_sha256']) == {'precheck_offload.py', 'revision_offload.py'}
    with pytest.raises(SystemExit):
        probe.main()


def test_seed_summary_preserves_conditions_and_detects_duplicates(tmp_path):
    sys.path.insert(0, str(ROOT / 'run'))
    import summarize_seeds
    csv_path = tmp_path / 'runs.csv'
    csv_path.write_text('config,rho,repair,seed,mlu\n'
                        'ours,0.3,nbr,2,1.7\n'
                        'ours,0.3,nbr,0,1.5\n'
                        'ours,0.3,zero,0,1.1\n'
                        'zero,0.3,nbr,0,2.0\n')
    rows = summarize_seeds.read_rows(csv_path)
    cells, columns = summarize_seeds.group_rows(rows)
    assert columns == ('config', 'rho', 'repair')
    assert len(cells) == 3
    ours = dict(cells)[('ours', '0.3', 'nbr')]
    summary = summarize_seeds.cell_summary(ours, expected_seeds=5)
    assert summary['mean'] == pytest.approx(1.6)
    assert summary['std'] == pytest.approx(2**-0.5 * 0.2)
    assert summary['seeds'] == ('0', '2')
    from io import StringIO
    output = StringIO()
    summarize_seeds.render(cells, columns, 5, compact=True, stream=output)
    assert 'config=ours rho=0.3 repair=nbr MLU=1.6000 ± 0.1414 (n=2/5' in output.getvalue()
    duplicated = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match='seed 重复'):
        summarize_seeds.group_rows(duplicated)


def test_phase_c_retry_requires_exact_known_missing_cells(tmp_path):
    sys.path.insert(0, str(ROOT / 'run'))
    import retry_phase_c
    source = tmp_path / 'norepair_main.csv'
    source.write_text('config,rho,seed,mlu\n'
                      'ours,0.05,0,1.0\n'
                      'ours,0.05,1,1.1\n'
                      'ours,0.1,0,1.2\n'
                      'ours,0.1,1,1.3\n'
                      'ours,0.1,2,1.4\n'
                      'ours,0.1,4,1.5\n'
                      'test-style,0.1,3,1.6\n'
                      'test-style,0.1,4,1.7\n')
    assert retry_phase_c.remaining_cells(source) == retry_phase_c.EXPECTED_MISSING
    command = retry_phase_c.command_for('/venv/python', '/repo/run/teal.py', 'ours', '0.05', '2')
    assert command[-4:] == ['--obs-ratio', '0.05', '--seed', '2']
    assert '--admm-steps' in command and command[command.index('--admm-steps') + 1] == '0'
    assert retry_phase_c.extract_result('Testing: 100% obj=2.1234 runtime=1') == pytest.approx(2.1234)
    assert retry_phase_c.extract_result('no completed test') is None
    source.write_text(source.read_text() + 'ours,0.05,2,1.8\n')
    with pytest.raises(ValueError, match='预期缺失格已存在'):
        retry_phase_c.remaining_cells(source)


def test_phase_cd_analysis_requires_same_seed_pairing(tmp_path):
    sys.path.insert(0, str(ROOT / 'run'))
    import analyze_phase_cd
    flow = tmp_path / 'flow.csv'
    node = tmp_path / 'node.csv'
    flow.write_text('config,rho,seed,mlu\nours,0.1,0,1.0\nours,0.1,1,1.2\n')
    node.write_text('config,rho,seed,mlu\nours,0.1,0,1.1\nours,0.1,2,0.9\n')
    report = analyze_phase_cd.node_flow_report(flow, node, tmp_path, include_coverage=False)
    ours = next(item for item in report['pairs'] if item['config'] == 'ours' and item['rho'] == 0.1)
    assert ours['paired_n'] == 1
    assert ours['shared_seeds'] == ['0']
    assert ours['delta_node_minus_flow']['mean'] == pytest.approx(0.1)
    assert analyze_phase_cd.sign_p_two_sided(0, 5) == pytest.approx(0.0625)
    assert analyze_phase_cd.sign_p_two_sided(3, 5) == pytest.approx(1.0)
    assert {'node_present': True, 'flow_present': False} in [
        {'node_present': item['node_present'], 'flow_present': item['flow_present']}
        for item in report['missing']
    ]
