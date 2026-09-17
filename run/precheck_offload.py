"""同语义 offload 内核预检；不读取测试结果、不启动正式训练或给全量实验放行。"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
# 支持将两个自有脚本放在隔离目录执行，不依赖主工程或第三方 DOTE 源码。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from revision_offload import OffloadedMLP, maxutil_objective, normalized_path_weights, offload_inventory

GIB = 2**30


def verify_equivalence(device='cpu', dtype=torch.float64, steps=3, chunk_size=5):
    """完整四节点小图：复制同一权重，逐项对照输出、梯度、Adam 一二阶矩与更新。"""
    device = torch.device(device)
    torch.manual_seed(42)
    pairs, paths, width, history = 12, 4, 8, 3
    dims = [pairs * history, width, width, width, width, pairs * paths]
    layers = [torch.nn.Linear(a, b).to(device=device, dtype=dtype)
              for a, b in zip(dims, dims[1:])]
    staged = OffloadedMLP(pairs, paths, history, width, chunk_size, dtype)
    for source, target in zip(layers, staged.layers):
        target.copy_from_dense(source)
    dense_opt = torch.optim.Adam([p for layer in layers for p in layer.parameters()],
                                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    staged_opt = torch.optim.Adam(staged.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    od = [(s, t) for s in range(4) for t in range(4) if s != t]
    edge_index = {edge: i for i, edge in enumerate(od)}
    incidence = torch.zeros((pairs * paths, len(od)), dtype=dtype, device=device)
    for q, (s, t) in enumerate(od):
        a, b = [n for n in range(4) if n not in (s, t)]
        candidates = [[s, t], [s, a, t], [s, b, t], [s, a, b, t]]
        for k, path in enumerate(candidates):
            for edge in zip(path[:-1], path[1:]):
                incidence[q * paths + k, edge_index[edge]] = 1
    valid = torch.ones((pairs, paths), dtype=torch.bool, device=device)
    valid[0, -1] = False
    valid[1, 1:] = False
    capacity = torch.linspace(0.8, 2.0, len(od), dtype=dtype, device=device)
    metrics = {}
    atol, rtol = (1e-11, 1e-9) if dtype == torch.float64 else (1e-6, 2e-5)

    def compare(name, left, right):
        left, right = left.detach().cpu(), right.detach().cpu()
        metrics[name] = max(metrics.get(name, 0.0), float((left - right).abs().max()))
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)

    for step in range(steps):
        dense_opt.zero_grad(set_to_none=True)
        staged_opt.zero_grad(set_to_none=True)
        x = torch.rand((2, pairs * history), dtype=dtype, device=device, requires_grad=True)
        other_x = x.detach().clone().requires_grad_(True)
        demand = torch.rand((2, pairs), dtype=dtype, device=device)
        demand[:, step % pairs] = 0  # 真值为零的 OD 仍保留在固定输入/输出域。
        dense_logits = x
        for layer in layers[:-1]:
            dense_logits = torch.relu(layer(dense_logits))
        dense_logits = layers[-1](dense_logits).reshape(2, pairs, paths)
        offload_logits = staged(other_x).reshape(2, pairs, paths)
        # 独立参考逐 OD 归一化，不调用被测试归一化函数。
        dense_weights = (torch.sigmoid(dense_logits) + 1e-16) * valid
        reference = torch.stack([dense_weights[:, q] / dense_weights[:, q].sum(dim=-1, keepdim=True)
                                 for q in range(pairs)], dim=1)
        actual = normalized_path_weights(offload_logits, valid)
        compare('logits', offload_logits, dense_logits)
        compare('ratios', actual, reference)
        dense_util = (reference * demand[..., None]).flatten(1) @ incidence / capacity
        actual_util = (actual * demand[..., None]).flatten(1) @ incidence / capacity
        maxima = torch.stack([row.max() for row in dense_util])
        reference_loss = torch.stack([m / m.detach() if m.detach().item() != 0 else 1 - m
                                      for m in maxima]).mean()
        actual_loss, actual_max = maxutil_objective(actual_util)
        compare('mlu', actual_max, maxima)
        compare('loss', actual_loss, reference_loss)
        reference_loss.backward()
        actual_loss.backward()
        compare('input_gradient', other_x.grad, x.grad)
        for dense, split in zip(layers, staged.layers):
            compare('weight_gradient', torch.cat([p.grad for p in split.weights], dim=split.axis), dense.weight.grad)
            compare('bias_gradient', split.bias.grad, dense.bias.grad)
        dense_opt.step()
        staged_opt.step()
        for dense, split in zip(layers, staged.layers):
            compare('updated_weight', torch.cat(list(split.weights), dim=split.axis), dense.weight)
            compare('updated_bias', split.bias, dense.bias)
            for state_name in ('exp_avg', 'exp_avg_sq'):
                compare(state_name, torch.cat([staged_opt.state[p][state_name] for p in split.weights], dim=split.axis),
                        dense_opt.state[dense.weight][state_name])
                compare(state_name, staged_opt.state[split.bias][state_name], dense_opt.state[dense.bias][state_name])
            for p in split.parameters():
                assert p.device.type == 'cpu' and p.grad.device.type == 'cpu'
                assert int(staged_opt.state[p]['step']) == step + 1
                assert all(v.device.type == 'cpu' for v in staged_opt.state[p].values() if torch.is_tensor(v))
    return dict(status='passed', device=str(device), dtype=str(dtype), steps=steps,
                absolute_tolerance=atol, relative_tolerance=rtol, max_absolute_errors=metrics)


def memory_environment():
    """同时查看宿主可用内存与当前 cgroup 配额，不能把宿主总 RAM 当容器预算。"""
    available, limits = None, []
    if sys.platform.startswith('linux'):
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(fields['MemAvailable'].split()[0]) * 1024
        roots = {Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')}
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, group = line.split(':', 2)
            if controllers == '' or 'memory' in controllers.split(','):
                roots.add(Path('/sys/fs/cgroup') / group.lstrip('/'))
                roots.add(Path('/sys/fs/cgroup/memory') / group.lstrip('/'))
        for root in roots:
            for limit_name, used_name in [('memory.max', 'memory.current'),
                                          ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
                limit_path, used_path = root / limit_name, root / used_name
                if limit_path.exists() and used_path.exists():
                    text = limit_path.read_text().strip()
                    if text != 'max' and int(text) < 2**60:
                        limit, used = int(text), int(used_path.read_text().strip())
                        available = min(available, max(0, limit - used))
                        limits.append(dict(path=str(root), limit_bytes=limit, used_bytes=used))
    result = dict(host_available_bytes=available, cgroup_limits=limits, torch_version=torch.__version__,
                  python=sys.version, cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_name=torch.cuda.get_device_name(), gpu_free_bytes=free, gpu_total_bytes=total)
    return result


def benchmark(args, environment):
    dtype = getattr(torch, args.dtype)
    estimate = offload_inventory(args.nodes, args.paths, args.history, args.width,
                                 args.chunk_size, torch.empty((), dtype=dtype).element_size(), args.batch)
    cpu_bytes = estimate['cpu_parameter_gradient_adam_bytes']
    working = estimate['kernel_device_working_estimate_bytes']
    # 额外空间覆盖 allocator 缓存、梯度交接与小块 Adam 临时张量。
    required_host = int(cpu_bytes * 1.25 + working + GIB)
    available = environment['host_available_bytes']
    if available is None or required_host > available * 0.75:
        return dict(status='blocked_host_budget', estimate=estimate, required_host_bytes=required_host)
    if args.device == 'cuda':
        if not environment['cuda_available']:
            return dict(status='blocked_no_cuda', estimate=estimate)
        total, free = environment['gpu_total_bytes'], environment['gpu_free_bytes']
        process_limit = min(4 * GIB, free - 0.25 * total)
        if working + 512 * 2**20 > process_limit:
            return dict(status='blocked_gpu_budget', estimate=estimate, process_limit_bytes=process_limit)
        torch.cuda.set_per_process_memory_fraction(process_limit / total)
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = OffloadedMLP(estimate['pairs'], args.paths, args.history, args.width, args.chunk_size, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    initialization_seconds = time.perf_counter() - started
    inputs = torch.rand((args.batch, args.history * estimate['pairs']), dtype=dtype, device=args.device)
    path_cost = torch.arange(1, args.paths + 1, dtype=dtype, device=args.device)
    timings = []

    def synchronized_time():
        if args.device == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        start = synchronized_time()
        logits = model(inputs).reshape(args.batch, estimate['pairs'], args.paths)
        # 仅作内核压力测试：非实验 MLU、非训练结果；所有 OD/输出都参与。
        loss = (normalized_path_weights(logits) * path_cost).sum(dim=-1).mean()
        forward = synchronized_time()
        loss.backward()
        backward = synchronized_time()
        optimizer.step()
        end = synchronized_time()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and p.grad.device.type == 'cpu' for p in model.parameters())
        timings.append(dict(step=step, forward_seconds=forward - start,
                            backward_seconds=backward - forward, adam_seconds=end - backward,
                            total_seconds=end - start))
        print(json.dumps(dict(event='kernel_step', **timings[-1])), file=sys.stderr, flush=True)
        del logits, loss
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(status='kernel_passed_full_training_gate_still_closed', estimate=estimate,
                  initialization_seconds=initialization_seconds, steps=timings,
                  process_peak_rss_bytes=peak * (1024 if sys.platform.startswith('linux') else 1),
                  scope='synthetic_MLP_kernel_not_full_routing_or_experiment',
                  cpu_threads=torch.get_num_threads(), precision=args.dtype)
    if args.device == 'cuda':
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    parser.add_argument('--nodes', type=int, default=16)
    parser.add_argument('--paths', type=int, default=4)
    parser.add_argument('--history', type=int, default=3)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--chunk-size', type=int, default=32768)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.steps - 1, args.threads, args.batch, args.paths, args.history, args.width, args.chunk_size, args.nodes - 1) < 1:
        parser.error('至少两个更新步；维度和线程数必须为正，节点数至少为2')
    if args.output and args.output.exists():
        parser.error('输出文件已存在，拒绝覆盖')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    environment = memory_environment()
    import revision_offload
    code_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (Path(__file__), Path(revision_offload.__file__))}
    report = dict(schema_version=1, environment=environment, arguments={**vars(args), 'output': str(args.output)},
                  implementation_sha256=code_hashes, full_training_gate_passed=False)
    if args.inspect_only:
        report['status'] = 'inspection_only'
    elif args.device == 'cuda' and not environment['cuda_available']:
        report['status'] = 'blocked_no_cuda'
    elif args.device == 'cuda' and environment['gpu_free_bytes'] < 0.25 * environment['gpu_total_bytes'] + GIB:
        report['status'] = 'blocked_gpu_headroom_before_verification'
    else:
        try:
            report['equivalence'] = verify_equivalence(args.device, getattr(torch, args.dtype))
            if args.verify_only:
                report['status'] = 'equivalence_only_not_resource_gate'
            else:
                report['benchmark'] = benchmark(args, memory_environment())
                report['status'] = report['benchmark']['status']
        except (AssertionError, RuntimeError, MemoryError) as error:
            report['status'] = 'failed_precheck'
            report['error'] = f'{type(error).__name__}: {error}'
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output:
        with args.output.open('x') as output:
            output.write(text + '\n')
    return 2 if report['status'].startswith(('blocked', 'failed')) else 0


if __name__ == '__main__':
    raise SystemExit(main())
