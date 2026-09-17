#!/usr/bin/env python
"""隔离且串行地补跑 Phase C 已失败的七个 seed。

不改写 norepair_main.csv、完成标记或既有日志。每个重试格使用独立工作目录，
原始逐快照输出、标准输出、资源快照和提取的单值结果均保存在输出目录。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


EXPECTED_MISSING = (
    ('ours', '0.05', '2'),
    ('ours', '0.05', '3'),
    ('ours', '0.05', '4'),
    ('ours', '0.1', '3'),
    ('test-style', '0.1', '0'),
    ('test-style', '0.1', '1'),
    ('test-style', '0.1', '2'),
)
ARCHITECTURES = {
    'ours': ('--mask-mode', 'embed', '--hist-len', '3'),
    'test-style': ('--mask-mode', 'zero', '--no-gate', '--hist-len', '3'),
}
BASE = (
    '--shared-paths', '--deterministic', '--obj', 'min_max_link_util',
    '--topo', 'Starlink2272.json', '--tm-model', 'starlink', '--prune-demands',
    '--samples', '30', '--slice-train-start', '0', '--slice-train-stop', '80',
    '--slice-val-start', '80', '--slice-val-stop', '90',
    '--slice-test-start', '90', '--slice-test-stop', '101',
    '--epochs', '60', '--early-stop', 'True', '--lr', '0.0001',
    '--num-restart', '3', '--warmup-epochs', '4', '--admm-steps', '0',
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def completed_cells(path):
    cells = set()
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row.get('config'), row.get('rho'), row.get('seed'))
            if None in key:
                raise ValueError(f'原始 CSV 缺少 config/rho/seed: {row}')
            if key in cells:
                raise ValueError(f'原始 CSV 有重复条件: {key}')
            cells.add(key)
    return cells


def remaining_cells(source_csv):
    completed = completed_cells(source_csv)
    missing = tuple(cell for cell in EXPECTED_MISSING if cell not in completed)
    unexpected = sorted(set(EXPECTED_MISSING) & completed)
    if unexpected:
        raise ValueError(f'这些预期缺失格已存在于原始 CSV，拒绝重复重试: {unexpected}')
    return missing


def command_for(python, teal, config, rho, seed):
    if config not in ARCHITECTURES:
        raise ValueError(f'未定义的 Phase C 架构: {config}')
    return [str(python), str(teal), *BASE, *ARCHITECTURES[config],
            '--obs-ratio', rho, '--seed', seed]


def command_output(command, cwd=None, check=True):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if check and result.returncode:
        raise RuntimeError(f'命令失败({result.returncode}): {command}\n{result.stdout}')
    return result


def gpu_snapshot():
    result = command_output([
        'nvidia-smi', '--query-gpu=memory.free,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], check=False)
    if result.returncode:
        raise RuntimeError(f'nvidia-smi 不可用: {result.stdout}')
    values = [part.strip() for part in result.stdout.strip().split(',')]
    if len(values) != 4:
        raise RuntimeError(f'无法解析 nvidia-smi 输出: {result.stdout!r}')
    free, total, used, utilization = map(int, values)
    return dict(free_mib=free, total_mib=total, used_mib=used,
                utilization_percent=utilization)


def no_active_teal():
    result = command_output(['pgrep', '-af', r'(^|/)python.*teal\.py( |$)'], check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f'无法检查活动 teal.py: {result.stdout}')
    return result.returncode == 1


def preflight():
    active_free = no_active_teal()
    gpu = gpu_snapshot()
    # 只在完全没有其它 teal 任务、GPU 至少空闲 75% 时启动，超过计划的25%余量要求。
    if not active_free:
        raise RuntimeError('存在活动 teal.py；拒绝与其它批次并发重试')
    if gpu['free_mib'] < gpu['total_mib'] * 0.75:
        raise RuntimeError(f'GPU 空闲 {gpu["free_mib"]}/{gpu["total_mib"]} MiB，低于75%阈值')
    return gpu


def extract_result(log_text):
    normalized = log_text.replace('\r', '\n')
    matches = re.findall(r'Testing:\s*100[^\n]*obj=([0-9.]+)', normalized)
    if not matches:
        return None
    return float(matches[-1])


def make_manifest(args, missing):
    repo = args.repo
    files = [repo / 'run' / 'teal.py', repo / 'run' / 'teal_helper.py',
             repo / 'lib' / 'teal_env.py', repo / 'lib' / 'teal_actor.py',
             repo / 'lib' / 'teal_model.py', Path(__file__)]
    git = command_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=False)
    return dict(
        schema_version=1,
        created_utc=now(),
        purpose='isolated_serial_retry_for_preexisting_phase_c_failures',
        source_csv=str(args.source_csv),
        source_csv_sha256=digest(args.source_csv),
        expected_missing=[dict(config=c, rho=r, seed=s) for c, r, s in missing],
        repo=str(repo),
        git_head=git.stdout.strip() if git.returncode == 0 else None,
        implementation_sha256={str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path): digest(path)
                               for path in files},
        fixed_protocol=dict(base=list(BASE), architectures={key: list(value)
                                                            for key, value in ARCHITECTURES.items()}),
        source_mutated=False,
        full_training_gate_passed=False,
    )


def write_csv_row(path, row, fields):
    created = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if created:
            writer.writeheader()
        writer.writerow(row)


def run_cell(args, output, config, rho, seed):
    tag = f'norepair-{config}-{rho}-{seed}'
    case = output / 'cases' / tag
    case.mkdir(parents=True, exist_ok=False)
    before = preflight()
    command = command_for(args.python, args.repo / 'run' / 'teal.py', config, rho, seed)
    (case / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    (case / 'preflight.json').write_text(json.dumps(dict(utc=now(), gpu=before), indent=2) + '\n')
    # teal.py 的历史入口假设当前目录是 run/；隔离工作目录时显式提供
    # 仓库根路径，避免把输出重新写进既有 run/ 目录。
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(args.repo) + os.pathsep + environment.get('PYTHONPATH', '')
    result = subprocess.run(command, cwd=case, env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = case / 'stdout.log'
    log.write_text(result.stdout)
    after = gpu_snapshot()
    (case / 'postflight.json').write_text(json.dumps(dict(utc=now(), gpu=after,
                                                           returncode=result.returncode), indent=2) + '\n')
    value = extract_result(result.stdout) if result.returncode == 0 else None
    raw = case / 'teal-min_max_link_util-all.csv'
    if value is not None and not raw.exists():
        value = None
    row = dict(config=config, rho=rho, seed=seed, tag=tag, command_sha256=digest(case / 'command.json'),
               log_sha256=digest(log), raw_csv_sha256=digest(raw) if raw.exists() else '',
               source='isolated_serial_retry_v1', completed_utc=now())
    if value is None:
        reason = 'oom' if 'OutOfMemoryError' in result.stdout else 'no_complete_testing_result'
        row.update(status='failed', reason=reason, returncode=result.returncode)
        write_csv_row(output / 'failures.csv', row,
                      ('config', 'rho', 'seed', 'tag', 'status', 'reason', 'returncode',
                       'command_sha256', 'log_sha256', 'raw_csv_sha256', 'source', 'completed_utc'))
        return row
    row.update(status='completed', mlu=f'{value:.10g}', returncode=result.returncode)
    write_csv_row(output / 'retry_results.csv', row,
                  ('config', 'rho', 'seed', 'mlu', 'tag', 'status', 'returncode',
                   'command_sha256', 'log_sha256', 'raw_csv_sha256', 'source', 'completed_utc'))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Teal 仓库绝对路径')
    parser.add_argument('--source-csv', type=Path, required=True, help='既有 Phase C CSV，只读')
    parser.add_argument('--output', type=Path, required=True, help='新的隔离输出目录，必须不存在')
    parser.add_argument('--python', type=Path, required=True, help='隔离环境使用的 Python 解释器')
    args = parser.parse_args(argv)
    args.repo, args.source_csv, args.output, args.python = (path.resolve() for path in
                                                              (args.repo, args.source_csv, args.output, args.python))
    if args.output.exists():
        parser.error(f'输出目录已存在，拒绝覆盖: {args.output}')
    if not args.source_csv.is_file() or not args.python.is_file():
        parser.error('source-csv 或 python 不存在')
    if not (args.repo / 'run' / 'teal.py').is_file():
        parser.error('repo 中不存在 run/teal.py')
    missing = remaining_cells(args.source_csv)
    if missing != EXPECTED_MISSING:
        parser.error(f'原始 CSV 与此重试批次预期不一致: {missing}')
    preflight()
    args.output.mkdir(parents=True)
    shutil.copy2(args.source_csv, args.output / 'source_norepair_main.csv')
    (args.output / 'manifest.json').write_text(json.dumps(make_manifest(args, missing), indent=2) + '\n')
    outcomes = []
    for config, rho, seed in missing:
        outcomes.append(run_cell(args, args.output, config, rho, seed))
    summary = dict(completed=sum(row['status'] == 'completed' for row in outcomes),
                   failed=sum(row['status'] != 'completed' for row in outcomes),
                   outcomes=outcomes, finished_utc=now())
    (args.output / 'completed.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['failed'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
