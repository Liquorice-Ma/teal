#!/usr/bin/env python
"""按实验条件汇总多 seed 原始 CSV；不丢弃逐 seed 数据。

默认将除 seed 与指标列外的全部列作为分组键，因此不会混合不同
config、rho、repair、观测类型或容量。典型用法：

    python summarize_seeds.py norepair_main.csv --expected-seeds 5
    python summarize_seeds.py norepair_main.csv --expected-seeds 5 \
        --where config=ours --where rho=0.5 --compact
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_EXCLUDED = {'seed', 'mlu', 'final_obj', 'obj_val', 'runtime', 'runtime_s'}


def read_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('CSV 没有表头')
    if not rows[0]:
        raise ValueError('CSV 没有数据列')
    return rows


def parse_filters(items):
    filters = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'筛选条件必须是 列=值，而不是 {item!r}')
        key, value = item.split('=', 1)
        if not key or not value:
            raise ValueError(f'筛选条件必须是非空的 列=值，而不是 {item!r}')
        if key in filters and filters[key] != value:
            raise ValueError(f'同一列给出了冲突筛选: {key}')
        filters[key] = value
    return filters


def select_rows(rows, filters):
    fields = set(rows[0])
    unknown = sorted(set(filters) - fields)
    if unknown:
        raise ValueError(f'筛选列不存在: {unknown}')
    return [row for row in rows if all(row[key] == value for key, value in filters.items())]


def group_rows(rows, metric='mlu', group_columns=None):
    if not rows:
        return [], ()
    fields = tuple(rows[0])
    if metric not in fields or 'seed' not in fields:
        raise ValueError(f'CSV 必须包含 {metric!r} 与 seed 列')
    if group_columns is None:
        group_columns = tuple(key for key in fields if key not in DEFAULT_EXCLUDED | {metric})
    else:
        group_columns = tuple(group_columns)
    if not group_columns:
        raise ValueError('至少保留一个分组列，不能将不同实验条件混成一个均值')
    unknown = sorted(set(group_columns) - set(fields))
    if unknown:
        raise ValueError(f'分组列不存在: {unknown}')
    cells = defaultdict(dict)
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, ValueError) as error:
            raise ValueError(f'无效 {metric} 值: {row.get(metric)!r}') from error
        if not math.isfinite(value):
            raise ValueError(f'{metric} 必须为有限数值: {row.get(metric)!r}')
        key = tuple(row[column] for column in group_columns)
        seed = row['seed']
        if seed in cells[key]:
            raise ValueError(f'同一条件的 seed 重复: {dict(zip(group_columns, key))}, seed={seed}')
        cells[key][seed] = value
    return sorted(cells.items()), group_columns


def cell_summary(values, expected_seeds=None):
    ordered = [values[seed] for seed in sorted(values, key=seed_order)]
    return dict(mean=mean(ordered), std=stdev(ordered) if len(ordered) > 1 else 0.0,
                minimum=min(ordered), maximum=max(ordered), count=len(ordered),
                expected=expected_seeds, seeds=tuple(sorted(values, key=seed_order)))


def seed_order(seed):
    try:
        return (0, int(seed))
    except ValueError:
        return (1, seed)


def render(cells, columns, expected_seeds=None, compact=False, stream=sys.stdout):
    if not cells:
        print('没有匹配的完成结果。', file=stream)
        return
    summaries = [(key, cell_summary(values, expected_seeds)) for key, values in cells]
    if compact:
        for key, result in summaries:
            label = ' '.join(f'{column}={value}' for column, value in zip(columns, key))
            target = f'/{result["expected"]}' if result['expected'] is not None else ''
            print(f'[mean] {label} MLU={result["mean"]:.4f} ± {result["std"]:.4f} '
                  f'(n={result["count"]}{target}; range {result["minimum"]:.4f}–{result["maximum"]:.4f})',
                  file=stream)
        return
    widths = [max(len(column), *(len(key[i]) for key, _ in summaries))
              for i, column in enumerate(columns)]
    print('多 seed 汇总（MLU 越低越好；std 为样本标准差）', file=stream)
    header = '  '.join(column.ljust(widths[i]) for i, column in enumerate(columns))
    print(f'{header}  mean ± std          n       range                 seeds', file=stream)
    for key, result in summaries:
        label = '  '.join(value.ljust(widths[i]) for i, value in enumerate(key))
        target = f'/{result["expected"]}' if result['expected'] is not None else ''
        seeds = ','.join(result['seeds'])
        print(f'{label}  {result["mean"]:.4f} ± {result["std"]:.4f}  '
              f'{result["count"]}{target:<6} {result["minimum"]:.4f}–{result["maximum"]:.4f}  {seeds}',
              file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, help='包含 seed 和 mlu 列的原始结果 CSV')
    parser.add_argument('--metric', default='mlu', help='待汇总的数值列，默认 mlu')
    parser.add_argument('--group', help='逗号分隔的分组列；默认所有条件列')
    parser.add_argument('--where', action='append', default=[], metavar='列=值',
                        help='可重复的精确筛选条件')
    parser.add_argument('--expected-seeds', type=int, help='只显示完成进度 n/N，不补造缺失结果')
    parser.add_argument('--compact', action='store_true', help='每组输出一行，适合批处理日志')
    args = parser.parse_args(argv)
    if args.expected_seeds is not None and args.expected_seeds < 1:
        parser.error('--expected-seeds 必须为正数')
    try:
        rows = select_rows(read_rows(args.csv), parse_filters(args.where))
        group_columns = tuple(args.group.split(',')) if args.group else None
        cells, columns = group_rows(rows, args.metric, group_columns)
        render(cells, columns, args.expected_seeds, args.compact)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
