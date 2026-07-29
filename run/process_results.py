#!/usr/bin/env python
"""Process GPU experiment results: PR normalization + quick-look figures.

Reads results-summary.csv (written by run_experiments.sh) and
lp-oracle-*.csv (written by lp_oracle.py), aggregates over seeds
(mean +/- std), normalizes MLU into PR = U / U_opt, and renders every
figure the paper needs into result-figs/ as PNG for quick inspection.

Partial results are fine: figures whose data is missing are skipped.

Usage (on the server, inside run/):
    python process_results.py
    python process_results.py --summary results-summary.csv \
        --oracle lp-oracle-Starlink2272.json.csv
Note: PR here is mean-MLU / mean-U_opt over the test slice (summary-level
approximation); the camera-ready figures recompute PR per snapshot.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linestyle': '--'})

GREEN, ORANGE, BLUE, GRAY, RED = \
    '#59A14F', '#F18F01', '#457B9D', '#9AA0A6', '#E15759'
RHOS = [0.7, 0.5, 0.3, 0.1]
OUT = 'result-figs'

METHOD_STYLE = {
    'ours':   dict(color=GREEN, marker='o', ls='-', label='Ours'),
    'mean':   dict(color=ORANGE, marker='s', ls='--', label='Mean-interp.'),
    'zero':   dict(color=BLUE, marker='^', ls='--', label='Zero-fill'),
    'oracle': dict(color=GRAY, ls=':', label='Full-obs oracle'),
}


def parse_summary(path):
    """Parse 'A obj=x method=y rho=0.3 seed=0' tags into a DataFrame."""
    rows = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        parts = str(r['tag']).split()
        d = {'batch': parts[0], 'obj': 'min_max_link_util'}
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=', 1)
                d[k] = v
        d['final_obj'] = float(r['final_obj'])
        d['runtime'] = float(r.get('runtime_s', np.nan))
        rows.append(d)
    out = pd.DataFrame(rows)
    if 'rho' in out:
        out['rho'] = pd.to_numeric(out['rho'], errors='coerce')
    return out


def agg(df, keys):
    """Aggregate final_obj over seeds -> mean, std."""
    g = df.groupby(keys)['final_obj'].agg(['mean', 'std', 'count'])
    return g.reset_index()


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  saved', name)


def to_pr(df, u_opt):
    """Normalize MLU columns into PR = U / U_opt (summary-level approx)."""
    df = df.copy()
    df['mean'] = df['mean'] / u_opt
    df['std'] = df['std'] / u_opt
    return df


def fig_main(df, u_opt):
    """Batch A: PR vs rho (MLU) and satisfied ratio vs rho (flow)."""
    for obj, ylabel, fname in [
            ('min_max_link_util', 'performance ratio $PR$', 'A-pr-vs-rho.png'),
            ('total_flow', 'satisfied demand ratio', 'A-flow-vs-rho.png')]:
        sub = df[(df.batch == 'A') & (df.obj == obj)]
        if sub.empty:
            print('  [skip] batch A,', obj)
            continue
        a = agg(sub, ['method', 'rho'])
        if obj == 'min_max_link_util' and u_opt:
            a = to_pr(a, u_opt)
        fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
        for m in ['ours', 'mean', 'zero']:
            s = a[a.method == m].sort_values('rho', ascending=False)
            if s.empty:
                continue
            st = METHOD_STYLE[m]
            ax.errorbar(s.rho, s['mean'], yerr=s['std'], capsize=2, **st)
        orc = a[a.method == 'oracle']
        if not orc.empty:
            ax.axhline(orc['mean'].iloc[0], **METHOD_STYLE['oracle'])
        ax.set_xlabel(r'observability ratio $\rho$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.75, 0.05)
        ax.legend(fontsize=7)
        save(fig, fname)


def fig_ablation(df, u_opt):
    """Batch B: ablation across rho + bar chart at rho=0.3."""
    full = df[(df.batch == 'A') & (df.obj == 'min_max_link_util')
              & (df.method == 'ours')]
    abl = df[df.batch == 'B']
    if abl.empty:
        print('  [skip] batch B')
        return
    a_full = agg(full, ['rho']).assign(ablate='full')
    a_abl = agg(abl, ['ablate', 'rho'])
    allc = pd.concat([a_full, a_abl], ignore_index=True)
    if u_opt:
        allc = to_pr(allc, u_opt)
    styles = {'full': dict(color=GREEN, marker='o', ls='-'),
              '-temporal': dict(color=ORANGE, marker='s', ls='--'),
              '-gate': dict(color=BLUE, marker='D', ls='--'),
              '-embed': dict(color=RED, marker='^', ls='--')}
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    for k, st in styles.items():
        s = allc[allc.ablate == k].sort_values('rho', ascending=False)
        if not s.empty:
            ax.errorbar(s.rho, s['mean'], yerr=s['std'], capsize=2,
                        label=k, **st)
    ax.set_xlabel(r'observability ratio $\rho$')
    ax.set_ylabel('performance ratio $PR$')
    ax.set_xlim(0.75, 0.05)
    ax.legend(fontsize=7)
    save(fig, 'B-ablation-vs-rho.png')

    at3 = allc[allc.rho == 0.3]
    if not at3.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.8), constrained_layout=True)
        order = [k for k in styles if k in set(at3.ablate)]
        vals = [at3[at3.ablate == k] for k in order]
        ax.bar(order, [v['mean'].iloc[0] for v in vals],
               yerr=[v['std'].iloc[0] for v in vals], capsize=3,
               color=[styles[k]['color'] for k in order])
        ax.set_ylabel('performance ratio $PR$')
        ax.set_title(r'ablation at $\rho=0.3$', fontsize=9)
        save(fig, 'B-ablation-bar.png')


def fig_zero_retrain(df, u_opt):
    """Batch C: print a text table (paper Table 4/5 material)."""
    sub = df[df.batch == 'C']
    base = df[(df.batch == 'A') & (df.obj == 'min_max_link_util')
              & (df.method == 'ours') & (df.rho == 0.3)]
    if sub.empty:
        print('  [skip] batch C')
        return
    lines = ['=== zero-retraining (MLU, rho=0.3) ===']
    if not base.empty:
        b = agg(base, ['method'])
        lines.append('baseline (same-set)   : %.4f +/- %.4f'
                     % (b['mean'].iloc[0], b['std'].iloc[0]))
    for exp in sorted(sub['exp'].dropna().unique()):
        e = sub[sub.exp == exp]
        for obj in e['obj'].unique():
            eo = e[e.obj == obj]
            lines.append('%-22s: %.4f +/- %.4f  (%s, n=%d)' % (
                exp, eo['final_obj'].mean(), eo['final_obj'].std(),
                obj, len(eo)))
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT, 'C-zero-retrain.txt'), 'w') as f:
        f.write(text + '\n')


def fig_sensitivity(df, u_opt):
    """Batch D: sampling granularity, hist-len, capacity, GDP."""
    sub = df[df.batch == 'D']
    if sub.empty:
        print('  [skip] batch D')
        return
    # sampling granularity: flow(=A ours) vs node vs top
    flow = df[(df.batch == 'A') & (df.obj == 'min_max_link_util')
              & (df.method == 'ours')].assign(exp='flow')
    samp = pd.concat([flow, sub[sub.exp.isin(['node', 'top'])]],
                     ignore_index=True)
    a = agg(samp, ['exp', 'rho'])
    if u_opt:
        a = to_pr(a, u_opt)
    if not a.empty:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
        for k, c, mk in [('flow', GREEN, 'o'), ('node', BLUE, '^'),
                         ('top', ORANGE, 's')]:
            s = a[a.exp == k].sort_values('rho', ascending=False)
            if not s.empty:
                ax.errorbar(s.rho, s['mean'], yerr=s['std'], capsize=2,
                            color=c, marker=mk, label=k + ' sampling')
        ax.set_xlabel(r'observability ratio $\rho$')
        ax.set_ylabel('performance ratio $PR$')
        ax.set_xlim(0.75, 0.05)
        ax.legend(fontsize=7)
        save(fig, 'D-sampling.png')

    # hist-len: 1(B -temporal) / 3(A ours) / 5 / 8 at rho=0.3
    h13 = pd.concat([
        df[(df.batch == 'B') & (df.ablate == '-temporal')
           & (df.rho == 0.3)].assign(hist='1'),
        df[(df.batch == 'A') & (df.obj == 'min_max_link_util')
           & (df.method == 'ours') & (df.rho == 0.3)].assign(hist='3'),
        sub[sub.exp == 'hist5'].assign(hist='5'),
        sub[sub.exp == 'hist8'].assign(hist='8')], ignore_index=True)
    a = agg(h13, ['hist'])
    if u_opt:
        a = to_pr(a, u_opt)
    if len(a) > 1:
        fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
        ax.bar(a['hist'], a['mean'], yerr=a['std'], capsize=3, color=GREEN)
        ax.set_xlabel('history length $L$')
        ax.set_ylabel('performance ratio $PR$')
        save(fig, 'D-histlen.png')

    # capacity / gdp: grouped text + bar
    for exp_prefix, fname in [('cap', 'D-capacity.png'), ('gdp', 'D-gdp.png')]:
        e = sub[sub.exp.astype(str).str.startswith(exp_prefix)]
        if e.empty:
            continue
        a = agg(e, ['exp', 'method'])
        fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
        labels = sorted(a['exp'].unique())
        width, x = 0.35, np.arange(len(labels))
        for i, (m, c) in enumerate([('ours', GREEN), ('zero', BLUE)]):
            s = a[a.method == m].set_index('exp').reindex(labels)
            ax.bar(x + (i - 0.5) * width, s['mean'], width, yerr=s['std'],
                   capsize=3, color=c, label=m)
        ax.set_xticks(x, labels)
        ax.set_ylabel('MLU (raw)')
        ax.legend(fontsize=7)
        save(fig, fname)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results-summary.csv')
    ap.add_argument('--oracle', default='lp-oracle-Starlink2272.json.csv')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = parse_summary(args.summary)
    print('parsed %d runs (batches: %s)'
          % (len(df), sorted(df.batch.unique())))

    u_opt = None
    if os.path.exists(args.oracle):
        u_opt = pd.read_csv(args.oracle)['opt_mlu'].mean()
        print('U_opt (mean over test snapshots) = %.4f' % u_opt)
    else:
        print('[warn] %s not found -- MLU plotted RAW, not PR-normalized'
              % args.oracle)

    fig_main(df, u_opt)
    fig_ablation(df, u_opt)
    fig_zero_retrain(df, u_opt)
    fig_sensitivity(df, u_opt)
    print('done -> %s/' % OUT)
