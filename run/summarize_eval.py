#!/usr/bin/env python
"""Summarize eval_denoised.csv the way the reference papers report results.

Design choices (see docs/experiment_design.md):
  - PR = MLU / MLU_opt as the primary metric (TEST/DOTE/HARP style).
  - HARP-style percentile bound: "PR <= X over Y% of runs", which is
    robust to the occasional divergent-seed outlier (e.g. a run stuck at
    val 3.2 while the rest sit near 1.5). We report median and P90 rather
    than mean +/- std for exactly this reason.
  - ELATE-style ablation gap: how much each removed module costs.

Tolerates partial results: whatever is in the CSV is summarized.

Usage (in run/):  python summarize_eval.py
"""

import os

import numpy as np
import pandas as pd

CSV = 'eval_denoised.csv'
ORACLE = 'lp-oracle-shared.csv'

if not os.path.exists(CSV):
    raise SystemExit('no %s yet' % CSV)

df = pd.read_csv(CSV)
u_opt = pd.read_csv(ORACLE)['opt_mlu'].mean() if os.path.exists(ORACLE) else None
if u_opt:
    df['pr'] = df['mlu']/u_opt

print('runs so far: %d | seeds: %s | configs: %s'
      % (len(df), sorted(df.seed.unique()), sorted(df.config.unique())))
if u_opt:
    print('LP oracle MLU = %.4f  (PR = MLU / this)\n' % u_opt)

metric = 'pr' if u_opt else 'mlu'
name = 'PR' if u_opt else 'MLU'


def summarize(sub):
    """median / P90 / n over a group (lower is better)."""
    v = sub[metric].to_numpy()
    return np.median(v), np.percentile(v, 90), len(v)


# ---- per (rho, config): median PR, P90 PR ----
print('=== %s by observability and config (median / P90) ===' % name)
print('%-6s %-12s %-9s %-9s %s' % ('rho', 'config', 'median', 'P90', 'n'))
for rho in sorted(df.rho.unique(), reverse=True):
    for cfg in ['ours', 'no-embed', 'no-gate', 'no-temporal',
                'nbr', 'zero-fill', 'mean-interp', 'full']:
        sub = df[(df.rho == rho) & (df.config == cfg)]
        if sub.empty:
            continue
        med, p90, n = summarize(sub)
        print('%-6s %-12s %-9.3f %-9.3f %d' % (rho, cfg, med, p90, n))
    print()

# ---- HARP-style headline bound for ours ----
ours = df[df.config == 'ours']
if u_opt and not ours.empty:
    frac = (ours['pr'] <= 1.5).mean()*100
    print('--- HARP-style bound (ours) ---')
    print('PR <= 1.5 over %.0f%% of runs; median PR = %.3f, P90 = %.3f\n'
          % (frac, ours['pr'].median(), ours['pr'].quantile(0.9)))

# ---- ablation gap at each rho (median, vs ours) ----
print('=== ablation gap (median %s, relative to ours; + = worse) ===' % name)
for rho in sorted([r for r in df.rho.unique() if r < 1.0], reverse=True):
    o = df[(df.rho == rho) & (df.config == 'ours')]
    if o.empty:
        continue
    om = np.median(o[metric])
    print('  rho=%-5s ours=%.3f' % (rho, om))
    for cfg in ['no-embed', 'no-gate', 'no-temporal',
                'zero-fill', 'mean-interp']:
        sub = df[(df.rho == rho) & (df.config == cfg)]
        if sub.empty:
            continue
        gap = (np.median(sub[metric]) - om)/om*100
        print('            %-12s %+.1f%%' % (cfg, gap))


# ---- imputation-strategy view: does filling values help at all? ----
# The comparison must hold the structural modules fixed. no-embed keeps the
# gate and the temporal encoder and only drops the embedding, whereas
# zero-fill also removes both modules, so the two are not a clean
# fill-strategy pair.
print('\n=== fill strategy at fixed structure (median %s) ===' % name)
print('%-6s %-18s %-9s %s' % ('rho', 'fill', 'median', 'vs zero fill'))
for rho in sorted([r for r in df.rho.unique() if r < 1.0], reverse=True):
    for label, group in [
            ('WITH gate+temporal',
             [('zero', 'no-embed'), ('learned-placeholder', 'ours'),
              ('neighbor-est', 'nbr'), ('global-mean', 'mean-gated')]),
            ('WITHOUT gate+temporal',
             [('zero', 'zero-fill'), ('global-mean', 'mean-interp')])]:
        ref = df[(df.rho == rho) & (df.config == group[0][1])]
        if ref.empty:
            continue
        refm = np.median(ref[metric])
        print('%-6s [%s]' % (rho, label))
        for fill, cfg in group:
            sub = df[(df.rho == rho) & (df.config == cfg)]
            if sub.empty:
                continue
            val = np.median(sub[metric])
            delta = '' if cfg == group[0][1] \
                else '%+.1f%%' % ((val - refm)/refm*100)
            print('       %-18s %-9.3f %s' % (fill, val, delta))

# ---- structural contribution: same fill (zero), modules on vs off ----
print('\n=== structural contribution (zero fill in both arms) ===')
for rho in sorted([r for r in df.rho.unique() if r < 1.0], reverse=True):
    a = df[(df.rho == rho) & (df.config == 'no-embed')]
    b = df[(df.rho == rho) & (df.config == 'zero-fill')]
    if a.empty or b.empty:
        continue
    am, bm = np.median(a[metric]), np.median(b[metric])
    print('  rho=%-5s with modules %.3f vs without %.3f  (%+.1f%%)'
          % (rho, am, bm, (bm - am)/am*100))

# ---- degradation vs full observation (LMTE style) ----
full = df[df.config == 'full']
if not full.empty:
    fm = np.median(full[metric])
    print('\n=== graceful degradation (ours vs full-observation) ===')
    for rho in sorted([r for r in df.rho.unique() if r < 1.0], reverse=True):
        o = df[(df.rho == rho) & (df.config == 'ours')]
        if o.empty:
            continue
        deg = (np.median(o[metric]) - fm)/fm*100
        print('  rho=%.1f : %s %.3f vs full %.3f  (%+.1f%%)'
              % (rho, name, np.median(o[metric]), fm, deg))
