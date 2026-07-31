#!/usr/bin/env python
"""Summarize the multi-seed sparse-robustness check.

Reads claim_seeds.csv (method,rho,seed,mlu) and the LP oracle csv, then
reports mean +/- std of the realized MLU and the performance ratio
PR = MLU / MLU_opt per method and observability level. Also flags
whether the gap between our method and each baseline exceeds the seed
noise, which is the question the check is meant to answer.

Usage (in run/):  python summarize_claim.py
"""

import os

import pandas as pd

CSV = 'claim_seeds.csv'
ORACLE = 'lp-oracle-Starlink2272.json.csv'

df = pd.read_csv(CSV)
u_opt = pd.read_csv(ORACLE)['opt_mlu'].mean() if os.path.exists(ORACLE) else None

print('runs: %d (seeds present: %s)'
      % (len(df), sorted(df.seed.unique())))
if u_opt:
    print('LP oracle MLU = %.4f\n' % u_opt)

agg = df.groupby(['rho', 'method'])['mlu'].agg(['mean', 'std', 'count'])
agg = agg.reset_index()
if u_opt:
    agg['PR'] = agg['mean']/u_opt

print('%-6s %-12s %-8s %-8s %-6s %s' %
      ('rho', 'method', 'mean', 'std', 'n', 'PR'))
for _, r in agg.sort_values(['rho', 'method'], ascending=[False, True]).iterrows():
    print('%-6s %-12s %-8.4f %-8.4f %-6d %.3f' %
          (r['rho'], r['method'], r['mean'], r['std'] if r['std'] == r['std']
           else 0, r['count'], r.get('PR', float('nan'))))

print('\n--- is the gap larger than the noise? (lower MLU is better) ---')
for rho in sorted(df.rho.unique(), reverse=True):
    if rho >= 1.0:
        continue
    sub = agg[agg.rho == rho].set_index('method')
    if 'ours' not in sub.index:
        continue
    ours, ours_sd = sub.loc['ours', 'mean'], sub.loc['ours', 'std']
    for base in ['zero-fill', 'mean-interp']:
        if base not in sub.index:
            continue
        b, b_sd = sub.loc[base, 'mean'], sub.loc[base, 'std']
        gap = (b - ours)/b*100
        pooled = ((ours_sd or 0)**2 + (b_sd or 0)**2)**0.5
        verdict = 'significant' if abs(b - ours) > pooled else 'WITHIN NOISE'
        print('rho=%-5s ours %.4f vs %-12s %.4f | gain %+.1f%% | '
              'pooled std %.4f -> %s'
              % (rho, ours, base, b, gap, pooled, verdict))
