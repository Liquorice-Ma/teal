#!/usr/bin/env python
"""Audit the experiment matrix collected so far.

Reports, for every (config, rho) cell: per-seed values, median, seed
spread; then PAIRED per-seed comparisons (same seed, same rho) between
ours and each baseline, separately for the with-repair and no-repair
conditions; the untrained control; and seed-noise vs config-signal.

Paired comparison matters here: seed-to-seed variation turned out to be
comparable to the config differences, so median-only reporting can flip
a conclusion.

Usage:  python analyze_final_matrix.py [csv_dir]
"""

import csv
import os
import sys
from collections import defaultdict
from statistics import median

CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
SEEDS = ['0', '1', '2', '3', '4']


def load(fn):
    path = os.path.join(CSV_DIR, fn)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def med(vals):
    return median(vals) if vals else None


def fmt(v):
    return '%8.4f' % v if v is not None else '%8s' % '--'


def dump(title, cells, configs, rhos):
    print('=' * 76)
    print(title)
    print('=' * 76)
    print('%-12s%5s %8s%8s%8s%8s%8s  %8s%8s'
          % ('config', 'rho', 's0', 's1', 's2', 's3', 's4',
             'median', 'spread'))
    for cfg in configs:
        for rho in rhos:
            d = cells.get((cfg, rho))
            if not d:
                continue
            vs = [d.get(s) for s in SEEDS]
            got = [v for v in vs if v is not None]
            print('%-12s%5s ' % (cfg, rho) + ''.join(fmt(v) for v in vs)
                  + '  %8.4f%8.4f' % (med(got), max(got) - min(got)))


def paired(cells_a, key_a, cells_b, key_b, rhos, label):
    """Per-seed paired win/loss; lower MLU wins."""
    tw = tl = 0
    lines = []
    for rho in rhos:
        a = cells_a.get((key_a, rho), {})
        b = cells_b.get((key_b, rho), {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        w = sum(1 for s in shared if a[s] < b[s])
        tw += w
        tl += len(shared) - w
        gap = (med(list(b.values())) - med(list(a.values()))) \
            / med(list(b.values())) * 100
        marks = ' '.join(
            's%s:%s' % (s, 'W' if a[s] < b[s] else 'L') for s in shared)
        lines.append('  rho=%-5s %dW-%dL  median gap %+6.2f%%   %s'
                     % (rho, w, len(shared) - w, gap, marks))
    print('%s   ->  %dW-%dL over %d paired cells' % (label, tw, tl, tw + tl))
    for ln in lines:
        print(ln)
    return tw, tl


# ---------------- load ----------------
fm_cells = defaultdict(dict)      # (config, rho) -> {seed: val}, repair=0 MLU
flow = defaultdict(list)
for r in load('final_matrix.csv'):
    if r['obj'] == 'mlu' and r['repair'] == '0':
        fm_cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])
    elif r['obj'] == 'flow':
        flow[r['config']].append(float(r['mlu']))

ed_cells = defaultdict(dict)      # with ADMM repair (admm-steps 2)
for r in load('eval_denoised.csv'):
    ed_cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])
# repair=1 MLU cells from the final matrix run under the same condition
# (admm-steps 2), so they merge into the same table: M1 mean-gated at
# rho 0.5/0.1 and the M3 zero-retraining configs at rho 0.3
for r in load('final_matrix.csv'):
    if r['obj'] == 'mlu' and r['repair'] == '1':
        ed_cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])

vf_cells = defaultdict(dict)      # untrained controls, with repair
for r in load('verify.csv'):
    if r['config'].startswith('untrained'):
        # covers 'untrained' and the per-topology 'untrained-drop5/10'
        vf_cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])
    else:
        # nbr at rho 0.05/0.02 was run by eval_verify.sh under identical
        # settings to eval_matrix.sh, so it belongs in the same table
        ed_cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])

nr_ours = [float(r['mlu']) for r in load('norepair.csv')
           if r['config'] == 'ours']

# no-repair curve: trained ours at rho 0.3 from norepair.csv, at 0.5/0.1
# from the repair=0 rows of final_matrix.csv; untrained from
# norepair_curve.csv (batch D)
nr_trained = defaultdict(dict)
for s, v in zip(['0', '1', '2'], nr_ours):
    nr_trained['0.3'][s] = v
for r in load('final_matrix.csv'):
    if r['obj'] == 'mlu' and r['repair'] == '0' and r['config'] == 'ours':
        nr_trained[r['rho']][r['seed']] = float(r['mlu'])
nr_untr = defaultdict(dict)
for r in load('norepair_curve.csv'):
    nr_untr[r['rho']][r['seed']] = float(r['mlu'])

MAIN = ['full', 'ours', 'zero-fill', 'no-embed', 'no-gate', 'no-temporal',
        'mean-gated', 'mean-interp', 'nbr']
RHOS = ['1.0', '0.5', '0.3', '0.1', '0.05', '0.02']

# ---------------- A / B: no-repair condition ----------------
dump('A. NO repair (admm-steps 0), MLU -- lower is better', fm_cells,
     MAIN, RHOS)
print()
print('=' * 76)
print('B. PAIRED, NO repair: ours vs each ablation/baseline')
print('=' * 76)
for base in ['zero-fill', 'no-embed', 'no-gate', 'no-temporal']:
    paired(fm_cells, 'ours', fm_cells, base, ['0.5', '0.3', '0.1'],
           'ours vs %-12s' % base)

# ---------------- C / D: with-repair condition ----------------
print()
dump('C. WITH ADMM repair, MLU -- lower is better', ed_cells, MAIN, RHOS)
print()
print('=' * 76)
print('D. PAIRED, WITH repair: ours vs zero-fill (the headline claim)')
print('=' * 76)
paired(ed_cells, 'ours', ed_cells, 'zero-fill',
       ['0.5', '0.3', '0.1', '0.05', '0.02'], 'ours vs zero-fill')

# ---------------- E: untrained controls ----------------
print()
print('=' * 76)
print('E. UNTRAINED CONTROL')
print('=' * 76)
print('E1. WITH repair: untrained vs ours, full rho curve')
for rho in ['0.5', '0.3', '0.1', '0.05', '0.02']:
    u = vf_cells.get(('untrained', rho), {})
    o = ed_cells.get(('ours', rho), {})
    if not (u and o):
        print('  rho=%-5s untrained MISSING' % rho)
        continue
    shared = sorted(set(u) & set(o))
    w = sum(1 for s in shared if o[s] < u[s])
    mu, mo = med(list(u.values())), med(list(o.values()))
    print('  rho=%-5s untrained %.4f | ours %.4f | gap %+6.2f%% | '
          'ours wins %d/%d -> %s'
          % (rho, mu, mo, (mu - mo) / mu * 100, w, len(shared),
             'ours better' if mo < mu else '** UNTRAINED BETTER **'))
print('E2. NO repair: training gain across the rho curve')
for rho in ['0.5', '0.3', '0.1', '0.05', '0.02']:
    t, u = nr_trained.get(rho, {}), nr_untr.get(rho, {})
    if not u:
        print('  rho=%-5s untrained-norepair MISSING' % rho)
        continue
    if not t:
        print('  rho=%-5s untrained %.4f (trained absent -- collapse '
              'region, training never converged here)'
              % (rho, med(list(u.values()))))
        continue
    shared = sorted(set(t) & set(u))
    w = sum(1 for s in shared if t[s] < u[s])
    mt, mu = med(list(t.values())), med(list(u.values()))
    print('  rho=%-5s ours %.4f | untrained %.4f | training gain %+6.1f%% '
          '(paired %d/%d)' % (rho, mt, mu, (mu - mt) / mu * 100,
                              w, len(shared)))

print('E3. Drop topologies: is drop10 easier, or is transfer good?')
for cfg in ['drop5', 'drop10']:
    t = ed_cells.get((cfg, '0.3'), {})
    u = vf_cells.get(('untrained-' + cfg, '0.3'), {})
    if not (t and u):
        print('  %-8s MISSING' % cfg)
        continue
    shared = sorted(set(t) & set(u))
    w = sum(1 for s in shared if t[s] < u[s])
    mt, mu2 = med(list(t.values())), med(list(u.values()))
    print('  %-8s trained %.4f | untrained %.4f | gap %+6.2f%% | '
          'trained wins %d/%d'
          % (cfg, mt, mu2, (mu2 - mt) / mu2 * 100, w, len(shared)))
d5 = vf_cells.get(('untrained-drop5', '0.3'), {})
d10 = vf_cells.get(('untrained-drop10', '0.3'), {})
if d5 and d10:
    print('  untrained baseline drop5 %.4f vs drop10 %.4f -> difference '
          '%.4f' % (med(list(d5.values())), med(list(d10.values())),
                    abs(med(list(d5.values())) - med(list(d10.values())))))
    print('  A near-zero difference means the two perturbed instances are')
    print('  equally hard, so the trained drop5-vs-drop10 gap is noise,')
    print('  not evidence about generalization.')

# ---------------- G: noise vs signal ----------------
print()
print('=' * 76)
print('G. SEED NOISE vs CONFIG SIGNAL')
print('=' * 76)
for name, cells in [('no-repair', fm_cells), ('with-repair', ed_cells)]:
    spreads = [max(d.values()) - min(d.values())
               for d in cells.values() if len(d) >= 2]
    print('  %-12s median within-cell seed spread %.4f  (max %.4f)'
          % (name, med(spreads), max(spreads)))
    for rho in ['0.5', '0.3', '0.1']:
        ms = [med(list(d.values())) for (c, r), d in cells.items()
              if r == rho and c in MAIN]
        if len(ms) > 1:
            print('    rho=%-5s across-config median range %.4f  (n=%d)'
                  % (rho, max(ms) - min(ms), len(ms)))

# ---------------- H: total_flow ceiling ----------------
print()
print('=' * 76)
print('H. total_flow = satisfied-demand ratio -- HIGHER is better')
print('=' * 76)
allv = []
for cfg, vs in sorted(flow.items()):
    allv += vs
    print('  %-12s n=%d  median %.4f  range [%.4f, %.4f]'
          % (cfg, len(vs), med(vs), min(vs), max(vs)))
if allv:
    print('  ALL n=%d  min %.4f  max %.4f  TOTAL SPAN %.4f'
          % (len(allv), min(allv), max(allv), max(allv) - min(allv)))

# ---------------- I: imputation baselines, all with repair ----------------
print()
print('=' * 76)
print('I. IMPUTATION BASELINES ranked per rho (with repair, lower better)')
print('=' * 76)
IMPUT = ['ours', 'nbr', 'mean-gated', 'mean-interp', 'zero-fill']
for rho in ['0.5', '0.3', '0.1', '0.05', '0.02']:
    got = [(c, med(list(ed_cells[(c, rho)].values())))
           for c in IMPUT if ed_cells.get((c, rho))]
    if not got:
        continue
    got.sort(key=lambda t: t[1])
    rank = '  >  '.join('%s %.4f' % (c, v) for c, v in got)
    winner = got[0][0]
    print('rho=%-5s %s' % (rho, rank))
    if winner != 'ours':
        print('        ^^ ours is NOT best at this rho (best: %s)' % winner)

# ---------------- J: M3 zero-retraining vs its reference ----------------
print()
print('=' * 76)
print('J. M3 ZERO-RETRAINING vs same-condition reference (ours rho=0.3)')
print('=' * 76)
ref = ed_cells.get(('ours', '0.3'), {})
if ref:
    rm = med(list(ref.values()))
    print('  reference ours rho=0.3         median %.4f  seeds %s'
          % (rm, sorted(ref.items())))
    for cfg in ['demand-split', 'drop5', 'drop10']:
        d = ed_cells.get((cfg, '0.3'))
        if not d:
            print('  %-14s MISSING' % cfg)
            continue
        vals = list(d.values())
        m = med(vals)
        shared = sorted(set(d) & set(ref))
        w = sum(1 for s in shared if d[s] < ref[s])
        print('  %-14s median %.4f  degradation %+6.2f%%  '
              'beats ref on %d/%d seeds  spread %.4f'
              % (cfg, m, (m - rm) / rm * 100, w, len(shared),
                 max(vals) - min(vals)))
    print('  NOTE: a NEGATIVE degradation means the perturbed setting scored')
    print('        better than the unperturbed reference -- physically')
    print('        impossible, same signature as the full-observation')
    print('        paradox (repair dominates the final MLU).')
