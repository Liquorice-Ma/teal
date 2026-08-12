#!/usr/bin/env python
"""Significance and stability checks on the with-repair comparison.

Two questions the median table cannot answer:
  1. Is the ours-vs-zero-fill win pattern beyond chance? Paired sign test
     is used because pairing on (rho, seed) removes the seed effect,
     which turned out to dominate the between-config differences.
  2. Does the advantage show up as a lower spread / worst case rather
     than a lower median? Reported per rho.

Usage:  python check_significance.py [csv_dir]
"""

import csv
import os
import sys
from collections import defaultdict
from math import comb
from statistics import median

CSV_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'


def sign_test_one_sided(wins, n):
    """P(X >= wins) under X ~ Binom(n, 0.5)."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


cells = defaultdict(dict)
with open(os.path.join(CSV_DIR, 'eval_denoised.csv')) as f:
    for r in csv.DictReader(f):
        cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])

untr = defaultdict(dict)
with open(os.path.join(CSV_DIR, 'verify.csv')) as f:
    for r in csv.DictReader(f):
        if r['config'].startswith('untrained'):
            untr[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])
        else:
            # nbr at rho 0.05/0.02 ran under identical settings, so it
            # belongs in the same repaired table as eval_denoised.csv
            cells[(r['config'], r['rho'])][r['seed']] = float(r['mlu'])

RHOS = ['0.5', '0.3', '0.1', '0.05', '0.02']

print('=' * 72)
print('1. PAIRED SIGN TEST, with repair: ours vs zero-fill / vs nbr')
print('=' * 72)
for rival in ('zero-fill', 'nbr'):
    w = n = 0
    for rho in RHOS:
        a, b = cells.get(('ours', rho), {}), cells.get((rival, rho), {})
        for s in sorted(set(a) & set(b)):
            n += 1
            w += a[s] < b[s]
    if n == 0:
        print('  ours vs %-10s no paired cells' % rival)
        continue
    print('  ours vs %-10s wins %d / %d   one-sided p = %.4f   '
          'two-sided p = %.4f'
          % (rival, w, n, sign_test_one_sided(w, n),
             min(1.0, 2 * sign_test_one_sided(w, n))))
    # per-rho breakdown, since the overall test pools different rho
    for rho in RHOS:
        a, b = cells.get(('ours', rho), {}), cells.get((rival, rho), {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        ww = sum(1 for s in shared if a[s] < b[s])
        print('      rho=%-5s %d/%d' % (rho, ww, len(shared)))

print()
print('=' * 72)
print('2. STABILITY: spread and worst case, ours vs zero-fill (with repair)')
print('=' * 72)
print('%5s  %-28s %-28s' % ('rho', 'ours', 'zero-fill'))
for rho in RHOS:
    a = list(cells[('ours', rho)].values())
    b = list(cells[('zero-fill', rho)].values())
    if not (a and b):
        continue
    sa, sb = max(a) - min(a), max(b) - min(b)
    wa, wb = max(a), max(b)
    flag = ''
    if sa < sb:
        flag += ' spread-win'
    if wa < wb:
        flag += ' worstcase-win %+.1f%%' % ((wb - wa) / wb * 100)
    print('%5s  med %.4f sprd %.4f worst %.4f  med %.4f sprd %.4f '
          'worst %.4f %s' % (rho, median(a), sa, wa, median(b), sb, wb, flag))

print()
print('=' * 72)
print('3. FULL-OBSERVATION SANITY: does less information hurt at all?')
print('=' * 72)
fu = list(cells[('full', '1.0')].values())
print('  full  rho=1.0  median %.4f' % median(fu))
for rho in RHOS:
    a = list(cells[('ours', rho)].values())
    if a:
        print('  ours  rho=%-5s median %.4f   %s'
              % (rho, median(a),
                 'WORSE than full (expected)' if median(a) > median(fu)
                 else '** BETTER than full -- impossible, repair dominates **'))

print()
print('=' * 72)
print('4. UNTRAINED vs OURS, with repair (coverage of the control)')
print('=' * 72)
for rho in ['0.5', '0.3', '0.1', '0.05', '0.02']:
    u = untr.get(('untrained', rho))
    o = cells.get(('ours', rho))
    if not u:
        print('  rho=%-5s untrained MISSING  <- needs to be run' % rho)
        continue
    shared = sorted(set(u) & set(o))
    w = sum(1 for s in shared if o[s] < u[s])
    print('  rho=%-5s untrained %.4f | ours %.4f | ours wins %d/%d'
          % (rho, median(list(u.values())), median(list(o.values())),
             w, len(shared)))
