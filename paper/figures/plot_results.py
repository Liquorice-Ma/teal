#!/usr/bin/env python
"""Generate PLACEHOLDER experiment figures to preview paper layout.

WARNING: all numbers here are mock/placeholder values chosen to match the
expected narrative trend. Replace with real results from GPU experiments
before submission. Run from paper/figures/:  python plot_results.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linestyle': '--'})
BLUE, ORANGE, GREEN, RED, GRAY = \
    '#457B9D', '#F18F01', '#59A14F', '#E63946', '#9AA0A6'

# ============ Fig A: satisfied ratio vs observability (RQ1) ============
rho = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
upper = 0.965                       # Teal-full upper reference (rho=1.0)
ours = np.array([0.965, 0.958, 0.951, 0.938, 0.905])
zero = np.array([0.965, 0.902, 0.861, 0.788, 0.642])
mi   = np.array([0.965, 0.918, 0.884, 0.826, 0.715])

fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.axhline(upper, color=GRAY, ls=':', lw=1.2, label='Teal-full (oracle)')
ax.plot(rho, ours, 'o-', color=GREEN, lw=1.8, label='Ours')
ax.plot(rho, mi,   's--', color=ORANGE, lw=1.5, label='Mean-interp.')
ax.plot(rho, zero, '^--', color=BLUE, lw=1.5, label='Zero-fill')
ax.set_xlabel('observability ratio $\\rho$')
ax.set_ylabel('satisfied demand ratio')
ax.set_xticks(rho); ax.invert_xaxis()
ax.legend(fontsize=7, loc='lower left')
fig.savefig('exp-overall.pdf', bbox_inches='tight')
fig.savefig('exp-overall.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ============ Fig B: ablation bars (RQ2, rho=0.3) ============
labels = ['Full', '$-$temporal', '$-$gate', '$-$embed', 'none\n(zero-fill)']
vals = np.array([0.938, 0.902, 0.889, 0.831, 0.788])
errs = np.array([0.004, 0.006, 0.007, 0.010, 0.012])
colors = [GREEN, ORANGE, ORANGE, ORANGE, BLUE]

fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.bar(range(len(vals)), vals, yerr=errs, capsize=3,
       color=colors, edgecolor='black', lw=0.6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel('satisfied demand ratio')
ax.set_ylim(0.70, 0.97)
ax.set_title('Ablation at $\\rho=0.3$', fontsize=9)
fig.savefig('exp-ablation.pdf', bbox_inches='tight')
fig.savefig('exp-ablation.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ============ Fig C: training curves (temporal stability) ============
ep = np.arange(0, 101)
rng = np.random.default_rng(0)
def curve(final, noise):
    base = final * (1 - np.exp(-ep / 18))
    return np.clip(base + rng.normal(0, noise, ep.size) * np.exp(-ep/40), 0, 1)
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.plot(ep, curve(0.951, 0.03), color=GREEN, lw=1.5, label='Ours (hist=3)')
ax.plot(ep, curve(0.902, 0.025), color=BLUE, lw=1.5, label='hist=1')
ax.set_xlabel('training epoch')
ax.set_ylabel('validation satisfied ratio')
ax.legend(fontsize=7, loc='lower right')
ax.set_title('Training stability', fontsize=9)
fig.savefig('exp-training.pdf', bbox_inches='tight')
fig.savefig('exp-training.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print('saved exp-overall / exp-ablation / exp-training (.pdf/.png) '
      '[PLACEHOLDER DATA]')

# ============ Fig D: flow vs node sampling (RQ5a) ============
ours_node = np.array([0.965, 0.949, 0.936, 0.914, 0.868])
zero_node = np.array([0.965, 0.871, 0.815, 0.722, 0.571])
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.plot(rho, ours, 'o-', color=GREEN, lw=1.8, label='Ours (flow-level)')
ax.plot(rho, ours_node, 'o--', color=GREEN, lw=1.4, alpha=0.65,
        label='Ours (node-level)')
ax.plot(rho, zero, '^-', color=BLUE, lw=1.5, label='Zero-fill (flow)')
ax.plot(rho, zero_node, '^--', color=BLUE, lw=1.2, alpha=0.65,
        label='Zero-fill (node)')
ax.set_xlabel('observability ratio $\\rho$')
ax.set_ylabel('satisfied demand ratio')
ax.set_xticks(rho); ax.invert_xaxis()
ax.legend(fontsize=6.5, loc='lower left')
ax.set_title('Measurement granularity', fontsize=9)
fig.savefig('exp-obstype.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Fig E: ablation across rho (RQ2 extended) ============
full   = ours
no_tmp = np.array([0.965, 0.941, 0.921, 0.902, 0.833])
no_gat = np.array([0.965, 0.936, 0.912, 0.889, 0.815])
no_emb = np.array([0.965, 0.921, 0.884, 0.831, 0.729])
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.plot(rho, full,   'o-', color=GREEN,  lw=1.8, label='Full')
ax.plot(rho, no_tmp, 's--', color=ORANGE, lw=1.4, label='$-$temporal')
ax.plot(rho, no_gat, 'd--', color=BLUE,   lw=1.4, label='$-$gate')
ax.plot(rho, no_emb, '^--', color=RED,    lw=1.4, label='$-$embed')
ax.set_xlabel('observability ratio $\\rho$')
ax.set_ylabel('satisfied demand ratio')
ax.set_xticks(rho); ax.invert_xaxis()
ax.legend(fontsize=7, loc='lower left')
ax.set_title('Ablation across $\\rho$', fontsize=9)
fig.savefig('exp-ablation-rho.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Fig F: zero-retraining per-snapshot (RQ3) ============
snap = np.arange(90, 101)
oracle_ts = 0.951 + rng.normal(0, 0.006, snap.size)
split_ts = oracle_ts - np.abs(rng.normal(0.004, 0.003, snap.size))
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.plot(snap, oracle_ts, 'o-', color=GRAY, lw=1.5, label='Oracle demand set')
ax.plot(snap, split_ts, 's-', color=GREEN, lw=1.5,
        label='Zero-retraining (rebuilt set)')
ax.set_xlabel('test snapshot index')
ax.set_ylabel('satisfied demand ratio')
ax.set_ylim(0.90, 0.98)
ax.legend(fontsize=7, loc='lower left')
ax.set_title('Generalization under demand-set drift', fontsize=9)
fig.savefig('exp-split-time.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Fig G: inference latency (RQ4, log scale) ============
methods = ['LP solver', 'Teal-full', 'Ours\n(hist=1)', 'Ours\n(hist=3)']
latency = np.array([46000, 12, 6, 9])          # ms, placeholder
colors_g = [GRAY, BLUE, GREEN, GREEN]
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
bars = ax.bar(range(4), latency, color=colors_g, edgecolor='black', lw=0.6)
ax.set_yscale('log')
ax.set_ylabel('inference latency (ms, log)')
ax.set_xticks(range(4)); ax.set_xticklabels(methods, fontsize=7)
for b, v in zip(bars, latency):
    ax.text(b.get_x()+b.get_width()/2, v*1.3,
            f'{v:,.0f}', ha='center', fontsize=7)
ax.set_title('Per-snapshot latency', fontsize=9)
fig.savefig('exp-runtime.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Fig H: history length sweep (RQ5b) ============
hist = np.array([1, 3, 5, 8])
obj_hist = np.array([0.902, 0.938, 0.944, 0.945])
lat_hist = np.array([6.0, 9.0, 11.5, 15.0])
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.plot(hist, obj_hist, 'o-', color=GREEN, lw=1.8)
ax.set_xlabel('history length $L$')
ax.set_ylabel('satisfied demand ratio', color=GREEN)
ax.tick_params(axis='y', labelcolor=GREEN)
ax2 = ax.twinx()
ax2.plot(hist, lat_hist, 's--', color=GRAY, lw=1.4)
ax2.set_ylabel('latency (ms)', color=GRAY)
ax2.tick_params(axis='y', labelcolor=GRAY)
ax2.grid(False)
ax.set_xticks(hist)
ax.set_title('History length ($\\rho=0.3$)', fontsize=9)
fig.savefig('exp-histlen.pdf', bbox_inches='tight')
plt.close(fig)

# ============ Fig I: capacity load sensitivity (RQ5c) ============
cap = ['500\n(high load)', '1000\n(medium)', '2000\n(low load)']
ours_c = np.array([0.842, 0.938, 0.981])
zero_c = np.array([0.633, 0.788, 0.902])
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
ax.bar(x-w/2, ours_c, w, color=GREEN, edgecolor='black', lw=0.6,
       label='Ours')
ax.bar(x+w/2, zero_c, w, color=BLUE, edgecolor='black', lw=0.6,
       label='Zero-fill')
ax.set_xticks(x); ax.set_xticklabels(cap, fontsize=7)
ax.set_xlabel('ISL capacity')
ax.set_ylabel('satisfied demand ratio')
ax.set_ylim(0.5, 1.02)
ax.legend(fontsize=7)
ax.set_title('Load sensitivity ($\\rho=0.3$)', fontsize=9)
fig.savefig('exp-capacity.pdf', bbox_inches='tight')
plt.close(fig)

print('saved 6 additional figures: obstype / ablation-rho / split-time / '
      'runtime / histlen / capacity [PLACEHOLDER DATA]')

# ============ Fig J: CDF of per-snapshot satisfied ratio (RQ1) ============
nsnap = 11 * 3          # test snapshots x seeds
samples = {
    'Ours':         np.clip(rng.normal(0.938, 0.012, nsnap), 0, 1),
    'Mean-interp.': np.clip(rng.normal(0.826, 0.030, nsnap), 0, 1),
    'Zero-fill':    np.clip(rng.normal(0.788, 0.038, nsnap), 0, 1),
    'Teal-full':    np.clip(rng.normal(0.965, 0.008, nsnap), 0, 1),
}
styles = {'Ours': (GREEN, '-'), 'Mean-interp.': (ORANGE, '--'),
          'Zero-fill': (BLUE, '--'), 'Teal-full': (GRAY, ':')}
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
for name, vals in samples.items():
    xs = np.sort(vals)
    ys = np.arange(1, xs.size + 1) / xs.size
    color, ls = styles[name]
    ax.plot(xs, ys, ls, color=color, lw=1.6, label=name)
ax.set_xlabel('satisfied demand ratio')
ax.set_ylabel('CDF')
ax.legend(fontsize=7, loc='upper left')
ax.set_title('Per-snapshot CDF ($\\rho=0.3$)', fontsize=9)
fig.savefig('exp-cdf.pdf', bbox_inches='tight')
plt.close(fig)
print('saved exp-cdf [PLACEHOLDER DATA]')
