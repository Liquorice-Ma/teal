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
