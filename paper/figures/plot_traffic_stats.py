#!/usr/bin/env python
"""Traffic characterization figures from REAL Starlink trace data.

Outputs:
  traffic-dynamics.pdf : per-satellite egress volume over 101 snapshots
                         (temporal variability, motivates temporal module)
  geo-heatmap.pdf      : total egress per satellite on the 72x22 grid
                         (geographic concentration, motivates pruning)
  prints TM statistics for tab:tm-stats
Run from paper/figures/:  python plot_traffic_stats.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({'font.size': 9, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linestyle': '--'})

TM_NPZ = '../../code/data/traffic_matrix/starlink_22_72.npz'
tms = np.load(TM_NPZ)['tm'].astype(float)          # [101, 1584, 1584]
T, N, _ = tms.shape

# ============ Fig: temporal dynamics of per-satellite egress ============
egress = tms.sum(axis=2)                            # [T, N]
busiest = np.argsort(-egress.mean(axis=0))[:3]      # 3 busiest satellites
median_sat = np.argsort(-egress.mean(axis=0))[N // 8]

fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
colors = ['#59A14F', '#F18F01', '#457B9D']
for c, s in zip(colors, busiest):
    ax.plot(range(T), egress[:, s], lw=1.2, color=c,
            label=f'satellite #{s}')
ax.plot(range(T), egress[:, median_sat], lw=1.2, color='#9AA0A6',
        ls='--', label=f'satellite #{median_sat}')
ax.set_yscale('log')
ax.set_xlabel('snapshot index (5-min interval)')
ax.set_ylabel('egress volume (Mbps, log)')
ax.legend(fontsize=6.5, loc='lower right', ncol=2)
fig.savefig('traffic-dynamics.pdf', bbox_inches='tight')
fig.savefig('traffic-dynamics.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ============ Fig: geographic concentration heatmap (orbit grid) ============
# node id -> (plane, phase); data uses 72 planes x 22 satellites per plane
PLANES, PER_PLANE = 72, 22
total_egress = egress.sum(axis=0)                   # [N]
grid = total_egress.reshape(PLANES, PER_PLANE).T    # [22, 72]

fig, ax = plt.subplots(figsize=(3.6, 2.2), constrained_layout=True)
im = ax.imshow(grid + 1, cmap='inferno', aspect='auto',
               norm=LogNorm(vmin=1, vmax=grid.max()))
ax.set_xlabel('orbital plane index')
ax.set_ylabel('in-plane phase index')
ax.grid(False)
cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
cbar.set_label('total egress (Mbps, log)', fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.savefig('geo-heatmap.pdf', bbox_inches='tight')
fig.savefig('geo-heatmap.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ============ Table: TM statistics ============
nz = tms[tms > 0]
union = (tms > 0).any(axis=0)
np.fill_diagonal(union, False)
per_snap_pairs = (tms > 0).reshape(T, -1).sum(axis=1)
print('=== tab:tm-stats (population-based random city sampling) ===')
print(f'snapshots               : {T}')
print(f'nonzero demand (Mbps)   : min {nz.min():.1f} / max {nz.max():.0f}'
      f' / mean {nz.mean():.2f} / std {nz.std():.2f}')
print(f'total demand per snap   : mean {tms.sum(axis=(1,2)).mean():.0f} Mbps')
print(f'nonzero ratio           : {np.count_nonzero(tms)/tms.size:.4f}')
print(f'active pairs per snap   : mean {per_snap_pairs.mean():.0f}')
print(f'active pair union       : {union.sum()} '
      f'({union.sum()/(N*(N-1))*100:.2f}% of all pairs)')
