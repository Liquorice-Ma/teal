#!/usr/bin/env python
"""Figure: complete vs. sparsely observed traffic matrix (real Starlink data).

Reproduces paper figure `figures/sparse-tm.pdf`:
left  = ground-truth traffic matrix (one epoch, log scale);
right = node-level sparse observation (50% metering satellites),
        unobserved rows shown in gray.
Run from paper/figures/:  python plot_sparse_tm.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

TM_NPZ = '../../code/data/traffic_matrix/starlink_22_72.npz'
EPOCH = 0
OBS_RATIO = 0.5
SEED = 0
# crop to the traffic-carrying sub-block for readability
CROP = 150

tm = np.load(TM_NPZ)['tm'][EPOCH].astype(float)

# order rows/cols by activity so the crop covers busy satellites
order = np.argsort(-(tm.sum(1) + tm.sum(0)))
tm = tm[np.ix_(order, order)][:CROP, :CROP]

# node-level mask: sampled source satellites meter all outgoing demands
rng = np.random.default_rng(SEED)
observed_nodes = rng.random(CROP) < OBS_RATIO

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
# color semantics: white = zero traffic, gray = unobserved, colored = observed
norm = LogNorm(vmin=max(tm[tm > 0].min(), 1.0), vmax=tm.max())
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='white')             # zero traffic in white

for ax, gray_rows, title in [
        (axes[0], None, '(a) Ground-truth traffic matrix'),
        (axes[1], ~observed_nodes,
         '(b) Sparse observation ($\\rho=0.5$, node-level)')]:
    # data layer: only positive demands are colored, zeros are white
    im = ax.imshow(np.ma.masked_less_equal(tm, 0), cmap=cmap, norm=norm,
                   interpolation='nearest')
    # overlay layer: unobserved source rows covered in opaque gray
    if gray_rows is not None:
        overlay = np.zeros((CROP, CROP, 4))
        overlay[gray_rows, :] = (0.72, 0.72, 0.72, 1.0)
        ax.imshow(overlay, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('destination satellite', fontsize=8)
    ax.set_ylabel('source satellite', fontsize=8)
    ax.tick_params(labelsize=7)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label('demand volume (log scale)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.savefig('sparse-tm.pdf', bbox_inches='tight')
plt.savefig('sparse-tm.png', dpi=200, bbox_inches='tight')
print('saved sparse-tm.pdf / sparse-tm.png')
