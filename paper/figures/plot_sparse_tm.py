#!/usr/bin/env python
"""Figure: complete vs. sparsely observed traffic matrix (real Starlink data).

Reproduces paper figure `figures/sparse-tm.pdf` with three panels:
(a) ground-truth traffic matrix (one epoch, log scale);
(b) flow-level sparse observation  -- a fraction rho of active demand
    pairs is metered, so missing entries are scattered;
(c) node-level sparse observation  -- a fraction rho of source satellites
    meters traffic, so one unmeasured source hides a whole demand row.
Both granularities match `--obs-type {flow,node}` in the evaluation.
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
CROP = 100

tm = np.load(TM_NPZ)['tm'][EPOCH].astype(float)

# order rows/cols by activity so the crop covers busy satellites
order = np.argsort(-(tm.sum(1) + tm.sum(0)))
tm = tm[np.ix_(order, order)][:CROP, :CROP]

active = tm > 0

# flow-level mask: a fraction rho of the active demand pairs is metered,
# so the hidden cells are scattered across the matrix
rng = np.random.default_rng(SEED)
hidden_flow = active & (rng.random(tm.shape) >= OBS_RATIO)

# node-level mask: a fraction rho of source satellites meters traffic, so an
# unmeasured source hides its entire demand row (matches teal_env.py)
observed_nodes = rng.random(CROP) < OBS_RATIO
hidden_node = active & ~observed_nodes[:, None]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
# color semantics: white = no demand / observed zero, dark gray = hidden
# demand, light gray band = source satellite without metering capability
norm = LogNorm(vmin=max(tm[tm > 0].min(), 1.0), vmax=tm.max())
cmap = plt.cm.viridis.copy()
cmap.set_bad(color=(0, 0, 0, 0))        # no demand: transparent, so the
#                                        unmetered-row band stays visible
HIDDEN = (0.42, 0.42, 0.42, 1.0)        # a demand the controller cannot see
UNMETERED_ROW = (0.87, 0.87, 0.87, 1.0)  # whole row is unobservable

for ax, gray_cells, row_band, title in [
        (axes[0], None, None, '(a) Ground-truth traffic matrix'),
        (axes[1], hidden_flow, None,
         '(b) Flow-level observation ($\\rho=0.5$)'),
        (axes[2], hidden_node, ~observed_nodes,
         '(c) Node-level observation ($\\rho=0.5$)')]:
    # background layer: rows whose source satellite performs no metering
    if row_band is not None:
        band = np.zeros((CROP, CROP, 4))
        band[row_band, :] = UNMETERED_ROW
        ax.imshow(band, interpolation='nearest')
    # data layer: only positive demands are colored, zeros are white
    im = ax.imshow(np.ma.masked_less_equal(tm, 0), cmap=cmap, norm=norm,
                   interpolation='nearest')
    # overlay layer: unmetered active demands covered in opaque gray
    if gray_cells is not None:
        overlay = np.zeros((CROP, CROP, 4))
        overlay[gray_cells] = HIDDEN
        ax.imshow(overlay, interpolation='nearest')
    ax.set_xlabel('destination satellite', fontsize=8)
    ax.tick_params(labelsize=7)
    # panel label (a)/(b)/(c) placed *below* the panel, under the x label
    ax.text(0.5, -0.30, title, transform=ax.transAxes,
            ha='center', va='top', fontsize=8.5)
axes[0].set_ylabel('source satellite', fontsize=8)

cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label('demand volume (log scale)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.savefig('sparse-tm.pdf', bbox_inches='tight')
plt.savefig('sparse-tm.png', dpi=200, bbox_inches='tight')
print('saved sparse-tm.pdf / sparse-tm.png')
