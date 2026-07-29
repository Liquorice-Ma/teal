# A Glimpse Is Enough: SiTE — Mask-Aware Sparse-Input Traffic Engineering for LEO Satellite Constellations

Learning-based traffic engineering (TE) for large-scale LEO satellite
constellations that operates directly on **sparse traffic observations**:
only a fraction (10%–70%) of demands is measured at each control epoch, yet
the system allocates traffic for all active demands at millisecond latency.

Built on top of [Teal (SIGCOMM '23)](https://github.com/harvard-cns/teal),
extended with:

- **Learnable mask embeddings** — unobserved demands are represented by
  trainable embeddings instead of zeros, so the model distinguishes
  "not measured" from "no traffic";
- **Mask-aware gated GNN** — unobserved path nodes receive link-state
  messages but do not inject placeholder volumes into link embeddings;
- **Temporal-spatial fusion** — a Transformer encodes the history of sparse
  observations and fuses it with GNN embeddings via per-demand
  cross-attention, recovering missing information from temporal correlation;
- **Demand pruning & zero-retraining deployment** — only demand pairs with
  observed traffic are modeled (2.5M → ~6K pairs on a 22×72 constellation);
  all weights are shared across demands and size-invariant, so a model
  trained once is reused as the topology and demand set drift, without
  retraining.

## Getting started

See [docs/operation_guide.md](docs/operation_guide.md) for full setup
(Chinese). In short, on a Linux + NVIDIA GPU machine:

```bash
conda create -n teal python=3.10 -y && conda activate teal
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-<ver>+cu121.html
pip install -r requirements.txt && pip install pandas
```

Prepare the Starlink 22×72 topology and traffic matrices (one-time; the
`code/` directory ships the raw satellite data):

```bash
cd run
python prepare_starlink.py --size-x 22 --size-y 72 --capacity 1000
```

## Running

```bash
cd run
python teal.py --obj total_flow --topo Starlink2272.json --tm-model starlink \
  --epochs 100 --admm-steps 2 --prune-demands \
  --obs-ratio 0.5 --hist-len 3 --seed 0 \
  --slice-train-start 0 --slice-train-stop 80 \
  --slice-val-start 80 --slice-val-stop 90 \
  --slice-test-start 90 --slice-test-stop 101
```

Key options added on top of Teal:

| Option | Description |
|---|---|
| `--obs-ratio` | fraction of observed demands (1.0 = original Teal behavior) |
| `--obs-type {flow,node}` | sampling granularity: demand pairs, or source nodes metering all outgoing demands |
| `--hist-len` | history window length; 1 disables the temporal module |
| `--mask-mode {embed,zero,mean}` | fill unobserved demands: learnable embedding (ours), zeros, or mean interpolation (two-stage baseline) |
| `--no-gate` | disable mask-aware gating (ablation) |
| `--prune-demands` | model only demand pairs with nonzero traffic |
| `--demand-split` | train on the training-slice demand set, rebuild from the test slice at test time with frozen weights (zero-retraining generalization) |
| `--seed` | global random seed (fully reproducible) |

Results append to `run/teal-total_flow-all.csv`; models are saved under
`run/teal-models/` with configuration-encoded filenames.

## Repository structure

```
├── lib/                  # source: env, actor, FlowGNN, TemporalEncoder, ADMM
├── run/                  # entry point (teal.py), data prep (prepare_starlink.py)
├── code/                 # raw satellite topology & traffic data (22x72 Starlink)
├── topologies/           # converted topologies + path caches
├── traffic-matrices/     # per-epoch traffic matrices (regenerated, not in git)
├── docs/                 # paper drafts (intro/related work/formulation/design/
│                         #   evaluation/conclusion) and the operation guide
└── pop-ncflow-lptop/     # LP-based baselines from the original Teal repo
```

## Acknowledgements & citation

This repository extends [Teal](https://github.com/harvard-cns/teal) by
Xu et al. If you use the Teal backbone, please cite:

```bibtex
@inproceedings{teal,
    title={Teal: Learning-Accelerated Optimization of WAN Traffic Engineering},
    author={Xu, Zhiying and Yan, Francis Y. and Singh, Rachee and Chiu, Justin T. and Rush, Alexander M. and Yu, Minlan},
    booktitle={Proceedings of the ACM SIGCOMM 2023 Conference},
    pages={378--393},
    year={2023}
}
```

A citation entry for this work will be added upon publication.
