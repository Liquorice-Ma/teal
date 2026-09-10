#!/usr/bin/env python3
"""根据已有数据生成 SpaTE 的重复实验散点图和统计摘要。

主对比读取 ``run/overall_valrepair_5seed.csv``，每行是一次独立重复在
固定 11 个测试快照上的平均值。图中展示各次重复，配对检验仍以相同
观测率、相同随机种子的记录配对，而不是把快照视为独立重复。
组件消融不使用后处理；邻居估计诊断的后处理设置由各自数据源确定。
"""

import csv
import math
import os
from collections import defaultdict
from statistics import mean, median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "figs"))
MAIN_SOURCE = os.path.join(ROOT, "run", "overall_valrepair_5seed.csv")
MAIN_COPY = os.path.join(DATA, "overall_valrepair_5seed.csv")
SUMMARY = os.path.join(DATA, "overall_5seed_summary.csv")
PAIRWISE = os.path.join(DATA, "overall_5seed_pairwise.csv")

RHOS = [0.02, 0.05, 0.1, 0.3, 0.5]
METHODS = ["ours", "test-style", "zero-fill", "mean-interp", "nbr-fill", "untrained"]
LABELS = {
    "ours": "SpaTE", "test-style": "TEST-inspired", "zero-fill": "zero-fill",
    "mean-interp": "mean-interp", "nbr-fill": "nbr-fill", "untrained": "untrained",
    "no-embed": "no embed", "no-gate": "no gate", "no-temporal": "no temporal",
}
COLORS = {
    "ours": "#0072B2", "test-style": "#E69F00", "zero-fill": "#56B4E9",
    "mean-interp": "#009E73", "nbr-fill": "#CC79A7", "untrained": "#7F7F7F",
    "no-embed": "#56B4E9", "no-gate": "#D55E00", "no-temporal": "#009E73",
}
JITTER = [-0.13, -0.065, 0.0, 0.065, 0.13]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 9,
    "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.7, "grid.alpha": 0.3,
    "grid.linewidth": 0.35, "figure.dpi": 220,
})


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def sign_p(wins, losses):
    """Exact two-sided sign-test p value after excluding ties."""
    n = wins + losses
    if not n:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_main():
    rows = read_rows(MAIN_SOURCE)
    cells = defaultdict(dict)
    for row in rows:
        cells[row["config"], float(row["rho"])][int(row["seed"])] = float(row["mlu"])
    for method in METHODS:
        for rho in RHOS:
            assert sorted(cells[method, rho]) == list(range(5)), (method, rho)
    return cells


def export_data(cells):
    os.makedirs(DATA, exist_ok=True)
    with open(MAIN_SOURCE, "r", newline="") as src, open(MAIN_COPY, "w", newline="") as dst:
        dst.write(src.read())
    with open(SUMMARY, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "rho", "mean", "std", "median", "min", "max", "n"])
        for method in METHODS:
            for rho in RHOS:
                vals = [cells[method, rho][seed] for seed in range(5)]
                writer.writerow([method, rho, f"{mean(vals):.4f}", f"{stdev(vals):.4f}",
                                 f"{median(vals):.4f}", f"{min(vals):.4f}",
                                 f"{max(vals):.4f}", len(vals)])
    with open(PAIRWISE, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "wins", "losses", "ties", "p_two_sided"])
        for baseline in METHODS[1:]:
            wins = losses = ties = 0
            for rho in RHOS:
                for seed in range(5):
                    delta = cells["ours", rho][seed] - cells[baseline, rho][seed]
                    if delta < 0:
                        wins += 1
                    elif delta > 0:
                        losses += 1
                    else:
                        ties += 1
            writer.writerow([baseline, wins, losses, ties, f"{sign_p(wins, losses):.4f}"])


def finish(fig, filename, left=0.10, right=0.99, bottom=0.18, top=0.92):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.20)
    fig.savefig(os.path.join(OUT, filename))
    plt.close(fig)


def main_distribution(cells):
    fig, axes = plt.subplots(1, len(RHOS), figsize=(7.15, 2.35), sharey=True)
    for ax, rho in zip(axes, RHOS):
        for idx, method in enumerate(METHODS):
            vals = [cells[method, rho][seed] for seed in range(5)]
            xs = [idx + offset for offset in JITTER]
            ax.scatter(xs, vals, s=13, color=COLORS[method], alpha=0.82, zorder=3)
            ax.hlines(mean(vals), idx - 0.22, idx + 0.22, color="black", lw=1.1, zorder=4)
        ax.set_title(rf"$\rho={rho:g}$")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(["SpaTE", "TEST", "zero", "mean", "nbr", "untr."], rotation=55, ha="right")
        ax.grid(axis="y")
    axes[0].set_ylabel("MLU per repeat\n(11-snapshot test average)")
    # two-line ylabel: needs a wider left margin than the single-line panels
    finish(fig, "overall_distribution.pdf", left=0.105, bottom=0.34)


def internal_training(cells):
    fig, axes = plt.subplots(1, len(RHOS), figsize=(7.15, 2.15), sharey=True)
    for ax, rho in zip(axes, RHOS):
        ours = [cells["ours", rho][seed] for seed in range(5)]
        untrained = [cells["untrained", rho][seed] for seed in range(5)]
        # 横向错开重复点以避免重叠；不画配对连线，统计配对保持不变。
        ax.scatter(JITTER, untrained, color=COLORS["untrained"], s=16, zorder=3)
        ax.scatter([1 + offset for offset in JITTER], ours,
                   color=COLORS["ours"], s=16, zorder=3)
        ax.hlines(mean(untrained), -0.14, 0.14, color="black", lw=1.1)
        ax.hlines(mean(ours), 0.86, 1.14, color="black", lw=1.1)
        ax.set_title(rf"$\rho={rho:g}$")
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["untrained", "SpaTE"], rotation=25, ha="right")
        ax.grid(axis="y")
    axes[0].set_ylabel("MLU with common\nneighbor refinement")
    # two-line ylabel: needs a wider left margin than the single-line panels
    finish(fig, "internal_training.pdf", left=0.105, bottom=0.33)


def component_lowrho():
    ablation = defaultdict(dict)
    for row in read_rows(os.path.join(ROOT, "run", "ablation_lowrho.csv")):
        ablation[row["config"], float(row["rho"])][int(row["seed"])] = float(row["mlu"])
    for row in read_rows(os.path.join(DATA, "norepair_curve.csv")):
        if row["config"] == "ours" and float(row["rho"]) in (0.02, 0.05):
            ablation["ours", float(row["rho"])][int(row["seed"])] = float(row["mlu"])
    methods = ["ours", "no-embed", "no-gate", "no-temporal"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    for ax, rho in zip(axes, [0.02, 0.05]):
        for idx, method in enumerate(methods):
            vals = [ablation[method, rho][seed] for seed in range(3)]
            xs = [idx - 0.07, idx, idx + 0.07]
            ax.scatter(xs, vals, color=COLORS[method], s=17, zorder=3)
            ax.hlines(mean(vals), idx - 0.16, idx + 0.16, color="black", lw=1.0)
        ax.set_title(rf"No post-processing, $\rho={rho:g}$")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["SpaTE", "no\nembed", "no\ngate", "no\ntemporal"])
        ax.grid(axis="y")
    axes[0].set_ylabel("MLU per repeat")
    finish(fig, "component_lowrho.pdf", left=0.10, bottom=0.28)


def neighbor_ladder():
    no_repair = [float(row["mlu"]) for row in read_rows(os.path.join(DATA, "norepair_curve.csv"))
                 if row["config"] == "untrained" and float(row["rho"]) == 0.3]
    repair = read_rows(os.path.join(DATA, "deployable_repair.csv"))
    obs_zero = [float(row["mlu"]) for row in repair if row["config"] == "untrained"
                and row["repair_input"] == "zero" and float(row["rho"]) == 0.3]
    obs_nbr = [float(row["mlu"]) for row in repair if row["config"] == "untrained"
               and row["repair_input"] == "nbr" and float(row["rho"]) == 0.3]
    sweep = read_rows(os.path.join(DATA, "repair_sweep.csv"))
    oracle = [float(row["mlu"]) for row in sweep if row["config"] == "untrained"
              and int(row["admm"]) == 5 and float(row["rho"]) == 0.3]
    groups = [("none", no_repair), ("obs-zero", obs_zero), ("obs-nbr", obs_nbr), ("oracle", oracle)]
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    for idx, (name, vals) in enumerate(groups):
        ax.scatter([idx - 0.07, idx, idx + 0.07], vals, color="#0072B2", s=18, alpha=0.8, zorder=3)
        ax.hlines(median(vals), idx - 0.20, idx + 0.20, color="black", lw=1.2, zorder=4)
    ax.plot(range(4), [median(vals) for _, vals in groups], "--", color="0.45", lw=0.8, zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([name for name, _ in groups])
    ax.set_ylabel("MLU of untrained control")
    ax.set_xlabel(r"Traffic estimate for post-processing ($\rho=0.3$)")
    ax.grid(axis="y")
    finish(fig, "neighbor_ladder.pdf", left=0.18, bottom=0.23)


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = load_main()
    export_data(cells)
    main_distribution(cells)
    internal_training(cells)
    component_lowrho()
    neighbor_ladder()
    print("已生成重复实验图表与 CSV 摘要")


if __name__ == "__main__":
    main()
