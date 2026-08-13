#!/usr/bin/env python3
"""Regenerate the three mechanism figures (teaser / rq1 / rq2) for the paper.

Data provenance (CSVs mirrored from the experiment server, run/ directory):
  - eval_denoised.csv : trained ("ours") WITH repair, 5 rho x 5 seeds  -> RQ1
  - verify.csv        : untrained WITH repair, 5 rho x 5 seeds         -> RQ1
  - norepair_curve.csv: ours (rho 0.02/0.05) + untrained (all rho), no repair
  - final_matrix.csv  : ours no-repair at rho 0.1/0.5 (obj=mlu rows only)
  - norepair.csv      : ours no-repair at rho 0.3
  (the repair-off trained curve is assembled from three historical batches)

The script asserts the plotted medians against the numbers quoted in
05-evaluation.tex (tab:rq1-strat and the RQ2 narrative) so the figures
cannot silently drift from the text.

X axis: true log spacing over rho in {0.02, 0.05, 0.1, 0.3, 0.5}, but with a
FixedLocator at exactly those five values and minor ticks disabled -- the
default LogFormatter labels every minor tick (2e-2, 3e-2, 4e-2, ...) which
overlaps into an unreadable band.
"""

import csv
import os
from statistics import median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.normpath(os.path.join(HERE, "..", "figs"))

RHOS = [0.02, 0.05, 0.1, 0.3, 0.5]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.dpi": 200,
})

C_TRAIN = "#0072B2"   # blue
C_UNTRAIN = "#D55E00" # vermillion
C_ON = "#CC0000"      # repair-on red (matches old teaser)
C_OFF = "#0072B2"
C_2224 = "#009E73"   # green for cross-scale


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def by_cell(rows, config):
    """{(rho, seed): mlu} for one config."""
    return {(float(r["rho"]), int(r["seed"])): float(r["mlu"])
            for r in rows if r["config"] == config}


def paired_gains(trained, untrained, seeds):
    """Median per-rho paired gain (%): (untrained - trained)/untrained."""
    out = []
    for rho in RHOS:
        gains = [(untrained[(rho, s)] - trained[(rho, s)])
                 / untrained[(rho, s)] * 100
                 for s in seeds if (rho, s) in trained and (rho, s) in untrained]
        assert gains, f"no paired seeds at rho={rho}"
        out.append(median(gains))
    return out


def medians(cell, seeds):
    return [median(cell[(rho, s)] for s in seeds if (rho, s) in cell)
            for rho in RHOS]


def series(cell, rho, seeds):
    """All MLU values for one rho across seeds (sorted)."""
    return sorted(cell[(rho, s)] for s in seeds if (rho, s) in cell)


def style_ax(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(RHOS))
    ax.xaxis.set_minor_locator(NullLocator())  # the actual fix
    ax.set_xticklabels([f"{r:g}" for r in RHOS])
    ax.set_xlim(0.017, 0.62)
    ax.grid(True, which="major")
    ax.set_xlabel(r"Observability $\rho$")


def close(fig, name):
    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.16, right=0.97)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    denoised = load("eval_denoised.csv")      # ours, repair ON
    verify = load("verify.csv")               # untrained, repair ON
    norepair = load("norepair_curve.csv")     # repair OFF (partial)
    final = load("final_matrix.csv")          # ours repair=0 at rho 0.1/0.5
    norepair03 = load("norepair.csv")         # ours repair=0 at rho 0.3

    on_train = by_cell(denoised, "ours")
    on_untrain = by_cell(verify, "untrained")
    off_untrain = by_cell(norepair, "untrained")

    # repair-off trained curve from three historical batches
    off_train = by_cell(norepair, "ours")     # rho 0.02, 0.05
    for r in final:                           # rho 0.1, 0.5
        if r["config"] == "ours" and r["repair"] == "0" and r["obj"] == "mlu":
            off_train[(float(r["rho"]), int(r["seed"]))] = float(r["mlu"])
    for r in norepair03:                      # rho 0.3
        if r["config"] == "ours":
            off_train[(0.3, int(r["seed"]))] = float(r["mlu"])

    seeds5 = range(5)
    seeds3 = range(3)

    # ---- consistency with the paper's tables (fail loudly on drift) ----
    exp_train = [1.665, 1.625, 1.763, 1.614, 1.587]   # tab:rq1-strat
    exp_untrain = [1.600, 1.634, 1.564, 1.568, 1.543]
    exp_gain_on = [4.8, 0.0, -1.3, -5.4, -3.3]
    exp_gain_off = [16.5, 16.4, 9.8, 28.1, 12.0]      # RQ2 narrative

    med_train = medians(on_train, seeds5)
    med_untrain = medians(on_untrain, seeds5)
    gain_on = paired_gains(on_train, on_untrain, seeds5)
    gain_off = paired_gains(off_train, off_untrain, seeds3)

    for got, exp, tol, tag in [
        (med_train, exp_train, 0.01, "rq1 trained median"),
        (med_untrain, exp_untrain, 0.01, "rq1 untrained median"),
        (gain_on, exp_gain_on, 0.6, "repair-on gain"),
        (gain_off, exp_gain_off, 0.6, "repair-off gain"),
    ]:
        for rho, g, e in zip(RHOS, got, exp):
            assert abs(g - e) <= tol, f"{tag} drifted at rho={rho}: {g:.3f} vs {e}"

    # ---- teaser: paired training gain, repair on vs off ----
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.plot(RHOS, gain_off, "o-", color=C_OFF, label="Repair off (learning visible)")
    ax.plot(RHOS, gain_on, "s--", color=C_ON, label="Repair on (absorbed)")
    style_ax(ax)
    ax.set_ylabel("Paired training gain (%)")
    ax.set_ylim(-9, 33)
    ax.legend(loc="upper left", framealpha=0.9)
    close(fig, "teaser.pdf")

    # ---- rq1: median MLU with repair on, trained vs untrained ----
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(RHOS, med_train, "o-", color=C_TRAIN, label="Trained")
    ax.plot(RHOS, med_untrain, "s--", color=C_UNTRAIN, label="Untrained")
    style_ax(ax)
    ax.set_ylabel("Median MLU (repair on)")
    ax.set_ylim(1.49, 1.88)
    ax.legend(loc="upper right", framealpha=0.9)
    close(fig, "rq1.pdf")

    # ---- rq2: repair-off per-seed gains + median ----
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    for rho in RHOS:
        ys = [(off_untrain[(rho, s)] - off_train[(rho, s)])
              / off_untrain[(rho, s)] * 100 for s in seeds3]
        ax.plot([rho] * len(ys), ys, "o", color=C_OFF, alpha=0.45,
                markersize=3.5, zorder=2)
    ax.plot(RHOS, gain_off, "o-", color=C_OFF, label="Median paired gain",
            zorder=3)
    style_ax(ax)
    ax.set_ylabel("Paired training gain (%)")
    ax.set_ylim(-6, 44)
    ax.legend(loc="upper left", framealpha=0.9)
    close(fig, "rq2.pdf")

    # ---- fig: repair-budget sweep (switch not dial) ----
    # budgets 0 (norepair), 1, 5 (repair_sweep); 2-sweep plateau is in
    # tab:saturation and omitted from the figure to keep seed counts uniform.
    sweep = load("repair_sweep.csv")
    sw_ours = {(int(r["admm"]), int(r["seed"])): float(r["mlu"])
               for r in sweep if r["config"] == "ours"}
    sw_untrain = {(int(r["admm"]), int(r["seed"])): float(r["mlu"])
                  for r in sweep if r["config"] == "untrained"}
    # budget 0 from norepair (rho=0.3)
    zero_ours = [off_train[(0.3, s)] for s in seeds3]
    zero_untr = [off_untrain[(0.3, s)] for s in seeds3]

    budgets = [0, 1, 5]
    tr_med = [median(zero_ours),
              median([sw_ours[(1, s)] for s in seeds3]),
              median([sw_ours[(5, s)] for s in seeds3])]
    un_med = [median(zero_untr),
              median([sw_untrain[(1, s)] for s in seeds3]),
              median([sw_untrain[(5, s)] for s in seeds3])]
    # assert against tab:saturation (3-seed)
    for got, exp in [(tr_med, [2.184, 1.655, 1.649]),
                     (un_med, [2.833, 1.597, 1.599])]:
        for b, g, e in zip(budgets, got, exp):
            assert abs(g - e) <= 0.01, f"saturation b={b}: {g:.3f} vs {e}"

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    x = range(len(budgets))
    ax.plot(x, un_med, "s--", color=C_UNTRAIN, label="Untrained")
    ax.plot(x, tr_med, "o-", color=C_TRAIN, label="Trained")
    # seed scatter
    for i, vals in enumerate([zero_untr, [sw_untrain[(1, s)] for s in seeds3],
                               [sw_untrain[(5, s)] for s in seeds3]]):
        ax.plot([i] * len(vals), vals, "s", color=C_UNTRAIN, alpha=0.3, ms=3)
    for i, vals in enumerate([zero_ours, [sw_ours[(1, s)] for s in seeds3],
                               [sw_ours[(5, s)] for s in seeds3]]):
        ax.plot([i] * len(vals), vals, "o", color=C_TRAIN, alpha=0.3, ms=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Repair budget (sweeps of ten)")
    ax.set_ylabel("Median MLU")
    ax.set_ylim(1.45, 3.05)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    close(fig, "saturation.pdf")

    # ---- fig: RQ3 stability (worst-case + spread, repair on) ----
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for rho, offst in zip(RHOS, [-0.015, 0.015]):
        tr_vals = series(on_train, rho, seeds5)
        un_vals = series(on_untrain, rho, seeds5)
        ax.errorbar([rho + offst], [median(tr_vals)],
                    yerr=[[median(tr_vals) - min(tr_vals)],
                          [max(tr_vals) - median(tr_vals)]],
                    fmt="o-", color=C_TRAIN, capsize=2, ms=4)
        ax.errorbar([rho + offst], [median(un_vals)],
                    yerr=[[median(un_vals) - min(un_vals)],
                          [max(un_vals) - median(un_vals)]],
                    fmt="s--", color=C_UNTRAIN, capsize=2, ms=4)
    style_ax(ax)
    ax.set_ylabel("MLU (median, min–max)")
    ax.set_ylim(1.4, 2.9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=C_TRAIN, label="Trained"),
        Line2D([0], [0], marker="s", color=C_UNTRAIN, ls="--", label="Untrained"),
    ], loc="upper right", framealpha=0.9)
    close(fig, "stability.pdf")

    # ---- fig: cross-scale (main vs 2224, repair-off gains) ----
    scale = load("scale_check.csv")
    sc_train = {(float(r["rho"]), int(r["seed"])): float(r["mlu"])
                for r in scale if r["config"] == "ours" and r["repair"] == "norepair"}
    sc_untrain = {(float(r["rho"]), int(r["seed"])): float(r["mlu"])
                  for r in scale if r["config"] == "untrained" and r["repair"] == "norepair"}
    sc_rhos = [0.05, 0.3]
    main_gains = [gain_off[RHOS.index(r)] for r in sc_rhos]
    sc_gains = [median([(sc_untrain[(r, s)] - sc_train[(r, s)])
                         / sc_untrain[(r, s)] * 100 for s in seeds3])
                for r in sc_rhos]
    # assert against tab:cross-scale (2224 off: -3.2% at 0.3, +24.4% at 0.05)
    for r, g, e in zip(sc_rhos, sc_gains, [24.4, -3.2]):
        assert abs(g - e) <= 0.6, f"2224 off gain at {r}: {g:.1f} vs {e}"

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    x = range(len(sc_rhos))
    w = 0.18
    ax.bar([i - w for i in x], main_gains, w, color=C_TRAIN, label="Main (1,584 sat)")
    ax.bar([i + w for i in x], sc_gains, w, color=C_2224, label="1/3 scale (528 sat)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r"$\rho=0.05$", r"$\rho=0.3$"])
    ax.set_ylabel("Paired training gain (%)")
    ax.set_ylim(-8, 32)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    close(fig, "crossscale.pdf")


if __name__ == "__main__":
    main()
