"""图 4：五种子单会话的成本证据（含周期重构 P=5）。

(a) 累计耗时（对数坐标）：增量演化 / 逐批全量重建 / 周期重构 P=5，均值 ± 标准差
(b) 逐种子累计加速比：对逐批重建与对周期重构两组，含 95% CI 带与一个数量级判定线
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)
TAG = "t36_full5"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
DARK, MID, LIGHT = "#333333", "#777777", "#B0B0B0"


def main():
    d = json.loads((ROOT / f"results/exp_cost/exp_cost_{TAG}.json").read_text(encoding="utf-8"))
    agg, per = d["aggregate"], d["per_seed"]

    inc = np.array(agg["incremental_cum_s"]["values"], dtype=float)
    full = np.array(agg["full_rebuild_cum_s"]["values"], dtype=float)
    per_cum = np.array(agg["periodic_cum_s"]["values"], dtype=float)
    sp = np.array(agg["speedup_cumulative"]["values"], dtype=float)
    spp = np.array(agg["periodic_speedup"]["values"], dtype=float)
    ci = agg["speedup_cumulative"]["ci95"]
    cip = agg["periodic_speedup"]["ci95"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    # ---- (a) 三条路径的累计耗时 ----
    ax = axes[0]
    series = [("Incremental\nevolution", inc, DARK),
              ("Per-batch\nfull rebuild", full, LIGHT),
              ("Periodic\nrebuild (P = 5)", per_cum, MID)]
    x = np.arange(len(series))
    means = [s[1].mean() for s in series]
    errs = [s[1].std(ddof=1) for s in series]
    ax.bar(x, means, yerr=errs, capsize=3, width=0.6,
           color=[s[2] for s in series], edgecolor="black", linewidth=0.6,
           error_kw={"elinewidth": 0.8, "ecolor": "black"})
    for xi, m in zip(x, means):
        ax.text(xi, m * 1.22, "%.1f s" % m, ha="center", va="bottom", fontsize=7.2)
    ax.set_yscale("log")
    ax.set_ylim(min(means) * 0.45, max(means) * 4.0)
    ax.set_xlim(-0.65, len(series) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in series], fontsize=7.2)
    ax.set_ylabel("Cumulative time (s, log scale)")
    ax.set_title("(a) Cumulative cost over T = 36 batches", fontsize=9, loc="left")
    ax.text(0.03, 0.97, "mean ± std over %d seeds" % len(inc),
            transform=ax.transAxes, fontsize=6.5, va="top")

    # ---- (b) 逐种子加速比 ----
    ax2 = axes[1]
    xs = np.arange(len(sp))
    w = 0.38
    ax2.bar(xs - w / 2, sp, width=w, color=MID, edgecolor="black", linewidth=0.6,
            zorder=2, label="vs per-batch rebuild")
    ax2.bar(xs + w / 2, spp, width=w, color="#D9D9D9", edgecolor="black", linewidth=0.6,
            hatch="//", zorder=2, label="vs periodic rebuild (P = 5)")
    ax2.axhspan(ci[0], ci[1], color="0.88", zorder=0)
    ax2.axhspan(cip[0], cip[1], color="0.94", zorder=0)
    ax2.axhline(10, color="black", linestyle="--", linewidth=0.8, zorder=1)
    for xi, v in zip(xs - w / 2, sp):
        ax2.text(xi, v * 1.02, "%.1f" % v, ha="center", va="bottom", fontsize=6.2)
    for xi, v in zip(xs + w / 2, spp):
        ax2.text(xi, v * 1.02, "%.1f" % v, ha="center", va="bottom", fontsize=6.2)
    ax2.set_ylim(0, max(sp) * 1.45)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["seed %d" % r["seed"] for r in per], fontsize=7.5)
    ax2.set_ylabel("Cumulative speed-up (×)")
    ax2.set_title("(b) Speed-up per seed", fontsize=9, loc="left")
    ax2.legend(frameon=False, fontsize=6.4, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_cost_speedup.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("inc %.2f  per-batch %.2f  periodic %.2f  speed %s  periodic-speed %s"
          % (inc.mean(), full.mean(), per_cum.mean(), np.round(sp, 2), np.round(spp, 2)))


if __name__ == "__main__":
    main()
