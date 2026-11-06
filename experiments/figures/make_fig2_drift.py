"""图 2：漂移检测 macro-F1 与三类分型 F1（数据取自 results/exp1_summary.json）。

本脚本只做标注语言与排版，数值与数据源由 `run_exp1.py` 决定。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.linewidth": 0.9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1a1a1a"
GRID = "#dcdcdc"
ORDER = ["ontoevolve", "adwin", "inc_clustering", "embdist", "embkmeans", "page_hinkley"]
LABEL = {"ontoevolve": "OntoEvolve\n(ours)", "adwin": "ADWIN",
         "inc_clustering": "Inc.\nclustering", "embdist": "Emb.\ndist.",
         "embkmeans": "Emb.\nk-means", "page_hinkley": "Page-\nHinkley"}
LEGEND = {"ontoevolve": "OntoEvolve (ours)", "adwin": "ADWIN",
          "inc_clustering": "Incremental clustering", "embdist": "Embedding distribution",
          "embkmeans": "Embedding k-means", "page_hinkley": "Page-Hinkley"}
STYLE = {"ontoevolve": dict(fc="#1a1a1a", h=""),
         "adwin": dict(fc="#5f5f5f", h=""),
         "inc_clustering": dict(fc="#9a9a9a", h="\\\\"),
         "embdist": dict(fc="#c8c8c8", h="//"),
         "embkmeans": dict(fc="#e6e6e6", h="xx"),
         "page_hinkley": dict(fc="#ffffff", h="..")}
TYPES = ["emergence", "extinction", "redefinition"]
TYPE_EN = {"emergence": "Emergence", "extinction": "Extinction",
           "redefinition": "Redefinition"}


def bar(ax, x, y, w, k, **kw):
    b = ax.bar(x, y, w, color=STYLE[k]["fc"], edgecolor=INK, linewidth=0.6, **kw)
    if STYLE[k]["h"]:
        for r in b:
            r.set_hatch(STYLE[k]["h"])
    return b


def finish(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    src = Path(os.getenv("ONTOEVOLVE_FIG_SRC", str(ROOT / "results/exp1_summary.json")))
    m = json.loads(src.read_text(encoding="utf-8"))["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    val = np.array([m[k]["macro_f1_mean"] for k in ORDER])
    lo = np.array([m[k]["macro_f1_ci95"][0] for k in ORDER])
    hi = np.array([m[k]["macro_f1_ci95"][1] for k in ORDER])
    x = np.arange(len(ORDER))
    for i, k in enumerate(ORDER):
        bar(ax, x[i], val[i], 0.62, k, zorder=2)
    ax.errorbar(x, val, yerr=[val - lo, hi - val], fmt="none", ecolor=INK,
                elinewidth=0.8, capsize=3, zorder=3)
    for i, (v, h) in enumerate(zip(val, hi)):
        ax.text(i, h + 0.025, "%.3f" % v, ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=6.5,
                       rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Macro-F1, bootstrap 95% CI", loc="left", fontsize=8.5)
    finish(ax)

    ax = axes[1]
    w = 0.135
    xi = np.arange(len(TYPES))
    for j, k in enumerate(ORDER):
        ys = [m[k]["per_type_mean"][t] for t in TYPES]
        b = bar(ax, xi + (j - 2.5) * w, ys, w, k, zorder=2)
        if k == "ontoevolve":
            for r, y in zip(b, ys):
                ax.text(r.get_x() + r.get_width() / 2, y + 0.025, "%.3f" % y,
                        ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(xi)
    ax.set_xticklabels([TYPE_EN[t] for t in TYPES], fontsize=8.5)
    ax.set_ylabel("F1 by type")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) F1 by drift type", loc="left", fontsize=8.5)
    finish(ax)

    handles = [Patch(facecolor=STYLE[k]["fc"], hatch=STYLE[k]["h"], edgecolor=INK,
                     linewidth=0.6, label=LEGEND[k]) for k in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=6.6, handlelength=1.2, handletextpad=0.5, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.26, left=0.075, right=0.99, top=0.9, bottom=0.30)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig2_drift_f1." + ext), bbox_inches="tight")
    plt.close(fig)
    print("fig2 写入 English 版：", OUT / "fig2_drift_f1.png")


if __name__ == "__main__":
    main()
