"""图 4：告警预算敏感性（数据取自 results/alert_budget.json，十档预算、两族选择规则）。

与 `run_alert_budget.py` 同口径：预算施加在**已归档**的告警流上，不重跑检测器，
匹配窗、评分代码与真值均不变。族 A 按到达顺序保留前 k 条；族 B 先每个概念只留第一条，
再截断到 k。本脚本只做排版与标注，数值由归档决定。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.linewidth": 0.9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1a1a1a"
GRID = "#dcdcdc"
BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20, 35, 10 ** 9]
TICKS = ["1", "2", "3", "5", "8", "10", "15", "20", "35", "\u221e"]
LEGEND = {"ontoevolve": "OntoEvolve (ours)", "adwin": "ADWIN",
          "inc_clustering": "Incremental clustering", "page_hinkley": "Page-Hinkley",
          "embdist": "Embedding distribution", "embkmeans": "Embedding k-means"}
STYLE = {
    "ontoevolve":    dict(color=INK,       ls="-",  lw=1.6, marker="o", ms=3.4, zorder=5),
    "adwin":         dict(color="#4d4d4d", ls="--", lw=1.3, marker="s", ms=3.0, zorder=4),
    "inc_clustering": dict(color="#8c8c8c", ls="-.", lw=1.1, marker="^", ms=3.0, zorder=3),
    "page_hinkley":  dict(color="#b3b3b3", ls=":",  lw=1.0, marker="v", ms=2.6, zorder=2),
    "embdist":       dict(color="#c9c9c9", ls=":",  lw=1.0, marker="d", ms=2.4, zorder=2),
    "embkmeans":     dict(color="#dedede", ls=":",  lw=1.0, marker="x", ms=2.6, zorder=2),
}
ORDER = ["ontoevolve", "adwin", "inc_clustering", "page_hinkley", "embdist", "embkmeans"]


def main() -> None:
    d = json.loads((ROOT / "results" / "alert_budget.json").read_text(encoding="utf-8"))
    m = d["methods"]
    x = range(len(BUDGETS))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, fam, title in ((axes[0], "arrival_cap", "(a) First k alarms per stream"),
                           (axes[1], "dedup_then_cap", "(b) One alarm per concept, then k")):
        for k in ORDER:
            ys = [m[k][fam][str(b)]["macro_f1_mean"] for b in BUDGETS]
            ax.plot(x, ys, **STYLE[k])
        ax.axvspan(-0.4, 2.2, color="#f0f0f0", zorder=0)
        ax.text(0.95, 0.925, "parity", ha="center", va="center", fontsize=7.0,
                color="#4d4d4d")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(list(x))
        ax.set_xticklabels(TICKS, fontsize=7.2)
        ax.set_xlim(-0.4, len(BUDGETS) - 0.6)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Alarm budget per stream (alarms allowed)")
        ax.set_title(title, loc="left", fontsize=8.5)

    axes[0].set_ylabel("Macro-F1")
    # 两个读数：打平处与分开处（族 A）
    o = [m["ontoevolve"]["arrival_cap"][str(b)]["macro_f1_mean"] for b in BUDGETS]
    a = [m["adwin"]["arrival_cap"][str(b)]["macro_f1_mean"] for b in BUDGETS]
    for i, txt in ((2, "%.3f" % o[2]), (4, "%.3f" % o[4]), (9, "%.3f" % o[9])):
        axes[0].annotate(txt, (i, o[i]), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=6.8, color=INK)
    for i, txt in ((0, "%.3f" % a[0]), (4, "%.3f" % a[4]), (9, "%.3f" % a[9])):
        axes[0].annotate(txt, (i, a[i]), textcoords="offset points", xytext=(0, -9),
                         ha="center", fontsize=6.8, color="#4d4d4d")

    handles = [Line2D([], [], label=LEGEND[k], **{kk: vv for kk, vv in STYLE[k].items()
                                                  if kk != "zorder"}) for k in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.0,
               handlelength=2.0, handletextpad=0.6, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.10, left=0.085, right=0.99, top=0.88, bottom=0.30)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig4_alert_budget." + ext), bbox_inches="tight")
    plt.close(fig)
    print("fig4 写入：", OUT / "fig4_alert_budget.png")


if __name__ == "__main__":
    main()
