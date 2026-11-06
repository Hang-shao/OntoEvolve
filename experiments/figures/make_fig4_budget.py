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

import figstyle as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

S.use(8.5)

FS = 8.5
FT = 8.0

BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20, 35, 10 ** 9]
TICKS = ["1", "2", "3", "5", "8", "10", "15", "20", "35", "\u221e"]
ORDER = S.METHODS


def main() -> None:
    d = json.loads((ROOT / "results" / "alert_budget.json").read_text(encoding="utf-8"))
    m = d["methods"]
    x = range(len(BUDGETS))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, fam, title in ((axes[0], "arrival_cap", "(a) First k alarms per stream"),
                           (axes[1], "dedup_then_cap", "(b) One alarm per concept, then k")):
        for k in ORDER:
            ys = [m[k][fam][str(b)]["macro_f1_mean"] for b in BUDGETS]
            st = dict(S.METHOD_STYLE[k])
            ax.plot(x, ys, **st)
        ax.axvspan(-0.4, 2.5, color=S.BAND, zorder=0)
        ax.text(1.05, 0.94, "not separable", ha="center", va="center", fontsize=FT,
                color=S.GREY)
        ax.set_xticks(list(x))
        ax.set_xticklabels(TICKS, fontsize=FT)
        ax.set_xlim(-0.4, len(BUDGETS) - 0.6)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Alarm budget per stream (alarms allowed)")
        S.panel_title(ax, title, pad=5.0)
        S.axis_style(ax)

    axes[0].set_ylabel("Macro-F1", labelpad=2)
    # 两个读数：打平处与分开处（族 A），标注位置与正文 4.3.2 的引数一致
    o = [m["ontoevolve"]["arrival_cap"][str(b)]["macro_f1_mean"] for b in BUDGETS]
    a = [m["adwin"]["arrival_cap"][str(b)]["macro_f1_mean"] for b in BUDGETS]
    ax0 = axes[0]
    # 1 档时本方法与 ADWIN 同为 0.150（两线在此重合），标一次即可
    ax0.annotate("%.3f" % o[0], (0, o[0]), textcoords="offset points", xytext=(7, -13),
                 ha="left", fontsize=FT, color=S.NAVY)
    ax0.annotate("%.3f" % o[2], (2, o[2]), textcoords="offset points", xytext=(-5, 8),
                 ha="right", fontsize=FT, color=S.NAVY)
    for i in (4, 9):
        ax0.annotate("%.3f" % o[i], (i, o[i]), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=FT, color=S.NAVY)
    # ADWIN 的无上限值 0.396 与 incremental clustering 重合，标注会指向两条线，故只标 8 档
    ax0.annotate("%.3f" % a[4], (4, a[4]), textcoords="offset points", xytext=(0, -11),
                 ha="center", fontsize=FT, color=S.ORANGE)

    handles = [Line2D([], [], label=S.METHOD_LABEL[k],
                      **{kk: vv for kk, vv in S.METHOD_STYLE[k].items() if kk != "zorder"})
               for k in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FS,
               handlelength=2.2, handletextpad=0.6, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.10, left=0.085, right=0.99, top=0.88, bottom=0.30)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig4_alert_budget." + ext), bbox_inches="tight")
    plt.close(fig)
    print("fig4 写入：", OUT / "fig4_alert_budget.png")


if __name__ == "__main__":
    main()
