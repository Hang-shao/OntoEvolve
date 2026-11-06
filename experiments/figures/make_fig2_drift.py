"""Fig. 3：漂移检测 macro-F1 与三类分型 F1（数据取自 results/exp1_summary.json）。

本脚本只做标注语言与排版，数值与数据源由 `run_exp1.py` 决定；配色与字号统一从
`figstyle` 取，与 Fig. 1 的深蓝/绿/橙体系一致，灰度下靠 hatch 区分。
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

import figstyle as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

S.use(8.5)

ORDER = ["ontoevolve", "adwin", "inc_clustering", "embdist", "embkmeans", "page_hinkley"]
LABEL = {"ontoevolve": "OntoEvolve\n(ours)", "adwin": "ADWIN",
         "inc_clustering": "Inc.\nclustering", "embdist": "Emb.\ndist.",
         "embkmeans": "Emb.\nk-means", "page_hinkley": "Page–\nHinkley"}
TYPES = ["emergence", "extinction", "redefinition"]


def bar(ax, x, y, w, k, **kw):
    st = S.METHOD_BAR[k]
    b = ax.bar(x, y, w, color=st["color"], edgecolor=S.INK, linewidth=0.6,
               hatch=st["hatch"], zorder=st["zorder"], **kw)
    return b


def main():
    src = Path(os.getenv("ONTOEVOLVE_FIG_SRC", str(ROOT / "results/exp1_summary.json")))
    m = json.loads(src.read_text(encoding="utf-8"))["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.3]})

    # ---- (a) macro-F1 + 95% CI
    ax = axes[0]
    val = np.array([m[k]["macro_f1_mean"] for k in ORDER])
    lo = np.array([m[k]["macro_f1_ci95"][0] for k in ORDER])
    hi = np.array([m[k]["macro_f1_ci95"][1] for k in ORDER])
    x = np.arange(len(ORDER))
    for i, k in enumerate(ORDER):
        bar(ax, x[i], val[i], 0.64, k)
    ax.errorbar(x, val, yerr=[val - lo, hi - val], fmt="none", ecolor=S.INK,
                elinewidth=0.8, capsize=2.6, zorder=8)
    for i, (v, h) in enumerate(zip(val, hi)):
        ax.text(i, h + 0.025, "%.3f" % v, ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k] for k in ORDER], fontsize=7.2,
                       rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    S.panel_title(ax, "(a) Detection accuracy")
    S.axis_style(ax)

    # ---- (b) 逐类型 F1
    ax = axes[1]
    w = 0.135
    xi = np.arange(len(TYPES))
    for j, k in enumerate(ORDER):
        ys = [m[k]["per_type_mean"][t] for t in TYPES]
        b = bar(ax, xi + (j - 2.5) * w, ys, w, k)
        if k == "ontoevolve":
            for r, y in zip(b, ys):
                ax.text(r.get_x() + r.get_width() / 2, y + 0.025, "%.3f" % y,
                        ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(xi)
    ax.set_xticklabels([S.TYPE_LABEL[t] for t in TYPES])
    ax.set_ylabel("F1 by drift type")
    ax.set_ylim(0, 1.05)
    S.panel_title(ax, "(b) Per-type breakdown")
    S.axis_style(ax)

    handles = [Patch(facecolor=S.METHOD_BAR[k]["color"], hatch=S.METHOD_BAR[k]["hatch"],
                     edgecolor=S.INK, linewidth=0.6, label=S.METHOD_LABEL[k])
               for k in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=7.4, handlelength=1.1, handleheight=0.9,
               handletextpad=0.45, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.26, left=0.075, right=0.99, top=0.9, bottom=0.30)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig2_drift_f1." + ext), bbox_inches="tight")
    plt.close(fig)
    print("fig3（文件 fig2_drift_f1）写入：", OUT / "fig2_drift_f1.png")


if __name__ == "__main__":
    main()
