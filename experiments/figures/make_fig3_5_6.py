"""生成稿件里的三张结果图（英文、配色与 Fig. 1 一致，灰度可印）。

Fig. 4  (a) 组件消融 macro-F1；(b) 融合门 vs 优先级分派
Fig. 7  受控轨道问答准确率（版本 a / b 两面板）
Fig. 8  反诈轨道四种维护策略按知识类型对比

数据来源：
  results/exp4_ablation.json                     (a)
  results/exp_fusion_gate/fusion_gate_main.json   (b，若缺失则跳过并保留占位)
正文表 7、表 10 的数值（此处硬编码，与表一致）

排版约定：图内字号 8.5 pt（按 174 mm 印宽缩放后仍 ≥ 8 pt），柱色与 hatch 双重编码，
数值标签 90° 竖排以适配窄柱，显著性用括号标在组上方（对应表 7 的 dynamic vs frozen）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import figstyle as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

S.use(8.5)

FS = 8.5        # 数值与刻度统一字号
FST = 8.0       # 类别刻度

FULL = dict(color=S.NAVY, hatch="")
ABL = dict(color=S.ORANGE_L, hatch="//")
GRAN = dict(color=S.SKY, hatch="xx")


def bracket(ax, x1, x2, y, text, *, h=0.018, fs=FS):
    """组上方的显著性括号（两端短竖线 + 顶部横线 + 居中标注）。"""
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color=S.INK,
            linewidth=0.7, solid_capstyle="butt", clip_on=False, zorder=5)
    ax.text((x1 + x2) / 2.0, y + 0.012, text, ha="center", va="bottom",
            fontsize=fs, zorder=5)


# ---------------------------------------------------------------- Fig. 4
def fig4():
    ab = json.loads((ROOT / "results/exp4_ablation.json").read_text(encoding="utf-8"))
    order = [
        ("full", "Full"),
        ("no_semantic", "w/o semantic"),
        ("no_stat", "w/o statistical"),
        ("no_axiom", "w/o axiom"),
        ("no_support_decay", "w/o supp. decay"),
        ("fixed_k_gt", "K = K*"),
        ("granularity_half", "K = K*/2"),
        ("granularity_double", "K = 2K*"),
    ]
    means, los, his = [], [], []
    for key, _ in order:
        v = ab["variants"][key]
        means.append(v["macro_f1_mean"])
        los.append(v["macro_f1_mean"] - v["macro_f1_ci95"][0])
        his.append(v["macro_f1_ci95"][1] - v["macro_f1_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    x = np.arange(len(order))
    styles = [FULL] + [ABL] * 4 + [GRAN] * 3
    ax.bar(x, means, yerr=[los, his], capsize=2.2,
           color=[s["color"] for s in styles], edgecolor=S.INK, linewidth=0.6,
           hatch=None, width=0.66, zorder=3,
           error_kw={"elinewidth": 0.8, "ecolor": S.INK})
    for xi, s in zip(x, styles):
        if s["hatch"]:
            ax.patches[int(xi)].set_hatch(s["hatch"])
    ax.axhline(ab["variants"]["full"]["macro_f1_mean"], color=S.BLUE,
               linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
    for xi, m, h in zip(x, means, his):
        ax.text(xi, m + h + 0.028, "%.3f" % m, ha="center", va="bottom",
                fontsize=FS, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in order], rotation=30, ha="right", fontsize=FST)
    ax.set_ylabel("macro-F1", labelpad=2)
    ax.set_ylim(0, 0.92)
    S.panel_title(ax, "(a) Component ablation")
    S.axis_style(ax)

    gate_path = ROOT / "results/exp_fusion_gate/fusion_gate_main.json"
    ax2 = axes[1]
    if gate_path.exists():
        g = json.loads(gate_path.read_text(encoding="utf-8"))
        tnames = ["emergence", "extinction", "redefinition"]
        pri = [g["variants"]["priority"]["per_type_f1"][t] for t in tnames]
        fus = [g["variants"]["fused_0.45"]["per_type_f1"][t] for t in tnames]
        xx = np.arange(len(tnames))
        w = 0.36
        ax2.bar(xx - w / 2, pri, width=w, color=S.NAVY, edgecolor=S.INK,
                linewidth=0.6, zorder=3, label="priority dispatch")
        ax2.bar(xx + w / 2, fus, width=w, color=S.ORANGE_L, edgecolor=S.INK,
                linewidth=0.6, hatch="//", zorder=3, label="fused gate")
        for xi, v in zip(xx - w / 2, pri):
            ax2.text(xi, v + 0.025, "%.3f" % v, ha="center", va="bottom", fontsize=FS)
        # 融合门在涌现/重定义上与 priority 完全相同，重复标注会互相压字；
        # 只在该系列确实改变的地方标数（淘汰的 0.000 是本面板的核心证据）。
        for xi, v, p in zip(xx + w / 2, fus, pri):
            if abs(v - p) < 1e-9:
                continue
            ax2.text(xi, 0.02, "%.3f" % v, ha="center", va="bottom", fontsize=FS,
                     rotation=90, bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                            ec="none"))
        ax2.set_xticks(xx)
        ax2.set_xticklabels([S.TYPE_LABEL[t] for t in tnames], fontsize=FST)
        ax2.set_ylim(0, 1.24)
        ax2.set_ylabel("F1 by type", labelpad=2)
        S.panel_title(ax2, "(b) Fused gate vs priority")
        S.axis_style(ax2)
        ax2.legend(loc="upper center", fontsize=FS, ncol=2,
                   bbox_to_anchor=(0.5, 1.05), handlelength=1.1,
                   handletextpad=0.45, columnspacing=1.2)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "fusion-gate run pending", ha="center", va="center",
                 fontsize=FS)
    fig.subplots_adjust(wspace=0.30, left=0.085, right=0.985, top=0.90, bottom=0.34)
    # (a) 的三类柱只有图例能说明含义；放在面板下方一行，避免压住 K=K* 的柱与数值
    p = axes[0].get_position()
    fig.legend(handles=[Patch(facecolor=FULL["color"], edgecolor=S.INK, label="full model"),
                        Patch(facecolor=ABL["color"], edgecolor=S.INK, hatch=ABL["hatch"],
                              label="signal removed"),
                        Patch(facecolor=GRAN["color"], edgecolor=S.INK, hatch=GRAN["hatch"],
                              label="cluster granularity")],
               loc="upper center", bbox_to_anchor=((p.x0 + p.x1) / 2.0, p.y0 - 0.20),
               fontsize=FS, ncol=3, handlelength=1.1, handletextpad=0.45,
               columnspacing=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_component_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig4 done; gate json present:", gate_path.exists())


# ---------------------------------------------------------------- Fig. 7
QA = {
    "a": {
        "Dynamic": [0.996, 0.855, 0.717],
        "Periodic": [0.967, 0.060, 0.686],
        "Frozen": [1.000, 0.053, 0.025],
    },
    "b": {
        "Dynamic": [0.928, 0.987, 0.793],
        "Periodic": [0.832, 0.532, 0.348],
        "Frozen": [0.938, 0.131, 0.027],
    },
}
# 表 7 的 dynamic vs frozen（版本 a/b × Stable/New/Redefined）
QA_SIG = {
    "a": ["n.s.", "***", "***"],
    "b": ["n.s.", "***", "***"],
}
TYPES = ["Stable", "New", "Redefined"]
SCHEMES = ["Dynamic", "Periodic", "Frozen"]


def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    w = 0.26
    for ax, ver, title in zip(axes, ("a", "b"),
                              ("(a) Question version a", "(b) Question version b")):
        x = np.arange(len(TYPES))
        for k, sc in enumerate(SCHEMES):
            st = S.SCHEME_STYLE[sc]
            vals = QA[ver][sc]
            ax.bar(x + (k - 1) * w, vals, width=w, color=st["color"], hatch=st["hatch"],
                   edgecolor=S.INK, linewidth=0.6, label=S.SCHEME_LABEL[sc], zorder=3 - k)
            for xi, v in zip(x + (k - 1) * w, vals):
                ax.text(xi, v + 0.018, "%.3f" % v, ha="center", va="bottom",
                        fontsize=FS, rotation=90)
        for ti, xt in enumerate(x):
            hi = max(QA[ver][s][ti] for s in SCHEMES)
            bracket(ax, xt - w, xt + w, hi + 0.30, QA_SIG[ver][ti])
        ax.set_xticks(x)
        ax.set_xticklabels(TYPES, fontsize=FST)
        ax.set_ylim(0, 1.38)
        S.panel_title(ax, title, pad=8.0)
        S.axis_style(ax)
        if ver == "a":
            ax.set_ylabel("QA accuracy")
    handles = [Patch(facecolor=S.SCHEME_STYLE[s]["color"], hatch=S.SCHEME_STYLE[s]["hatch"],
                     edgecolor=S.INK, linewidth=0.6, label=S.SCHEME_LABEL[s])
               for s in SCHEMES]
    fig.legend(handles=handles, loc="upper center", fontsize=FS, ncol=3,
               bbox_to_anchor=(0.5, 0.075), handlelength=1.1, handletextpad=0.45,
               columnspacing=1.8)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.24, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6_qa_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig7 done")


# ---------------------------------------------------------------- Fig. 8
AF = {
    "Dynamic evolution": [0.531, 0.750, 0.000, 0.000],
    "Periodic (P = 4)": [0.281, 0.812, 0.562, 0.000],
    "Frozen": [0.594, 0.000, 0.000, 0.000],
    "LLM regeneration": [0.438, 0.344, 0.625, 0.708],
}
AF_KEY = {"Dynamic evolution": "Dynamic", "Periodic (P = 4)": "Periodic",
          "Frozen": "Frozen", "LLM regeneration": "LLM regeneration"}
AF_TYPES = ["Stable", "New type", "Redefined", "Extinct"]


def fig8():
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    w = 0.2
    x = np.arange(len(AF_TYPES))
    for k, sc in enumerate(AF):
        st = S.SCHEME_STYLE[AF_KEY[sc]]
        vals = AF[sc]
        ax.bar(x + (k - 1.5) * w, vals, width=w, color=st["color"], hatch=st["hatch"],
               edgecolor=S.INK, linewidth=0.6, label=sc, zorder=4 - k)
        for xi, v in zip(x + (k - 1.5) * w, vals):
            ax.text(xi, v + 0.012, "%.3f" % v, ha="center", va="bottom",
                    fontsize=FS, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(AF_TYPES, fontsize=FST)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("QA accuracy")
    S.axis_style(ax)
    ax.legend(loc="upper center", fontsize=FS, ncol=4,
              bbox_to_anchor=(0.5, -0.09), handlelength=1.1, handletextpad=0.45,
              columnspacing=1.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig7_antifraud_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig8 done")


if __name__ == "__main__":
    fig4()
    fig7()
    fig8()
