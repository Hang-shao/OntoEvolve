"""生成 Discover Computing 稿的三个新图（英文字样、灰度可印）。

Fig. 3  (a) 组件消融 macro-F1；(b) 融合门 vs 优先级分派
Fig. 5  受控轨道问答准确率（版本 a / b 两面板）
Fig. 6  反诈轨道四种维护策略按知识类型对比

数据来源：
  results/exp4_ablation.json            (a)
  results/exp_fusion_gate/fusion_gate_main.json (b，若缺失则跳过并保留占位)
正文表 7、表 10 的数值（此处硬编码，与表一致）
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

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

DARK, MID, LIGHT, OPEN = ("#333333", "#777777", "#B0B0B0", "#E8E8E8")
HATCH = ("", "//", "..", "xx")


# ---------------------------------------------------------------- Fig. 3
def fig3():
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

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    x = np.arange(len(order))
    colors = [DARK] + [MID] * 4 + [LIGHT] * 3
    ax.bar(x, means, yerr=[los, his], capsize=2.5, color=colors,
           edgecolor="black", linewidth=0.6, width=0.68, zorder=2,
           error_kw={"elinewidth": 0.8, "ecolor": "black"})
    ax.axhline(ab["variants"]["full"]["macro_f1_mean"], color="black",
               linewidth=0.8, linestyle="--", zorder=1)
    for xi, m, h in zip(x, means, his):
        ax.text(xi, m + h + 0.035, "%.3f" % m, ha="center", va="bottom", fontsize=7,
                zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in order], rotation=32, ha="right", fontsize=7.0)
    ax.set_ylabel("macro-F1", labelpad=2)
    ax.set_ylim(0, 0.95)
    ax.set_title("(a) Component ablation", fontsize=9, loc="left")

    gate_path = ROOT / "results/exp_fusion_gate/fusion_gate_main.json"
    ax2 = axes[1]
    if gate_path.exists():
        g = json.loads(gate_path.read_text(encoding="utf-8"))
        tnames = ["emergence", "extinction", "redefinition"]
        tlabels = ["Emergence", "Extinction", "Redefinition"]
        pri = [g["variants"]["priority"]["per_type_f1"][t] for t in tnames]
        fus = [g["variants"]["fused_0.45"]["per_type_f1"][t] for t in tnames]
        xx = np.arange(len(tnames))
        w = 0.36
        ax2.bar(xx - w / 2, pri, width=w, color=DARK, edgecolor="black",
                linewidth=0.6, zorder=2, label="priority")
        ax2.bar(xx + w / 2, fus, width=w, color=LIGHT, edgecolor="black",
                linewidth=0.6, hatch="//", zorder=2, label="fused gate")
        for xi, v in zip(xx - w / 2, pri):
            ax2.text(xi, v + 0.03, "%.3f" % v, ha="center", va="bottom", fontsize=6.6)
        # 融合门在涌现/重定义上与 priority 完全相同，重复标注会互相压字；
        # 只在该系列确实改变的地方标数（淘汰的 0.000 是本面板的核心证据）。
        for xi, v, p in zip(xx + w / 2, fus, pri):
            if abs(v - p) < 1e-9:
                continue
            ax2.text(xi, 0.03, "%.3f" % v, ha="center", va="bottom", fontsize=6.0,
                     rotation=90, bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                            ec="none"))
        ax2.set_xticks(xx)
        ax2.set_xticklabels(tlabels, fontsize=7.5)
        ax2.set_ylim(0, 1.18)
        ax2.set_ylabel("F1 by type", labelpad=2)
        ax2.set_title("(b) Fused gate vs priority", fontsize=9, loc="left")
        ax2.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), handlelength=1.2,
                   handletextpad=0.4, columnspacing=1.2)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "fusion-gate run pending", ha="center", va="center",
                 fontsize=8)
    fig.subplots_adjust(wspace=0.30, left=0.085, right=0.985, top=0.9, bottom=0.30)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_component_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig3 done; gate json present:", gate_path.exists())


# ---------------------------------------------------------------- Fig. 5
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
TYPES = ["Stable", "New", "Redefined"]
SCHEMES = ["Dynamic", "Periodic", "Frozen"]


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    w = 0.26
    for ax, ver, title in zip(axes, ("a", "b"),
                              ("(a) Question version a", "(b) Question version b")):
        x = np.arange(len(TYPES))
        for k, (sc, color, hatch) in enumerate(zip(SCHEMES, (DARK, MID, LIGHT),
                                                   HATCH[:3])):
            vals = QA[ver][sc]
            ax.bar(x + (k - 1) * w, vals, width=w, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.5, label=sc)
            for ti, (xi, v) in enumerate(zip(x + (k - 1) * w, vals)):
                ax.text(xi, v + 0.015, "%.3f" % v, ha="center", va="bottom",
                        fontsize=6.4, rotation=90)
                if sc == "Dynamic":
                    mark = "n.s." if TYPES[ti] == "Stable" else "***"
                    ax.text(xi, v + 0.175, mark, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(TYPES)
        ax.set_ylim(0, 1.32)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("QA accuracy" if ver == "a" else "")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center",
                   bbox_to_anchor=(1.05, -0.12))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_qa_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


# ---------------------------------------------------------------- Fig. 6
AF = {
    "Dynamic evolution": [0.531, 0.750, 0.000, 0.000],
    "Periodic (P = 4)": [0.281, 0.812, 0.562, 0.000],
    "Frozen": [0.594, 0.000, 0.000, 0.000],
    "LLM regeneration": [0.438, 0.344, 0.625, 0.708],
}
AF_TYPES = ["Stable", "New type", "Redefined", "Extinct"]


def fig6():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.2
    x = np.arange(len(AF_TYPES))
    for k, (sc, color, hatch) in enumerate(zip(AF, (DARK, MID, LIGHT, OPEN), HATCH)):
        vals = AF[sc]
        ax.bar(x + (k - 1.5) * w, vals, width=w, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.5, label=sc)
        for xi, v in zip(x + (k - 1.5) * w, vals):
            ax.text(xi, v + 0.015, "%.3f" % v, ha="center", va="bottom",
                    fontsize=6.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(AF_TYPES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("QA accuracy")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6_antifraud_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    fig3()
    fig5()
    fig6()
