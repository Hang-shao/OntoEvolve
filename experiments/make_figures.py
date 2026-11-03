#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文插图（图 2—图 6）。

数据源全部来自 results/ 下的实验产物，不重跑实验。
输出：矢量 PDF（投稿用，TrueType 字体内嵌）+ 预览 PNG（校对用）。

中文核心期刊视觉规范：
  - 中文宋体 / 西文与数字 Times New Roman（font.family 列表启用逐字回退）；
  - 线宽 0.6—2.0 pt，最小字号 6 pt；
  - 以填充灰阶 + 网纹区分序列，保证灰度打印可辨；
  - 无渐变、无阴影、无三维效果。
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W_DOUBLE = 170 * MM
W_SINGLE = 84 * MM

INK = "#1a1a1a"
C_MAIN = "#1f3b73"       # 本文方法 / 动态演化
C_PERIOD = "#7f9dc4"     # 周期全量重构
C_FROZEN = "#bdbdbd"     # 冻结本体
C_GRID = "#dcdcdc"

METHOD_ORDER = ["ontoevolve", "adwin", "inc_clustering", "embdist", "embkmeans", "page_hinkley"]
METHOD_CN = {
    "ontoevolve": "OntoEvolve（本文）",
    "adwin": "ADWIN",
    "inc_clustering": "增量聚类",
    "embdist": "嵌入分布距离",
    "embkmeans": "嵌入 k-means",
    "page_hinkley": "Page-Hinkley",
}
# 以灰阶 + 网纹区分，保证灰度打印可辨；本文方法用最深色
METHOD_STYLE = {
    "ontoevolve": dict(facecolor=C_MAIN, hatch=""),
    "adwin": dict(facecolor="#6f8fbf", hatch=""),
    "inc_clustering": dict(facecolor="#a8c0dd", hatch="\\\\"),
    "embdist": dict(facecolor="#cfcfcf", hatch="//"),
    "embkmeans": dict(facecolor="#e2e2e2", hatch="xx"),
    "page_hinkley": dict(facecolor="#ffffff", hatch=".."),
}
TYPE_ORDER = ["emergence", "extinction", "redefinition"]
TYPE_CN = {"emergence": "涌现", "extinction": "淘汰", "redefinition": "重定义"}
KCLASS_ORDER = ["stable", "new", "redefined"]
KCLASS_CN = {"stable": "稳定知识", "new": "新知识", "redefined": "重定义知识"}
ARM_STYLE = {
    "dynamic": dict(facecolor=C_MAIN, hatch=""),
    "periodic": dict(facecolor=C_PERIOD, hatch="//"),
    "frozen": dict(facecolor=C_FROZEN, hatch="\\\\"),
}
ARM_CN = {"dynamic": "动态演化", "periodic": "周期全量重构", "frozen": "冻结本体"}

plt.rcParams.update({
    # 列表形式启用逐字回退：西文/数字取 Times New Roman，中文取宋体
    "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "patch.linewidth": 0.6,
    "lines.linewidth": 1.0,
})


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf / {name}.png")


def _finish(ax, *, grid_axis="y"):
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.6)


def _bar(ax, x, ys, width, style, **kw):
    """按给定样式逐柱设置填充与网纹（网纹不支持列表传参，故逐柱设置）。"""
    bars = ax.bar(x, ys, width, facecolor=style["facecolor"],
                  edgecolor=INK, linewidth=0.6, **kw)
    h = style["hatch"]
    if h:
        for r in bars:
            r.set_hatch(h)
    return bars


def _load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 图 2
def fig2() -> None:
    """命题 A：macro-F1 与三类分型 F1 对比。"""
    m = _load("exp1_summary.json")["methods"]

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 60 * MM))

    # ---- (a) macro-F1 + 95% CI ----
    ax = axes[0]
    val = np.array([m[k]["macro_f1_mean"] for k in METHOD_ORDER])
    lo = np.array([m[k]["macro_f1_ci95"][0] for k in METHOD_ORDER])
    hi = np.array([m[k]["macro_f1_ci95"][1] for k in METHOD_ORDER])
    x = np.arange(len(METHOD_ORDER))
    for i, k in enumerate(METHOD_ORDER):
        _bar(ax, x[i], val[i], 0.62, METHOD_STYLE[k])
    ax.errorbar(x, val, yerr=[val - lo, hi - val], fmt="none", ecolor=INK,
                elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)
    for i, (v, h) in enumerate(zip(val, hi)):
        ax.text(i, h + 0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.2, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["OntoEvolve\n（本文）", "ADWIN", "增量聚类", "嵌入分布\n距离",
                        "嵌入\nk-means", "Page-\nHinkley"], fontsize=7.2)
    ax.set_ylabel("宏平均 F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) 宏平均 F1 对比（误差棒为 bootstrap 95% 置信区间）", loc="left", fontsize=8.5)
    _finish(ax)

    # ---- (b) 三类分型 F1 ----
    ax = axes[1]
    w = 0.135
    xi = np.arange(len(TYPE_ORDER))
    for j, k in enumerate(METHOD_ORDER):
        ys = [m[k]["per_type_mean"][t] for t in TYPE_ORDER]
        b = _bar(ax, xi + (j - 2.5) * w, ys, w, METHOD_STYLE[k])
        if k == "ontoevolve":
            for r, y in zip(b, ys):
                ax.text(r.get_x() + r.get_width() / 2, y + 0.02, f"{y:.3f}",
                        ha="center", va="bottom", fontsize=6.4, color=INK)
        elif k in ("adwin", "inc_clustering"):
            # 重定义柱是正文分析的关键证据（ADWIN 0.048、增量聚类 0.000），单独标注
            r, y = b[2], ys[2]
            ax.text(r.get_x() + r.get_width() / 2, y + 0.02, f"{y:.3f}",
                    ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.set_xticks(xi)
    ax.set_xticklabels([TYPE_CN[t] for t in TYPE_ORDER], fontsize=8.5)
    ax.set_ylabel("分型 F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) 三类漂移分型 F1 对比", loc="left", fontsize=8.5)
    _finish(ax)

    # 两面板共用同一套方法图例，置于底部，避免遮挡柱体
    handles = [Patch(facecolor=METHOD_STYLE[k]["facecolor"], hatch=METHOD_STYLE[k]["hatch"],
                     edgecolor=INK, linewidth=0.6, label=METHOD_CN[k])
               for k in METHOD_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=6.8, handlelength=1.3, handletextpad=0.5, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.30, left=0.062, right=0.985, top=0.875, bottom=0.265)
    _save(fig, "fig2_drift_f1")


# ---------------------------------------------------------------- 图 3
def fig3() -> None:
    """命题 A：告警质量的精确率—召回率权衡（气泡面积 ∝ 每轮告警数）。"""
    rows = list(csv.DictReader((RES / "exp1_main.csv").open(encoding="utf-8")))
    agg = {}
    for r in rows:
        a = agg.setdefault(r["method"], {"tp": 0.0, "fp": 0.0, "fn": 0.0, "al": 0.0, "n": 0})
        a["tp"] += float(r["tp"])
        a["fp"] += float(r["fp"])
        a["fn"] += float(r["fn"])
        a["al"] += float(r["n_alarms"])
        a["n"] += 1

    data = []
    for k in METHOD_ORDER:
        a = agg[k]
        prec = a["tp"] / max(a["tp"] + a["fp"], 1e-12)
        rec = a["tp"] / max(a["tp"] + a["fn"], 1e-12)
        data.append((k, rec, prec, a["al"] / a["n"]))
    print("  图 3 复核（方法, 召回率, 精确率, 每轮告警数）:")
    for k, rec, prec, al in data:
        print(f"    {k:15s} R={rec:.3f} P={prec:.3f} alarms={al:.1f}")

    la = np.log10([d[3] for d in data])
    sizes = 30 + 150 * (la - la.min()) / (la.max() - la.min())
    # 各点标签的避让偏移（points, (dx, dy, ha, va)）
    off = {
        "ontoevolve": (0, 11, "center", "bottom"),
        "adwin": (0, 11, "center", "bottom"),
        "inc_clustering": (0, -13, "center", "top"),
        "embkmeans": (0, 9, "center", "bottom"),
        "embdist": (0, -11, "center", "top"),
        "page_hinkley": (0, -24, "center", "top"),
    }

    fig, ax = plt.subplots(figsize=(W_SINGLE, 74 * MM))
    for (k, rec, prec, al), s in zip(data, sizes):
        st = METHOD_STYLE[k]
        ax.scatter(rec, prec, s=s, facecolor=st["facecolor"], hatch=st["hatch"],
                   edgecolor=INK, linewidth=0.8, zorder=3)
        dx, dy, ha, va = off[k]
        ax.annotate(f"{METHOD_CN[k]}（{al:g}）", (rec, prec),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=6.0, color=INK, zorder=4)

    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_xlim(0.40, 1.06)
    ax.set_ylim(-0.20, 0.92)
    ax.set_title("告警质量：精确率—召回率权衡\n（气泡面积 ∝ 每轮告警数，对数映射）",
                 loc="left", fontsize=8.5)
    _finish(ax, grid_axis="both")
    fig.subplots_adjust(left=0.15, right=0.965, top=0.845, bottom=0.125)
    _save(fig, "fig3_pr_tradeoff")


# ---------------------------------------------------------------- 图 4
def fig4() -> None:
    """命题 B：累计成本对数柱、逐批耗时曲线、逐种子加速比。"""
    s1 = _load("exp_cost/exp_cost_t36.json")
    msd = _load("exp_cost/exp_cost_t36_multiseed.json")

    inc0 = s1["incremental"]["cum_ms"] / 1000.0
    per0 = s1["periodic"]["cum_ms"] / 1000.0
    full0 = s1["full_rebuild"]["cum_ms"] / 1000.0
    inc_m = float(np.mean([p["incremental_cum_s"] for p in msd["per_seed"]]))
    full_m = float(np.mean([p["full_rebuild_cum_s"] for p in msd["per_seed"]]))

    seeds = [p["seed"] for p in msd["per_seed"]]
    sp = [p["speedup_cumulative"] for p in msd["per_seed"]]
    cum = msd["aggregate"]["speedup_cumulative"]

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 54 * MM))

    # ---- (a) 累计耗时（水平柱，对数坐标；数值置于柱内，避免重叠） ----
    ax = axes[0]
    arms = ["增量演化", "周期全量\n重构 (P=5)", "逐批全量\n重构"]
    y = np.arange(3)
    ax.barh(y - 0.19, [inc0, per0, full0], 0.36, facecolor=C_MAIN,
            edgecolor=INK, linewidth=0.6, label="seed 0")
    for yy, v in zip(y - 0.19, [inc0, per0, full0]):
        ax.text(v * 1.06, yy, f"{v:.2f}", ha="left", va="center",
                fontsize=6.2, color=INK, zorder=4)
    ax.barh(y[0] + 0.19, inc_m, 0.36, facecolor="#cfcfcf", hatch="//",
            edgecolor=INK, linewidth=0.6, label="5 种子均值")
    ax.barh(y[2] + 0.19, full_m, 0.36, facecolor="#cfcfcf", hatch="//",
            edgecolor=INK, linewidth=0.6)
    for yy, v in [(y[0] + 0.19, inc_m), (y[2] + 0.19, full_m)]:
        ax.text(v * 1.06, yy, f"{v:.2f}", ha="left", va="center",
                fontsize=6.2, color=INK, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(4, 4000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y)
    ax.set_yticklabels(arms, fontsize=7.0)
    ax.set_xlabel("累计耗时 / s（对数坐标）")
    ax.legend(frameon=False, fontsize=6.4, loc="lower right",
              handlelength=1.2, handletextpad=0.5)
    ax.set_title("(a) 累计耗时对比（T=36）", loc="left", fontsize=8.5)
    _finish(ax, grid_axis="x")

    # ---- (b) 逐批耗时曲线：实心=实测，空心=外推 ----
    ax = axes[1]
    meta = {"full_rebuild": ("^", C_FROZEN, "逐批全量重构"),
            "periodic_P5": ("s", C_PERIOD, "周期重构 (P=5)"),
            "incremental": ("o", C_MAIN, "增量演化")}
    for tk, (mk, col, lb) in meta.items():
        pts = sorted([(r["t"], r["ms"], r["kind"]) for r in s1["rows"] if r["track"] == tk])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        meas = [p[2] == "measured" for p in pts]
        ax.plot(xs, ys, color=col, lw=0.9, zorder=2)
        ax.scatter([xx for xx, mm in zip(xs, meas) if mm],
                   [yy for yy, mm in zip(ys, meas) if mm],
                   marker=mk, s=13, facecolor=col, edgecolor=INK, linewidth=0.6,
                   zorder=3, label=lb)
        ax.scatter([xx for xx, mm in zip(xs, meas) if not mm],
                   [yy for yy, mm in zip(ys, meas) if not mm],
                   marker=mk, s=13, facecolor="white", edgecolor=col,
                   linewidth=0.7, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0, 37)
    ax.set_ylim(30, 4e5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("批次 t")
    ax.set_ylabel("单批耗时 / ms（对数坐标）")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left",
              handlelength=1.0, handletextpad=0.4, borderaxespad=0.3)
    ax.set_title("(b) 逐批耗时（实心=实测，空心=外推）", loc="left", fontsize=8.5)
    _finish(ax, grid_axis="both")

    # ---- (c) 逐种子累计加速比 ----
    ax = axes[2]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.11, 0.11, len(seeds))
    ax.axhspan(cum["ci95"][0], cum["ci95"][1], color="#e8eef6", zorder=0)
    ax.axhline(cum["mean"], color=C_MAIN, lw=1.2, zorder=1)
    ax.axhline(10, color=INK, ls="--", lw=0.7, zorder=2)
    ax.scatter(np.array(seeds) + jit, sp, s=28, facecolor="white",
               edgecolor=C_MAIN, linewidth=1.0, zorder=3)
    for s, v, j in zip(seeds, sp, jit):
        ax.text(s + j, v * 1.16, f"{v:.1f}", ha="center", fontsize=6.4, color=INK)
    ax.set_xticks(seeds)
    ax.set_xticklabels([f"{s}" for s in seeds], fontsize=7.2)
    ax.set_xlabel("随机种子")
    ax.set_ylabel("累计加速比 / ×")
    ax.set_ylim(7, 500)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.text(0.97, 0.965,
            "实线：均值 96.71×\n阴影：95% CI [68.70, 132.70]\n虚线：一个数量级判定线",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=INK)
    ax.set_title("(c) 逐种子累计加速比", loc="left", fontsize=8.5)
    _finish(ax)

    fig.subplots_adjust(wspace=0.42, left=0.075, right=0.985, top=0.855, bottom=0.185)
    _save(fig, "fig4_cost_speedup")


# ---------------------------------------------------------------- 图 6（先画，文件名按论文图号）
def fig5() -> None:
    """配对效应量森林图（论文图 6；数据：附录 B 表 B1）。"""
    rows = [
        ("漂移检测：本文 vs ADWIN", 10, 0.403, 0.337, 0.469, 1.000),
        ("漂移检测：本文 vs 增量聚类", 10, 0.491, 0.416, 0.559, 1.000),
        ("问答：稳定知识（动态−冻结）", 532, -0.0075, -0.021, 0.008, -0.333),
        ("问答：新知识（动态−冻结）", 290, 0.828, 0.803, 0.850, 0.992),
        ("问答：重定义知识（动态−冻结）", 220, 0.723, 0.659, 0.782, 0.988),
    ]
    rows = rows[::-1]

    fig, ax = plt.subplots(figsize=(W_SINGLE, 76 * MM))
    for i, (name, n, d, l, h, rb) in enumerate(rows):
        sig = not (l <= 0 <= h)
        col = C_MAIN if sig else C_FROZEN
        ax.plot([l, h], [i, i], color=INK, lw=1.0, zorder=2)
        for xx in (l, h):
            ax.plot([xx, xx], [i - 0.16, i + 0.16], color=INK, lw=1.0, zorder=2)
        ax.scatter([d], [i], s=34 + 26 * abs(rb), facecolor=col,
                   edgecolor=INK, linewidth=0.8, zorder=3)

    ax.axvline(0.0, color=INK, ls="--", lw=0.8, zorder=1)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_xlabel("配对均值差（横线为 95% CI）")
    ax.set_xlim(-0.075, 0.92)
    ax.grid(axis="x", color=C_GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.6)
    ax.set_title("配对效应量森林图", loc="left", fontsize=8.5)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=C_MAIN,
               markeredgecolor=INK, label="CI 不跨 0（差异显著）"),
        Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=C_FROZEN,
               markeredgecolor=INK, label="CI 跨 0（无显著差异）"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.155), ncol=2, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.2)

    fig.subplots_adjust(left=0.44, right=0.955, top=0.90, bottom=0.22)
    _save(fig, "fig6_forest")


# ---------------------------------------------------------------- 图 5
def fig6() -> None:
    """三类知识的问答准确率（论文图 5；三臂 × 两题干版本）。"""
    acc = {}
    for f in ("exp_qa/summary.json", "exp_qa_periodic/summary.json"):
        d = _load(f)
        for r in d["summary"]:
            acc[(r["kclass"], r["version"], r["system"])] = r["accuracy"]

    arms = [("dynamic", C_MAIN, ""), ("periodic", C_PERIOD, "//"),
            ("frozen", C_FROZEN, "\\\\")]
    pvals = {"a": {"stable": "n.s.", "new": "***", "redefined": "***"},
             "b": {"stable": "n.s.", "new": "***", "redefined": "***"}}

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 64 * MM))
    for ax, ver in zip(axes, ["a", "b"]):
        xi = np.arange(len(KCLASS_ORDER))
        w = 0.26
        for j, (arm, col, ht) in enumerate(arms):
            ys = [acc[(k, ver, arm)] for k in KCLASS_ORDER]
            b = ax.bar(xi + (j - 1) * w, ys, w, facecolor=col, hatch=ht,
                       edgecolor=INK, linewidth=0.6, zorder=3)
            for r, yv in zip(b, ys):
                ax.text(r.get_x() + r.get_width() / 2, yv + 0.015, f"{yv:.3f}",
                        ha="center", va="bottom", fontsize=5.8, color=INK)
        for i, k in enumerate(KCLASS_ORDER):
            ax.text(i, 1.14, pvals[ver][k], ha="center", va="bottom",
                    fontsize=6.8, color=INK)
        ax.set_xticks(xi)
        ax.set_xticklabels([KCLASS_CN[k] for k in KCLASS_ORDER], fontsize=8)
        ax.set_ylim(0, 1.28)
        ax.set_ylabel("问答准确率")
        ax.set_title(f"({'a' if ver == 'a' else 'b'}) 题干版本 {ver}"
                     f"（{'概念原名' if ver == 'a' else '概念改写释义'}）",
                     loc="left", fontsize=8.5)
        _finish(ax)

    handles = [Patch(facecolor=c, hatch=h, edgecolor=INK, linewidth=0.6, label=lb)
               for lb, (a, c, h) in
               [(ARM_CN["dynamic"], arms[0]), (ARM_CN["periodic"], arms[1]),
                (ARM_CN["frozen"], arms[2])]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.2, handlelength=1.3, handletextpad=0.5, columnspacing=1.6)
    fig.text(0.995, 0.012, "***：p<1e-6（Holm 校正后）　n.s.：无显著差异",
             ha="right", va="bottom", fontsize=6.2, color=INK)
    fig.subplots_adjust(wspace=0.26, left=0.062, right=0.985, top=0.845, bottom=0.205)
    _save(fig, "fig5_qa_accuracy")


def main() -> None:
    print("生成论文插图 ...")
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"完成，输出目录：{OUT}")


if __name__ == "__main__":
    main()
