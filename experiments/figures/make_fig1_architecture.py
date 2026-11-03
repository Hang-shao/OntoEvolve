"""英文版 图 1：一次批次内的演化回路（按 Fig1_设计规格.md 的元素与连线规则重绘）。

画布 174 mm × 64 mm（双栏宽）。图内不含题注/图号/标题，全部文字为英文。
只用 Okabe–Ito 子集中的三色：机制 #0072B2、继承 #8C8C8C、回滚 #E69F00。
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# 输出目录：默认仓库根下的 figures/；要覆盖稿件里的图，设 ONTOEVOLVE_FIG_DIR。
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

BLUE, GREY, ORANGE, INK = "#0072B2", "#8C8C8C", "#E69F00", "#1A1A1A"
LIGHT, PALE = "#F2F2F2", "#F2F7FB"
# 全图字号统一放大：期刊要求图内文字在成品尺寸下可读（≥ 7 pt）。
FS_SCALE = 1.15


def box(ax, x0, y0, w, h, *, edge=GREY, fc=LIGHT, ls="solid", lw=1.0):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.5,rounding_size=1.1",
                                linewidth=lw, edgecolor=edge, facecolor=fc, linestyle=ls))


def txt(ax, x, y, s, fs=6.5, color=INK, bold=False, ha="center", va="center", rot=0,
        pad=False):
    kw = {}
    if pad:      # 箭头标签：白底，避免压线
        kw["bbox"] = dict(facecolor="white", edgecolor="none", pad=0.8)
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs * FS_SCALE, color=color, rotation=rot,
            fontweight="bold" if bold else "normal", **kw)


def arrow(ax, p0, p1, *, color=INK, ls="solid", lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(6.9, 2.45))
    ax.set_xlim(0, 197)
    ax.set_ylim(0, 70)
    ax.axis("off")

    y0, h = 30, 20                      # 主轴
    yl = y0 + h + 2.2                   # 箭头标签所在高度（框外，避免压字）
    # 1 输入
    box(ax, 1, y0, 28, h)
    txt(ax, 15, y0 + 13.5, "$D_t$", fs=8, bold=True)
    txt(ax, 15, y0 + 7.5, "docs · stream\ntables", fs=5.8, color="#333333")
    # 2 冷启动基座
    box(ax, 37, y0, 27, h)
    txt(ax, 50.5, y0 + 12.5, "static base", fs=6.6)
    txt(ax, 50.5, y0 + 6.5, "modules A/B/C", fs=5.8, color="#333333")
    # 3 检测（内含子过程 G）
    box(ax, 72, y0 - 2, 54, h + 4, edge=BLUE, fc=PALE, lw=1.2)
    txt(ax, 99, y0 + 15, "detect drift", fs=7.2, color=BLUE, bold=True)
    txt(ax, 99, y0 + 10.5, "semantic · statistical · axiom", fs=5.6, color=BLUE)
    ax.add_patch(FancyBboxPatch((76, y0 + 1.5), 46, 6.5,
                                boxstyle="round,pad=0.4,rounding_size=0.9",
                                linewidth=0.8, edgecolor=BLUE, facecolor="white"))
    txt(ax, 99, y0 + 4.8, "G   extract $\\delta A_t$", fs=6.0, color=BLUE)
    # 4 校验
    box(ax, 134, y0, 29, h, edge=BLUE, fc=PALE, lw=1.2)
    txt(ax, 148.5, y0 + 15, "check", fs=7.2, color=BLUE, bold=True)
    txt(ax, 148.5, y0 + 8.6, "1 structural\n2 OWL-DL\n3 z3", fs=5.6, color=BLUE)
    # 5 存储
    box(ax, 170, y0, 26, h)
    txt(ax, 183, y0 + 12.5, "$O_t \\rightarrow O_{t+1}$", fs=6.4)
    txt(ax, 183, y0 + 6.5, "OWL / RDF", fs=5.8, color="#333333")

    # 主轴连线
    arrow(ax, (29, y0 + h / 2), (37, y0 + h / 2))
    txt(ax, 33, yl, "once", fs=5.4, pad=True)
    arrow(ax, (64, y0 + h / 2), (72, y0 + h / 2))
    txt(ax, 68, yl, "$O_0$", fs=5.8, pad=True)
    arrow(ax, (126, y0 + h / 2), (134, y0 + h / 2))
    txt(ax, 130, yl, "$\\delta A_t$", fs=5.8, pad=True)
    arrow(ax, (163, y0 + h / 2), (170, y0 + h / 2), color=BLUE)
    txt(ax, 166.5, yl, "commit", fs=5.2, color=BLUE, pad=True)

    # 顶部回弧：下一批
    ax.plot([183, 183, 99, 99], [y0 + h, 62, 62, y0 + h + 2.6],
            color=INK, linewidth=0.9, solid_capstyle="butt")
    arrow(ax, (99, 60), (99, y0 + h + 2.8))
    txt(ax, 150, 63.6, "next batch  t + 1", fs=6.0)

    # 底部回滚：check ⇢ 存储（保持 O_t）
    ax.plot([148.5, 148.5, 183, 183], [y0 - 0.5, 21.5, 21.5, y0 - 0.5],
            color=ORANGE, linewidth=1.0, linestyle=(0, (4, 2)))
    arrow(ax, (183, 23), (183, y0 - 0.8), color=ORANGE, ls=(0, (4, 2)))
    txt(ax, 165, 19.2, "rollback · keep $O_t$", fs=5.8, color=ORANGE)

    # F 治理（设计层，虚线）
    ax.add_patch(FancyBboxPatch((1, 4), 194, 8, boxstyle="round,pad=0.4,rounding_size=1.1",
                                linewidth=0.9, edgecolor="#B3B3B3", facecolor="#FAFAFA",
                                linestyle=(0, (4, 2))))
    txt(ax, 98, 8, "F   governance · design only", fs=7.0, color=GREY)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig1_architecture." + ext), bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("fig1 写入 English 版：", OUT / "fig1_architecture.png")


if __name__ == "__main__":
    main()
