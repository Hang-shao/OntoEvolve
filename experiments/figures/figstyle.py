# -*- coding: utf-8 -*-
"""图件统一配色与排版（与 Fig. 1 的深蓝/绿/橙体系一致）。

设计约束
--------
* 颜色全部取 Fig. 1 已用的蓝—绿—橙家族，外加一个紫、一个青作为第六条方法的区分色；
* **灰度可印**：每条曲线/每组柱都另有 marker 形状、线型或 hatch，去掉颜色仍能区分
  （Springer 要求图在黑白下仍可读）；
* 字号下限 8 pt（期刊 8–12 pt），字体用 Arial（无衬线，符合投稿须知）；
* 不用图内标题，面板标签 (a)/(b) 由各脚本自己写。

用法::

    from figstyle import METHOD_STYLE, METHOD_LABEL, INK, GRID, use, axis_style
    use()
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 调色板
NAVY = "#1F4E79"        # 主方法 / 强调
BLUE = "#2E75B6"        # 次级蓝（面板标题、参考线）
SKY = "#9DC3E6"
PALE = "#DEEBF7"
GREEN = "#548235"
GREEN_L = "#C5E0B4"
ORANGE = "#C55A11"
ORANGE_L = "#F4CCAC"
PURPLE = "#7030A0"
PURPLE_L = "#D5C3E6"
TEAL = "#2E9BA6"
TEAL_L = "#B7DEE2"
GREY = "#8C8C8C"
GREY_L = "#D9D9D9"
INK = "#1A1A1A"
GRID = "#E6E6E6"
BAND = "#F2F6FA"

METHODS = ["ontoevolve", "adwin", "inc_clustering", "page_hinkley",
           "embdist", "embkmeans"]

METHOD_LABEL = {
    "ontoevolve": "OntoEvolve (ours)",
    "adwin": "ADWIN",
    "inc_clustering": "Incremental clustering",
    "page_hinkley": "Page–Hinkley",
    "embdist": "Embedding distribution",
    "embkmeans": "Embedding k-means",
}

# 曲线样式：颜色之外，marker 与线型也互不相同
METHOD_STYLE = {
    "ontoevolve":     dict(color=NAVY,   marker="o", ms=4.4, ls="solid", lw=1.9, zorder=6),
    "adwin":          dict(color=ORANGE, marker="s", ms=3.8, ls=(0, (4.5, 1.6)), lw=1.5, zorder=5),
    "inc_clustering": dict(color=GREEN,  marker="^", ms=3.8, ls=(0, (6, 1.6, 1.4, 1.6)), lw=1.3, zorder=4),
    "page_hinkley":   dict(color=PURPLE, marker="v", ms=3.6, ls=(0, (1.4, 1.3)), lw=1.3, zorder=3),
    "embdist":        dict(color=TEAL,   marker="D", ms=3.2, ls=(0, (5.5, 1.2, 1.2, 1.2)), lw=1.1, zorder=2),
    "embkmeans":      dict(color=GREY,   marker="x", ms=3.6, ls=(0, (2.2, 1.4)), lw=1.1, zorder=1),
}

# 柱状样式：淡色填充 + hatch（灰度下靠 hatch 区分）
METHOD_BAR = {
    "ontoevolve":     dict(color=NAVY,     hatch="",     zorder=6),
    "adwin":          dict(color=ORANGE_L, hatch="//",   zorder=5),
    "inc_clustering": dict(color=GREEN_L,  hatch="\\\\", zorder=4),
    "page_hinkley":   dict(color=PURPLE_L, hatch="..",   zorder=3),
    "embdist":        dict(color=SKY,      hatch="xx",   zorder=2),
    "embkmeans":      dict(color=GREY_L,   hatch="",     zorder=1),
}

# 维护策略（下游问答、反诈轨）
SCHEME_STYLE = {
    "Dynamic":          dict(color=NAVY,     hatch=""),
    "Periodic":         dict(color=SKY,      hatch="//"),
    "Frozen":           dict(color=GREY_L,   hatch=".."),
    "LLM regeneration": dict(color=ORANGE_L, hatch="xx"),
}

SCHEME_LABEL = {
    "Dynamic": "Dynamic evolution",
    "Periodic": "Periodic rebuild",
    "Frozen": "Frozen",
    "LLM regeneration": "LLM regeneration",
}

# 漂移类型
TYPE_COLOR = {"emergence": BLUE, "extinction": ORANGE, "redefinition": GREEN}
TYPE_FILL = {"emergence": PALE, "extinction": "#FBEDE5", "redefinition": "#EAF1E2"}
TYPE_MARK = {"emergence": "o", "extinction": "s", "redefinition": "^"}
TYPE_LABEL = {"emergence": "Emergence", "extinction": "Extinction",
              "redefinition": "Redefinition"}


def use(font_size: float = 8.5) -> None:
    """统一 rcParams：Arial、8.5 pt 基准、无顶/右边框、600 dpi 输出。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": font_size + 1.0,
        "axes.titleweight": "normal",
        "axes.labelsize": font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.fontsize": font_size - 1.0,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "hatch.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def axis_style(ax, *, ygrid: bool = True, xgrid: bool = False) -> None:
    """浅网格、网格置底、去掉多余刻度线。"""
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.tick_params(length=2.5, width=0.7)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)


def panel_title(ax, text: str, *, pad: float = 6.0) -> None:
    """左对齐的面板标签，不用粗体（图内不放粗重标题）。"""
    ax.set_title(text, loc="left", pad=pad, fontsize=plt.rcParams["font.size"] + 0.5)
