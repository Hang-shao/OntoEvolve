# -*- coding: utf-8 -*-
"""图 2：一个漂移事件的生命周期（受控轨道的一条 stream，逐 batch 时间轴）。

这张图补的是 Fig. 1（模块与数据流）之外缺的那一层：**时间**。它把四件在正文不同
位置定义的东西画在同一根 batch 轴上——

* `t_start`：事件在数据里物理开始漂移的批次；
* `t_declare`：该事件首次满足检测器所检验判据的批次（§4.2.4 的匹配锚点，平均 +1.89 批）；
* ±3 batch 的命中窗（§4.2.4）：锚在 `t_declare`，不是 `t_start`；
* 告警落在哪一批：本方法带类型、每概念 99 批冷却（§3.4）；ADWIN 无类型。

数据全部来自既有归档，不重跑检测器、不新增实验：
* 告警：`results/alarms_cd99.jsonl`（repeat / method / t / cid / type）；
* 事件表：由 `make_controlled_stream` 确定性重建（leaf / dtype / t_start / t_declare），
  与 `results/offset_detail.json` 的 `declare_minus_start` 直方图对账；
* 匹配：复用 `experiments/analyze_match_sensitivity.py` 的贪心匹配（与 §4.2.4 同构）。

选哪一条 stream：取**本方法 macro-F1 最接近十次均值**的那次 repeat，避免挑好看的样例。
该选择写在 `results/fig2_lifecycle.json` 里，正文图注引用同一组数。

Run:
    python experiments/figures/make_fig2_lifecycle.py
    python experiments/figures/make_fig2_lifecycle.py --repeat 3   # 指定 repeat（默认自动）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src import config as C                                    # noqa: E402
from src.controlled_stream import make_controlled_stream       # noqa: E402
from analyze_match_sensitivity import (                        # noqa: E402
    _DeterministicEmbedder, greedy_match, load_alarms, load_labels, _score)

OUT = Path(os.getenv("ONTOEVOLVE_FIG_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results"
LABEL_CANDIDATES = [
    ROOT.parent / "Discover Computing" / "ESM_2" / "labels.csv",
    ROOT.parent / "ESM_2" / "labels.csv",
]

REPEATS = 10
TOL = 3
MEAN_OE, MEAN_ADWIN = 0.7782, 0.3963

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1a1a1a"
MID = "#4d4d4d"
LIGHT = "#b3b3b3"
FAINT = "#cccccc"
BAND = "#e6e6e6"
GRID = "#dcdcdc"

Y_EM, Y_EXT, Y_RED = 3.00, 2.60, 2.20
Y_OE, Y_AD = 1.15, 0.40
TIER = {"emergence": Y_EM, "extinction": Y_EXT, "redefinition": Y_RED}
MARK = {"emergence": "o", "extinction": "s", "redefinition": "^"}


def gather(repeat: int, l2c: dict, raw: dict, emb) -> dict:
    st = make_controlled_stream(emb, seed=C.PARAMS.seed + repeat, params=C.PARAMS)
    events = st.events
    out = {"repeat": repeat, "seed": C.PARAMS.seed + repeat, "events": events}
    for m in ("ontoevolve", "adwin"):
        al = raw.get((repeat, m), [])
        used = greedy_match(events, al, l2c, m == "ontoevolve", False, TOL)
        f1, tp, fp = _score(events, al, used, l2c)
        out[m] = {"alarms": al, "used": used, "f1": float(f1), "tp": tp, "fp": fp,
                  "n_hits": sum(1 for j in used if j is not None)}
    return out


def draw(d: dict) -> None:
    events, alarms = d["events"], d["ontoevolve"]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0.2, 36.8)
    ax.set_ylim(0.02, 3.62)
    ax.set_yticks([Y_EM, Y_EXT, Y_RED, Y_OE, Y_AD])
    ax.set_yticklabels(["emergence", "extinction", "redefinition",
                        "OntoEvolve", "ADWIN"], fontsize=9)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel("batch (t)", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    for y in (Y_OE, Y_AD):
        ax.axhline(y, color=GRID, linewidth=0.5, zorder=0)

    # --- 真值：±3 命中窗、t_start 刻线、t_declare 标记（形状即类型）
    for ev in events:
        y = TIER[ev.dtype]
        ax.add_patch(Rectangle((ev.t_declare - TOL - 0.5, y - 0.135), 2 * TOL + 1, 0.27,
                               facecolor=BAND, edgecolor="none", zorder=1))
        ax.plot([ev.t_start, ev.t_start], [y - 0.20, y + 0.20], color=MID, lw=1.0, zorder=3)
        ax.plot([ev.t_declare], [y], marker=MARK[ev.dtype], ms=5.0, mfc=INK, mec=INK,
                zorder=4)

    # --- 本方法：命中画带类型实心标记并连线到对应事件，未命中画 ×
    for j, a in enumerate(alarms["alarms"]):
        if not (0.5 <= a.t <= 36.5):
            continue
        if alarms["used"].count(j):
            ev = events[alarms["used"].index(j)]
            ax.plot([a.t, ev.t_declare], [Y_OE + 0.09, TIER[ev.dtype] - 0.24],
                    color=FAINT, lw=0.6, zorder=2)
            ax.plot([a.t], [Y_OE], marker=MARK.get(a.dtype, "o"), ms=5.0,
                    mfc=INK, mec=INK, zorder=5)
        else:
            ax.plot([a.t], [Y_OE], marker="x", ms=4.8, mec=MID, mew=1.1, zorder=5)

    # --- ADWIN：无类型；实心=命中，空心=未命中（误报）
    for j, a in enumerate(d["adwin"]["alarms"]):
        if not (0.5 <= a.t <= 36.5):
            continue
        hit = d["adwin"]["used"].count(j) > 0
        ax.plot([a.t], [Y_AD], marker="s", ms=4.4, mfc=INK if hit else "none",
                mec=INK, mew=0.9, zorder=5)

    # --- 只留一处标注：窗口锚在 t_declare
    last_em = max((e for e in events if e.dtype == "emergence"), key=lambda e: e.t_declare)
    ax.annotate("\u00b13-batch hit window,\nanchored at t_declare (Section 4.2.4)",
                xy=(last_em.t_declare + TOL + 0.5, Y_EM + 0.14),
                xytext=(last_em.t_declare + TOL + 1.2, Y_EM + 0.40),
                fontsize=8.2, color=MID, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=LIGHT,
                                connectionstyle="arc3,rad=0.0"))

    handles = [
        Patch(facecolor=BAND, edgecolor="none", label="hit window"),
        Line2D([], [], color=MID, lw=1.0, label="t_start"),
        Line2D([], [], color=INK, marker="o", ls="none", ms=5.0, label="t_declare"),
        Line2D([], [], color=FAINT, lw=1.0, label="matched alarm"),
        Line2D([], [], color=MID, marker="x", ls="none", ms=4.8, label="unmatched"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.40),
              ncol=5, frameon=False, fontsize=8.4, handletextpad=0.4,
              columnspacing=1.6)
    fig.tight_layout()
    png, pdf = OUT / "fig2_lifecycle.png", OUT / "fig2_lifecycle.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png.name, "and", pdf.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=None,
                    help="指定 repeat；默认取本方法 macro-F1 最接近十次均值的那次")
    ap.add_argument("--alarms", default=str(RES / "alarms_cd99.jsonl"))
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()

    lab = Path(args.labels) if args.labels else next(
        (p for p in LABEL_CANDIDATES if p.exists()), None)
    if lab is None or not lab.exists():
        raise SystemExit("找不到 ESM 2 的 labels.csv；用 --labels 指定")
    l2c = load_labels(lab)
    raw = load_alarms(Path(args.alarms))
    emb = _DeterministicEmbedder()

    # ---- 自检 1：事件表与存档的 t_declare − t_start 分布一致
    streams = {r: make_controlled_stream(emb, seed=C.PARAMS.seed + r, params=C.PARAMS)
               for r in range(REPEATS)}
    ref = json.loads((RES / "offset_detail.json").read_text(encoding="utf-8"))
    gaps = [int(e.t_declare - e.t_start) for st in streams.values() for e in st.events]
    mine = {str(k): gaps.count(k) for k in sorted(set(gaps))}
    ref_hist = {k: v for k, v in ref["declare_minus_start"]["hist"].items() if v}
    if mine != ref_hist:
        raise SystemExit(f"事件表重建与存档不符：{mine} != {ref_hist}")
    mean_gap = round(sum(gaps) / len(gaps), 3)

    # ---- 自检 2：贪心口径复现正文引用的均值
    f1 = {}
    for m in ("ontoevolve", "adwin"):
        vals = []
        for r in range(REPEATS):
            al = raw.get((r, m), [])
            used = greedy_match(streams[r].events, al, l2c, m == "ontoevolve", False, TOL)
            vals.append(_score(streams[r].events, al, used, l2c)[0])
        f1[m] = round(sum(vals) / len(vals), 4)
    for m, want in (("ontoevolve", ref["ontoevolve"]["sym_f1_mean"]),
                    ("adwin", ref["adwin"]["sym_f1_mean"])):
        if abs(f1[m] - want) > 5e-4:
            raise SystemExit(f"{m} 均值未能复现存档值：{f1[m]} != {want}")

    # ---- 选 repeat：本方法 F1 最接近十次均值
    per = []
    for r in range(REPEATS):
        ev = streams[r].events
        used = greedy_match(ev, raw.get((r, "ontoevolve"), []), l2c, True, False, TOL)
        per.append(_score(ev, raw.get((r, "ontoevolve"), []), used, l2c)[0])
    pick = args.repeat if args.repeat is not None else min(
        range(REPEATS), key=lambda r: abs(per[r] - f1["ontoevolve"]))

    d = gather(pick, l2c, raw, emb)
    draw(d)

    payload = {
        "protocol": {"alarms": Path(args.alarms).name, "delay_tol": TOL,
                     "labels": lab.name, "repeats": REPEATS,
                     "choice_rule": "repeat whose present-method macro-F1 is closest to "
                                    "the ten-repeat mean"},
        "means": {"ontoevolve": f1["ontoevolve"], "adwin": f1["adwin"]},
        "selfcheck": {"declare_minus_start": mine, "declare_minus_start_mean": mean_gap,
                      "matches_offset_detail": True},
        "chosen": {
            "repeat": pick, "seed": d["seed"],
            "ontoevolve": {"macro_f1": round(d["ontoevolve"]["f1"], 4),
                           "alarms": len(d["ontoevolve"]["alarms"]),
                           "hits": d["ontoevolve"]["n_hits"],
                           "unmatched": len(d["ontoevolve"]["alarms"]) - d["ontoevolve"]["n_hits"]},
            "adwin": {"macro_f1": round(d["adwin"]["f1"], 4),
                      "alarms": len(d["adwin"]["alarms"]),
                      "hits": d["adwin"]["n_hits"],
                      "unmatched": len(d["adwin"]["alarms"]) - d["adwin"]["n_hits"]},
            "events": [{"type": e.dtype, "leaf": e.leaf,
                        "t_start": int(e.t_start), "t_declare": int(e.t_declare)}
                       for e in d["events"]],
            "alerts_ontoevolve": [(int(a.t), a.cid, a.dtype)
                                  for a in d["ontoevolve"]["alarms"]],
            "alerts_adwin": [int(a.t) for a in d["adwin"]["alarms"]],
        },
    }
    dest = RES / "fig2_lifecycle.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[selfcheck] declare−start = {mine}, mean {mean_gap}")
    print(f"[selfcheck] means: OntoEvolve {f1['ontoevolve']}, ADWIN {f1['adwin']}")
    print(f"[pick] repeat {pick} (seed {d['seed']}), Ours macro-F1 "
          f"{payload['chosen']['ontoevolve']['macro_f1']}")
    print("wrote", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
