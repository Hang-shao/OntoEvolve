# -*- coding: utf-8 -*-
"""实验：模块 G 的 OWL 2 DL 合法性与公理增长（论文中原标注为“未实施”的量化）。

回答两个此前缺失的量化问题：

  1) **OWL 合法性通过率**：模板化公理抽取产生的公理，在导出成 OWL 2 DL 时
     有多少能够合法保留。导出侧受 OWL 2 DL "基数限制只能作用于 simple
     property" 的硬约束，同一属性上同时出现 TRANSITIVE 与 CARD 时必须放弃
     传递性声明。冲突次数由 ontology_to_graph 的 notes 记录，据此算通过率。
  2) **公理爆炸与否**：随演化推进，公理总量对版本数做线性拟合；若线性拟合
     R² 高且每版本增量平稳，则说明增长受控而非爆炸。

每个演化版本还经 HermiT 做一次一致性检查（真实解析 + 判定），报告解析成功率
与一致率——这是“导出的 OWL 确实能被 OWL 2 DL 推理机读取”的直接证据。

用法：
    python experiments/run_exp_gowl.py --seed 0
    python experiments/run_exp_gowl.py --quick          # 冒烟
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.build_ontology import build_initial_ontology          # noqa: E402
from src.consistency import consistency_report                 # noqa: E402
from src.controlled_stream import make_controlled_stream       # noqa: E402
from src.drift_detect import DriftDetector                     # noqa: E402
from src.embedding import Embedder                             # noqa: E402
from src.evolve import OntologyEvolver                         # noqa: E402
from src.extract import extract_candidates                     # noqa: E402
from src.owl_export import export_owl, ontology_to_graph       # noqa: E402
from src.reasoner import find_java, get_reasoner               # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

FIELDS = ["version", "t", "n_concepts", "n_active", "n_axioms", "n_skipped",
          "legal_rate", "owl_ok", "owl_consistent", "owl_engine", "owl_ms",
          "unsat_n", "kinds"]


def onto_stats(onto) -> dict:
    """公理构成 + 导出侧合法性过滤统计。"""
    kinds: Counter = Counter()
    for c in onto.concepts.values():
        for ax in c.axioms:
            kinds[ax.kind] += 1
    notes: list = []
    ontology_to_graph(onto, notes=notes)      # 仅用于收集导出侧的跳过记录
    skipped = sum(1 for n in notes if str(n).startswith("skip"))
    total = sum(kinds.values())
    return {
        "n_concepts": len(onto.concepts),
        "n_active": len([c for c in onto.concepts.values() if c.active]),
        "n_axioms": total,
        "n_skipped": skipped,
        "legal_rate": round(1.0 - skipped / total, 4) if total else 1.0,
        "kinds": dict(kinds),
        "notes": notes,
        "M": getattr(onto, "M", None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-every", type=int, default=1,
                    help="每多少个演化版本做一次 OWL 导出 + HermiT 检查")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    base = C.PARAMS
    if args.quick:
        base = base.copy(n_batch=12, n_init_per_leaf=6, pool_per_leaf=12,
                         base_rate=3.0, n_emergence=1, n_extinction=1,
                         n_redefinition=1)

    r = get_reasoner()
    print("java:", find_java())
    print("reasoner:", (r.name if r else None))

    emb = Embedder(allow_fallback=False)
    stream = make_controlled_stream(emb, seed=args.seed, params=base)
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    p = base.copy(link_sim=max(0.50, float(info.get("link_sim_p05",
                                                    base.link_sim)) - 0.02))
    det = DriftDetector(onto, p)
    evo = OntologyEvolver(onto)

    owls = C.DIR_ONTOLOGY / f"exp_gowl_seed{args.seed}"
    rows = []
    last_versions = evo.stats.versions          # 初始本体作为一个版本基线
    new_cid = len(onto.concepts)
    t0 = time.perf_counter()

    # 版本 V0（初始本体 O_0）
    st0 = onto_stats(onto)
    rows.append({"version": 0, "t": 0, "n_concepts": st0["n_concepts"],
                 "n_active": st0["n_active"], "n_axioms": st0["n_axioms"],
                 "n_skipped": st0["n_skipped"],
                 "legal_rate": st0["legal_rate"], "owl_ok": None,
                 "owl_consistent": None, "owl_engine": None, "owl_ms": None,
                 "unsat_n": None, "kinds": json.dumps(st0["kinds"],
                                                      ensure_ascii=False)})

    for t, batch in enumerate(stream.batches, start=1):
        alarms = det.observe(t, batch)
        for a in alarms:
            if a.dtype == "emergence":
                new_cid += 1
                cid = f"C{new_cid}"
                items = list(a.detail.get("items", []))
                if not items:
                    continue
                axs = extract_candidates(cid, [o.as_dict() for o in items],
                                         parent=None)
                cen = a.detail.get("centroid")
                ok = evo.add_concept(
                    cid, a.cid,
                    centroid=np.asarray(cen, dtype=np.float32) if cen is not None
                    else np.mean([o.vec for o in items], axis=0),
                    axioms=axs)
                if ok and cid in onto.concepts:
                    det.centroids[cid] = np.asarray(
                        onto.concepts[cid].centroid, dtype=np.float32)
                    det.counts[cid] = []
                    det.weights[cid] = []
            elif a.dtype == "redefinition":
                axs = det.last_projected.get(a.cid) or []
                if axs:
                    evo.redefine(a.cid, axs)
            elif a.dtype == "extinction":
                evo.retire(a.cid)

        # 版本推进后做一次快照（含真实 OWL 导出 + HermiT 判定）
        if evo.stats.versions == last_versions:
            continue
        last_versions = evo.stats.versions
        if args.check_every > 1 and last_versions % args.check_every:
            continue
        st = onto_stats(onto)
        owl = export_owl(onto, owls / f"v{last_versions:03d}.owl")
        rep = consistency_report(onto, engine="hermit", owl_out=owl,
                                 use_cache=False)
        rec = {"version": last_versions, "t": t,
               "n_concepts": st["n_concepts"], "n_active": st["n_active"],
               "n_axioms": st["n_axioms"], "n_skipped": st["n_skipped"],
               "legal_rate": st["legal_rate"],
               "owl_ok": bool(rep.get("ok", True)),
               "owl_consistent": bool(rep.get("consistent")),
               "owl_engine": rep.get("engine"),
               "owl_ms": round(float(rep.get("ms", 0.0)), 1),
               "unsat_n": len(rep.get("unsatisfiable", []) or []),
               "kinds": json.dumps(st["kinds"], ensure_ascii=False)}
        rows.append(rec)
        print("v%-3d t=%-3d axioms=%-4d skipped=%d legal=%.4f OWL=%s(%s) %.0fms"
              % (last_versions, t, st["n_axioms"], st["n_skipped"],
                 st["legal_rate"], rec["owl_consistent"], rec["owl_engine"],
                 rec["owl_ms"]), flush=True)

    el = time.perf_counter() - t0
    checked = [x for x in rows if x["owl_ok"] is not None]
    n_ok = sum(1 for x in checked if x["owl_ok"])
    n_cons = sum(1 for x in checked if x["owl_consistent"])
    tot_ax = sum(x["n_axioms"] for x in checked)
    tot_skip = sum(x["n_skipped"] for x in checked)

    # 公理增长：对版本数做线性拟合，判断是否爆炸
    vs = np.array([x["version"] for x in checked], dtype=float)
    ax = np.array([x["n_axioms"] for x in checked], dtype=float)
    growth = None
    if vs.size >= 3:
        A = np.vstack([vs, np.ones_like(vs)]).T
        coef, *_ = np.linalg.lstsq(A, ax, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((ax - pred) ** 2))
        ss_tot = float(np.sum((ax - ax.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        growth = {"slope_axioms_per_version": round(float(coef[0]), 3),
                  "intercept": round(float(coef[1]), 3),
                  "r2_linear": round(r2, 4),
                  "n_versions": int(vs.size),
                  "axioms_first": int(ax[0]), "axioms_last": int(ax[-1])}

    summary = {
        "seed": args.seed, "tag": args.tag, "quick": args.quick,
        "reasoner": (r.name if r else None),
        "n_versions_checked": len(checked),
        "owl_parse_ok_rate": round(n_ok / len(checked), 4) if checked else None,
        "owl_consistent_rate": round(n_cons / len(checked), 4) if checked else None,
        "owl_unsat_total": int(sum(int(x["unsat_n"] or 0) for x in checked)),
        "owl_ms_mean": round(float(np.mean([x["owl_ms"] for x in checked])), 1)
        if checked else None,
        "owl_ms_max": round(float(np.max([x["owl_ms"] for x in checked])), 1)
        if checked else None,
        "legal_rate_pooled": round(1.0 - tot_skip / tot_ax, 4) if tot_ax else None,
        "axioms_pooled": tot_ax, "skipped_pooled": tot_skip,
        "legal_rate_final": rows[-1]["legal_rate"],
        "final_axioms": rows[-1]["n_axioms"],
        "final_kinds": rows[-1]["kinds"],
        "growth": growth,
        "evolution": {"events": len(stream.events), "versions": evo.stats.versions,
                      "applied": dict(evo.stats.applied),
                      "rollbacks": dict(evo.stats.rollbacks),
                      "rollback_rate": evo.stats.rollback_rate,
                      "inconsistent_versions": evo.stats.inconsistent_versions},
        "elapsed_s": round(el, 1),
        "rows": rows,
    }

    out = C.DIR_RESULTS / "exp_gowl"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"exp_gowl_{args.tag}"
    (out / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("\n== 模块 G 的 OWL 2 DL 合法性与公理增长 ==")
    print(f"检查版本数 {summary['n_versions_checked']}，"
          f"解析成功 {summary['owl_parse_ok_rate']}，"
          f"一致 {summary['owl_consistent_rate']}，"
          f"不可满足类 {summary['owl_unsat_total']}")
    print(f"HermiT 耗时 均值 {summary['owl_ms_mean']} ms / 最大 "
          f"{summary['owl_ms_max']} ms")
    print(f"OWL 2 DL 合法性通过率（汇总）{summary['legal_rate_pooled']} "
          f"（{summary['skipped_pooled']}/{summary['axioms_pooled']} 条因 "
          f"simple property 约束放弃传递性声明）；最终版本 "
          f"{summary['legal_rate_final']}，公理 {summary['final_axioms']} 条")
    if growth:
        print(f"公理增长：每版本 +{growth['slope_axioms_per_version']} 条，"
              f"线性拟合 R²={growth['r2_linear']}，"
              f"{growth['axioms_first']} → {growth['axioms_last']}")
    print(f"\n结果写入 {out / (stem + '.json')}（用时 {el:.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
