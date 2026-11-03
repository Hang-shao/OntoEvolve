"""experiments/run_track4.py — 阶段 D4—D6：轨道 4 的演化闭环与检测层评测。

定位：应用可行性验证轨道。检测层真值由注入事件给出，仅用于**评测**，
不参与任何阈值标定（θ_red 用 transfer 模式，M/M_ref 线性缩放，同轨道 2）。

流程：
    1) 读 data/track4/batches（D3 物化产物）→ 嵌入 → Occurrence
    2) b01 构建 O_0（模块 A/B/C），导出 OWL 并做 HermiT 校验
    3) b02..b10：模块 D 检测 → 演化算子（涌现/重定义/淘汰）→ 三级一致性检查 → 提交/回滚
    4) 每次告警记录"该概念当批实例的类型构成"，用于 ±1 批窗的类型一致匹配
    5) 输出 main_s{seed}.csv / _summary.json / alarms_s{seed}.jsonl / detect_eval_s{seed}.json

用法：
    python experiments/run_track4.py --seed 0
    python experiments/run_track4.py --seed 0 --embedder api     # API 恢复后同口径复跑
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config as C                    # noqa: E402
from src.build_ontology import build_from_occurrences  # noqa: E402
from src.consistency import consistency_report  # noqa: E402
from src.drift_detect import DriftDetector      # noqa: E402
from src.evolve import OntologyEvolver          # noqa: E402
from src.extract import extract_candidates      # noqa: E402
from src.fraud_ontology import build_skeleton_ontology, skeleton_table  # noqa: E402
from src.fraud_stream import (FraudItem, read_materialized,  # noqa: E402
                              to_occurrences)
from src.fraud_vocab import DP_VOCAB, OP_VOCAB, PARENT_EVENT  # noqa: E402
from src.ontology_model import Ontology         # noqa: E402
from src.owl_export import export_owl           # noqa: E402

RES = ROOT / "results" / "track4"
DATA = ROOT / "data" / "track4"


def make_embedder(kind: str, verbose: bool = False):
    if kind == "api":
        from src.embedding import Embedder
        return Embedder(cache_dir=C.DIR_CACHE_API, allow_fallback=False,
                        verbose=verbose)
    from src.local_embed import LocalEmbedder
    return LocalEmbedder(cache_dir=C.DIR_CACHE_LOCAL, verbose=verbose)


def type_of_leaf(leaf: str) -> str:
    return leaf.split("·")[0]


def axi_stats(onto: Ontology) -> Dict[str, int]:
    out: Counter = Counter()
    for c in onto.concepts.values():
        for a in c.axioms:
            out[a.kind] += 1
    return dict(sorted(out.items()))


def leaf_label(it: FraudItem) -> str:
    """细粒度分析标签：类型·首个命中步骤（无命中的只用类型）。"""
    if it.steps:
        return f"{it.ftype}·{it.steps[0]}"
    return it.ftype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", choices=("local", "api"), default="local")
    ap.add_argument("--o0-build", choices=("skeleton", "cluster"), default="skeleton",
                    help="skeleton=类型×步骤骨架（默认，见 src/fraud_ontology）；"
                         "cluster=模块 A/B/C 聚类（用于复现聚类退化这一实验事实）")
    ap.add_argument("--k-fixed", type=int, default=18,
                    help="模块 A/B/C 的目标簇数；默认 18 使 O_0 落在 15—25 概念区间"
                         "（该值由论文设定的规模目标给出，未用注入真值标定）；0=自动定 K")
    ap.add_argument("--theta-red-mode", choices=("transfer", "stat"),
                    default="stat",
                    help="默认 stat=无监督 μ+zσ（真实/半合成数据无标定真值，与轨道 2 同口径）")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--min-members", type=int, default=10,
                    help="骨架构造：概念至少包含的实例数")
    ap.add_argument("--granularity", choices=("type", "type_step"), default="type",
                    help="骨架粒度：type=概念即类型（默认）；type_step=类型×主步骤（消融）")
    ap.add_argument("--adaptive-link", action="store_true",
                    help="按概念自适应链接阈值 max(link_sim, μ_c−k·σ_c)（仅轨道 4 实验用，"
                         "默认关闭以保持轨道 1—3 口径）")
    ap.add_argument("--cooldown", type=int, default=99)
    ap.add_argument("--link-offset", type=float, default=0.0,
                    help="实例链接阈值 = D_0 成员相似度 5 分位 - 该偏移（默认 0）")
    ap.add_argument("--dump-alarms", action="store_true", default=True)
    ap.add_argument("--no-owl", action="store_true")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    batches, truth, manifest = read_materialized()
    if not batches:
        print("[err] 未找到 data/track4/batches，请先运行 build_track4_stream.py")
        return 2
    print(f"[data] T={len(batches)} 每批={[len(b) for b in batches]}")

    emb = make_embedder(args.embedder, verbose=True)
    t0 = time.time()
    streams: List[List] = []
    iid_type: Dict[int, str] = {}
    iid_batch: Dict[int, int] = {}
    n = 0
    for bi, items in enumerate(batches, start=1):
        occ = to_occurrences(items, emb, start_iid=n)
        for o, it in zip(occ, items):
            o.leaf = leaf_label(it)
            iid_type[o.iid] = it.ftype
            iid_batch[o.iid] = bi
        n += len(occ)
        streams.append(occ)
    print(f"[embed] texts={n} local_calls={getattr(emb, 'n_local_calls', 0)} "
          f"api_calls={emb.n_api_calls} time={time.time()-t0:.1f}s")

    # ---- O_0 ----
    if args.o0_build == "skeleton":
        onto, info = build_skeleton_ontology(
            streams[0], min_members=args.min_members,
            op_vocab=OP_VOCAB, dp_vocab=DP_VOCAB,
            granularity=args.granularity)
    else:
        onto, info = build_from_occurrences(
            streams[0], seed=args.seed,
            k_gt=(args.k_fixed or 30),
            k_fixed=(args.k_fixed or None),
            op_vocab=OP_VOCAB, dp_vocab=DP_VOCAB)
    rep0 = consistency_report(onto, engine="auto")
    o0 = {
        "seed": args.seed, "embedder": args.embedder,
        "concepts": len(onto.concepts), "K_star": info.get("K_star"),
        "M": onto.M, "op_vocab": OP_VOCAB, "dp_vocab": DP_VOCAB,
        "axioms": axi_stats(onto),
        "axioms_total": sum(len(c.axioms) for c in onto.concepts.values()),
        "consistent": rep0["consistent"], "engine": rep0["engine"],
        "unsatisfiable": rep0.get("unsatisfiable", []),
        "link_sim_p05": info.get("link_sim_p05"),
        "link_sim_p50": info.get("link_sim_p50"),
        "concept_names": {cid: c.name for cid, c in sorted(onto.concepts.items())},
        "concept_parent": {cid: c.parent for cid, c in sorted(onto.concepts.items())},
        "build": args.o0_build,
        "concept_table": skeleton_table(onto),
    }
    (RES / "o0_stats.json").write_text(json.dumps(o0, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    if not args.no_owl:
        (RES / "owl").mkdir(exist_ok=True)
        export_owl(onto, RES / "owl" / f"o0_s{args.seed}.owl")
    print(f"[O_0] concepts={len(onto.concepts)} K*={info.get('K_star')} "
          f"axioms={o0['axioms_total']} {o0['axioms']} consistent={o0['consistent']} "
          f"link_p05={info.get('link_sim_p05')}")

    base_rate = len(streams[0]) / max(1, len(onto.concepts))
    p = C.PARAMS.copy(base_rate=base_rate,
                      link_sim=max(0.30, float(info.get("link_sim_p05", 0.5))
                                   - args.link_offset))
    print(f"[params] base_rate={base_rate:.1f} link_sim={p.link_sim:.3f} "
          f"theta_sem={p.theta_sem} theta_red={p.theta_red} M={onto.M}")

    det = DriftDetector(onto, p, cooldown=args.cooldown,
                        adaptive_link=args.adaptive_link)
    evolver = OntologyEvolver(onto)
    if args.theta_red_mode == "stat":
        det.theta_red_eff = float("inf")
    warm: List[float] = []
    print(f"[theta] theta_red_eff={det.theta_red_eff:.2f} ({args.theta_red_mode})")

    rows: List[Dict] = []
    alarms_out: List[Dict] = []
    alarm_total: Counter = Counter()

    for t, batch in enumerate(streams[1:], start=2):
        t_b = time.time()
        alarms = det.observe(t, batch)
        if args.theta_red_mode == "stat":
            warm.extend(e["signals"]["ax"] for e in det.log if e["t"] == t)
            if t == args.warmup + 1 and warm:
                mu, sd = float(np.mean(warm)), float(np.std(warm))
                det.theta_red_eff = mu + args.z * max(sd, 1e-6)
                print(f"[calib] theta_red = {mu:.2f}+{args.z}*{sd:.2f}="
                      f"{det.theta_red_eff:.2f}")
        # 告警落库（必须在演化算子改动本体之前）
        for a in alarms:
            if a.dtype == "emergence":
                items = a.detail.get("items", [])
            else:
                nf = det.last_nonempty.get(a.cid)
                items = det.last_items.get(a.cid) or (nf[1] if nf else [])
            tc = Counter(iid_type.get(o.iid, "?") for o in items)
            alarms_out.append({
                "seed": args.seed, "t": a.t, "cid": a.cid, "dtype": a.dtype,
                "score": round(float(a.score), 4),
                "signals": {k: round(float(v), 4) for k, v in a.signals.items()},
                "n_items": len(items), "types": dict(tc),
                "dominant_type": (tc.most_common(1)[0][0] if tc else None),
                "nearest_cid": a.detail.get("nearest_cid"),
                "min_dist": (round(float(a.detail["min_dist"]), 4)
                             if "min_dist" in a.detail else None),
                "size": a.detail.get("size"),
            })
        by = Counter(a.dtype for a in alarms)
        for k, v in by.items():
            alarm_total[k] += v
        # 演化算子
        for a in alarms:
            if a.dtype == "redefinition":
                evolver.redefine(a.cid, det.last_projected.get(a.cid, []))
            elif a.dtype == "extinction":
                evolver.retire(a.cid)
            elif a.dtype == "emergence":
                items = a.detail.get("items", [])
                axioms = extract_candidates(
                    a.cid, [o.as_dict() for o in items],
                    parent=PARENT_EVENT)
                leaf = Counter(o.leaf for o in items).most_common(1)[0][0]
                evolver.add_concept(a.cid, name=f"新增·{leaf}",
                                    centroid=a.detail.get("centroid"),
                                    axioms=axioms)
        rep = consistency_report(onto)
        n_ax = sum(len(c.axioms) for c in onto.concepts.values())
        rows.append({
            "batch": t, "events": len(batch), "alarms": len(alarms),
            "n_emergence": by.get("emergence", 0),
            "n_extinction": by.get("extinction", 0),
            "n_redefinition": by.get("redefinition", 0),
            "concepts_active": sum(1 for c in onto.concepts.values() if c.active),
            "concepts_total": len(onto.concepts), "axioms": n_ax,
            "consistent": int(rep["consistent"]),
            "engine": rep.get("engine"), "ms_check": round(float(rep.get("ms") or 0), 1),
            "version": onto.version,
            "seconds": round(time.time() - t_b, 2),
        })
        print(f"[b{t:02d}] n={len(batch)} alarms={len(alarms)} {dict(by)} "
              f"active={rows[-1]['concepts_active']} axioms={n_ax} "
              f"consistent={rows[-1]['consistent']} ({rows[-1]['seconds']}s)")

    # ---- O_T 校验与导出 ----
    repT = consistency_report(onto)
    if not args.no_owl:
        export_owl(onto, RES / "owl" / f"oT_s{args.seed}.owl")

    from src.onto_io import dump_onto
    dump_onto(onto, RES / f"onto_s{args.seed}.json")

    # ---- 检测层评测（批窗不对称、类型一致）----
    evals = []
    used = set()
    for ev in truth:
        tgt, dt, tb = ev["target"], ev["dtype"], ev["batch_declare"]
        wb = int(ev.get("win_before", 1))
        wa = int(ev.get("win_after", 1))
        cands = [a for a in alarms_out if a["dtype"] == dt
                 and -wb <= (a["t"] - tb) <= wa]
        hits = [a for a in cands if a["dominant_type"] == tgt]
        for a in hits:
            used.add((a["t"], a["cid"], a["dtype"]))
        evals.append({"event_id": ev["event_id"], "dtype": dt, "target": tgt,
                      "batch_declare": tb, "synthetic": ev["synthetic"],
                      "win": [-wb, wa], "hit": bool(hits),
                      "delay": (hits[0]["t"] - tb) if hits else None,
                      "hit_alarm": ({"t": hits[0]["t"], "cid": hits[0]["cid"],
                                     "signals": hits[0]["signals"]} if hits else None),
                      "n_alarms_in_window": len(cands),
                      "n_type_consistent": len(hits)})
    n_alarms = len(alarms_out)
    n_hit = sum(1 for e in evals if e["hit"])
    n_fp = sum(1 for a in alarms_out
               if (a["t"], a["cid"], a["dtype"]) not in used)
    detect = {
        "seed": args.seed, "windows": "per-event (win_before/win_after)",
        "n_truth": len(truth), "n_hit": n_hit,
        "hit_rate": round(n_hit / max(1, len(truth)), 3),
        "n_alerts": n_alarms, "n_unmatched": n_fp,
        "alert_precision_upper": round(n_hit / max(1, n_alarms), 3),
        "events": evals,
        "alarms_by_type": dict(alarm_total),
        "note": "±1 批窗、类型一致匹配；仅注入真值下评测，不构成真实无监督检测精度",
    }
    (RES / f"detect_eval_s{args.seed}.json").write_text(
        json.dumps(detect, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 落盘 ----
    if args.dump_alarms:
        with (RES / f"alarms_s{args.seed}.jsonl").open("w", encoding="utf-8") as f:
            for a in alarms_out:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
    fn = ["batch", "events", "alarms", "n_emergence", "n_extinction",
          "n_redefinition", "concepts_active", "concepts_total", "axioms",
          "consistent", "engine", "ms_check", "version", "seconds"]
    with (RES / f"main_s{args.seed}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "seed": args.seed, "embedder": args.embedder,
        "theta_red_mode": args.theta_red_mode,
        "theta_red_eff_initial": round(C.PARAMS.theta_red * max(1.0, onto.M / 18), 3),
        "config": {"k_fixed": args.k_fixed, "link_sim": round(p.link_sim, 3),
                   "base_rate": round(base_rate, 2), "cooldown": args.cooldown},
        "O_0": {"concepts": o0["concepts"], "axioms": o0["axioms_total"],
                "K_star": o0["K_star"], "consistent": o0["consistent"]},
        "alarms_total": dict(alarm_total),
        "consistency_rate": round(float(np.mean([r["consistent"] for r in rows])), 3),
        "rollback_rate": round(evolver.stats.rollback_rate, 4),
        "applied": evolver.stats.applied, "rollbacks": evolver.stats.rollbacks,
        "final_concepts": len(onto.concepts),
        "final_active": sum(1 for c in onto.concepts.values() if c.active),
        "final_axioms": sum(len(c.axioms) for c in onto.concepts.values()),
        "final_axioms_by_kind": axi_stats(onto),
        "final_consistent": repT["consistent"],
        "final_engine": repT.get("engine"),
        "unsatisfiable_final": repT.get("unsatisfiable", []),
        "detect": {k: detect[k] for k in ("n_hit", "hit_rate", "n_alerts",
                                          "n_unmatched")},
        # 每概念逐批链接实例数（失败分析用：区分"漏检"与"概念饥饿"）
        "concept_counts": {cid: list(v) for cid, v in det.counts.items()},
        "concept_names": {cid: c.name for cid, c in sorted(onto.concepts.items())},
        "ms_check_total": round(evolver.stats.ms_check, 1),
        "total_seconds": round(time.time() - t_all, 1),
        "embed": {"texts": n, "api_calls": emb.n_api_calls,
                  "local_calls": getattr(emb, "n_local_calls", 0)},
    }
    (RES / f"summary_s{args.seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== 轨道 4 汇总（seed %d）====" % args.seed)
    print(f"  告警 {dict(alarm_total)} | 一致率 {summary['consistency_rate']} "
          f"| 回退率 {summary['rollback_rate']}")
    print(f"  检测：命中 {n_hit}/{len(truth)} | 告警 {n_alarms} | 未匹配 {n_fp}")
    print(f"  末态 概念 {summary['final_concepts']}"
          f"（活跃 {summary['final_active']}）公理 {summary['final_axioms']} "
          f"{summary['final_axioms_by_kind']}")
    print(f"  总耗时 {summary['total_seconds']}s -> {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
