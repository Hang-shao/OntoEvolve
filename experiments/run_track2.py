"""实验 2（轨道 2）：ICEWS18 真实时序数据上的本体演化。

流程：
    1) 24 个时间片，每片按关系分层抽样；
    2) 事件文本嵌入 → 第 1 片用模块 A/B/C 构建 O_0；
    3) 后续各片：模块 D 检测 → 演化算子（新增/重定义/淘汰）→ 一致性检查 →
       不一致则回滚；
    4) 输出概念/公理演化曲线、一致率、回退率、资源消耗。

阈值沿用轨道 1 的标定结果（真实数据无注入真值，无法按 F1 标定）。

Run:
    python experiments/run_track2.py [--per-slice 1200] [--n-slices 24]
                                     [--llm topk] [--no-evolve]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.alarm_pack import record_alarm
from src.build_ontology import build_from_occurrences
from src.consistency import consistency_report
from src.drift_detect import DriftDetector
from src.embedding import Embedder
from src.evolve import OntologyEvolver
from src.extract import extract_candidates
from src.icews import load_icews, make_slices, slice_stat
from src.icews_stream import build_icews_stream
from src.llm import AxiomJudge


def load_calibrated(path: Path):
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        cal = d["calibration"]
        return C.DriftParams(**cal["params"]), float(cal.get("link_offset", 0.0))
    print(f"[warn] 未找到 {path}，使用默认阈值")
    return C.PARAMS, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slice", type=int, default=1200)
    ap.add_argument("--n-slices", type=int, default=24)
    ap.add_argument("--k-gt", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-topk", type=int, default=0,
                    help="每概念每片送大模型判定的候选公理条数；0=不接大模型")
    ap.add_argument("--llm-cache", default="",
                    help="大模型判定缓存文件；留空用默认缓存。多种子对照时请为"
                         "每次运行指定不同路径，避免跨运行命中缓存而低估调用数与耗时")
    ap.add_argument("--no-evolve", action="store_true", help="只检测不演化")
    ap.add_argument("--theta-red-mode", choices=("transfer", "stat"),
                    default="stat",
                    help="重定义阈值：transfer=按 M/M_ref 迁移轨道1标定值；"
                         "stat=无监督 μ+zσ（真实数据无真值，默认）")
    ap.add_argument("--warmup", type=int, default=6, help="stat 模式的热身片数")
    ap.add_argument("--z", type=float, default=2.0, help="stat 模式的 σ 倍数")
    ap.add_argument("--diag", action="store_true",
                    help="打印每片公理距离/置信度最高的 3 个概念，用于诊断阈值")
    ap.add_argument("--out", default=str(C.DIR_RESULTS / "track2_main.csv"))
    ap.add_argument("--summary", default=str(C.DIR_RESULTS / "track2_summary.json"))
    ap.add_argument("--dump-alarms", default="",
                    help="把每条告警的证据上下文导出为 JSONL，供人工/LLM 标注")
    args = ap.parse_args()

    t_all = time.time()
    params, link_offset = load_calibrated(C.DIR_RESULTS / "exp1_summary.json")

    data = load_icews()
    slices = make_slices(data, args.n_slices)
    print(f"[data] slices={len(slices)} "
          f"raw_per_slice={[slice_stat(data, s)['events'] for s in slices]}")

    emb = Embedder(allow_fallback=False, verbose=True)
    t1 = time.time()
    stream = build_icews_stream(data, slices, emb, per_slice=args.per_slice,
                                seed=args.seed)
    t_embed = time.time() - t1
    print(f"[embed] texts={sum(len(b) for b in stream)} "
          f"api_calls={emb.n_api_calls} time={t_embed:.1f}s")

    # 公理签名使用 ICEWS 的真实关系词表（定长维度 M = |relations| + 2 + 6）
    op_vocab = list(data.relations)
    dp_vocab = sorted({str(p) for b in stream for o in b for p, _ in o.dp_slots})
    onto, info = build_from_occurrences(stream[0], seed=0, k_gt=args.k_gt,
                                        op_vocab=op_vocab, dp_vocab=dp_vocab)
    print(f"[O_0] 签名维度 M={onto.M} (relations={len(op_vocab)}, "
          f"dp={len(dp_vocab)})")
    base_rate = len(stream[0]) / max(1, len(onto.concepts))
    p = params.copy(base_rate=base_rate,
                    link_sim=max(0.40, float(info.get("link_sim_p05", 0.9))
                                 - link_offset))
    print(f"[O_0] concepts={len(onto.concepts)} K*={info.get('K_star')} "
          f"base_rate={base_rate:.1f} link_sim={p.link_sim:.3f}")

    judge = (AxiomJudge(top_k=args.llm_topk,
                        cache_path=(args.llm_cache or None))
             if args.llm_topk > 0 else None)
    det = DriftDetector(onto, p)
    evolver = OntologyEvolver(onto) if not args.no_evolve else None

    FIELDNAMES = ["slice", "events", "alarms", "n_emergence", "n_extinction",
                  "n_redefinition", "concepts_active", "concepts_total",
                  "axioms", "consistent", "version", "seconds"]

    # 无监督阈值：热身期内先用无穷大阈值（不发重定义告警）收集 L1 距离分布，
    # 再取 μ + zσ 作为 θ_red（真实数据无注入真值，不能按 F1 标定）
    stat_mode = args.theta_red_mode == "stat"
    if stat_mode:
        det.theta_red_eff = float("inf")
    warm_dist: List[float] = []

    rows: List[Dict] = []
    alarm_total: Dict[str, int] = {}
    alarm_recs: List[Dict] = []
    for t, batch in enumerate(stream[1:], start=2):
        t0 = time.time()
        alarms = det.observe(t, batch)
        if stat_mode:
            warm_dist.extend(e["signals"]["ax"] for e in det.log if e["t"] == t)
            if t == args.warmup + 1 and warm_dist:
                mu, sd = float(np.mean(warm_dist)), float(np.std(warm_dist))
                det.theta_red_eff = mu + args.z * max(sd, 1e-6)
                print(f"[calib] 无监督 θ_red = μ+zσ = {mu:.2f}+{args.z}×{sd:.2f} "
                      f"= {det.theta_red_eff:.2f} (warmup={args.warmup} 片)")
        by_type: Dict[str, int] = {}
        for a in alarms:
            by_type[a.dtype] = by_type.get(a.dtype, 0) + 1
            alarm_total[a.dtype] = alarm_total.get(a.dtype, 0) + 1
            if args.dump_alarms:
                # 必须在演化算子修改本体之前记录：重定义需要改动前的公理，
                # 淘汰需要概念被 retire 之前的名称与历史证据
                alarm_recs.append(record_alarm(
                    track="ICEWS18", seed=args.seed, alarm=a,
                    detector=det, onto=onto))
            if evolver is None:
                continue
            if a.dtype == "redefinition":
                axioms = det.last_projected.get(a.cid, [])
                if judge is not None and axioms:
                    axioms = judge.apply(axioms, onto.get(a.cid).name
                                         if onto.get(a.cid) else a.cid)
                evolver.redefine(a.cid, axioms)
            elif a.dtype == "extinction":
                evolver.retire(a.cid)
            elif a.dtype == "emergence":
                items = a.detail.get("items", [])
                axioms = extract_candidates(
                    a.cid, [o.as_dict() for o in items], parent=None)
                if judge is not None and axioms:
                    axioms = judge.apply(axioms, f"NEW concept {a.cid}")
                evolver.add_concept(a.cid, name=f"new_concept_{a.cid}",
                                    centroid=a.detail.get("centroid"),
                                    axioms=axioms)
        if args.diag:
            top = sorted((e for e in det.log if e["t"] == t),
                         key=lambda e: -e["signals"]["ax"])[:3]
            print("      diag: " + " | ".join(
                f"{e['cid']} d={e['signals']['ax']:.1f} "
                f"conf={e['signals']['mean_conf']:.2f}" for e in top)
                + f" || θ_red={det.theta_red_eff:.1f} "
                  f"τ_conf={p.tau_conf:.2f}")
        rep = consistency_report(onto)
        n_ax = sum(len(c.axioms) for c in onto.concepts.values())
        rows.append({
            "slice": t, "events": len(batch), "alarms": len(alarms),
            "n_emergence": by_type.get("emergence", 0),
            "n_extinction": by_type.get("extinction", 0),
            "n_redefinition": by_type.get("redefinition", 0),
            "concepts_active": sum(1 for c in onto.concepts.values() if c.active),
            "concepts_total": len(onto.concepts), "axioms": n_ax,
            "consistent": int(rep["consistent"]),
            "version": onto.version,
            "seconds": round(time.time() - t0, 2),
        })
        print(f"[slice {t:>2}] events={len(batch):>5} alarms={len(alarms):>3} "
              f"{by_type} active={rows[-1]['concepts_active']:>3} "
              f"axioms={n_ax:>5} consistent={int(rep['consistent'])} "
              f"({time.time()-t0:.1f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.dump_alarms and alarm_recs:
        dp = Path(args.dump_alarms)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with dp.open("w", encoding="utf-8") as f:
            for rec in alarm_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[alarms] {len(alarm_recs)} 条已导出 -> {dp}")

    cons_rate = float(np.mean([r["consistent"] for r in rows])) if rows else 0.0
    summary = {
        "config": {"per_slice": args.per_slice, "n_slices": args.n_slices,
                   "k_gt": args.k_gt, "llm_topk": args.llm_topk,
                   "evolve": not args.no_evolve},
        "O_0": {"concepts": len(onto.concepts), "K_star": info.get("K_star"),
                "base_rate": base_rate, "link_sim": p.link_sim},
        "alarms_total": alarm_total,
        "consistency_rate": cons_rate,
        "rollback_rate": evolver.stats.rollback_rate if evolver else None,
        "rollback_detail": {"applied": evolver.stats.applied,
                            "rollbacks": evolver.stats.rollbacks} if evolver else None,
        # 逐次提交由哪一层判定（hermit / owlready2 / z3(fallback:…)），供论文口径核对
        "engines": dict(evolver.stats.engines) if evolver else None,
        "final_concepts": len(onto.concepts),
        "final_active": sum(1 for c in onto.concepts.values() if c.active),
        "final_axioms": sum(len(c.axioms) for c in onto.concepts.values()),
        "embed": {"texts": sum(len(b) for b in stream),
                  "api_calls": emb.n_api_calls, "seconds": round(t_embed, 1)},
        "llm": {"calls": judge.stats.calls, "cache_hits": judge.stats.cache_hits,
                "failures": judge.stats.failures,
                "seconds": round(judge.stats.seconds, 1)} if judge else None,
        "total_seconds": round(time.time() - t_all, 1),
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print("\n==== 轨道 2 汇总 ====")
    print(f"  告警 {alarm_total}  |  一致率 {cons_rate:.3f}  |  回退率 "
          f"{summary['rollback_rate']}")
    print(f"  概念 {len(onto.concepts)}（活跃 {summary['final_active']}）"
          f"  公理 {summary['final_axioms']}  |  总耗时 "
          f"{summary['total_seconds']}s")
    print(f"-> {out}\n-> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
