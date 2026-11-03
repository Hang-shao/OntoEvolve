"""轨道 2 干跑：只跑前 3 个时间片，目的是实测成本与可行性。

产出：
    1) 抽样后每片事件/实体/关系规模
    2) 嵌入调用次数与耗时（用于外推全量 24 片）
    3) O_0 的概念数、K*
    4) 前 2 片的漂移告警（按类型）与耗时
    5) 候选公理去重后的数量 → LLM 判定调用量估计（是否接 Qwen3-32B 的依据）

Run:
    python experiments/dryrun_track2.py [--slices 3] [--per-slice 1200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.build_ontology import build_from_occurrences
from src.drift_detect import Alarm, DriftDetector
from src.embedding import Embedder
from src.extract import extract_candidates
from src.icews import load_icews, make_slices, sample_slice, slice_stat
from src.icews_stream import build_icews_stream


def load_calibrated(path: Path):
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        cal = d["calibration"]
        return C.DriftParams(**cal["params"]), float(cal.get("link_offset", 0.0))
    print(f"[warn] 未找到 {path}，使用默认阈值")
    return C.PARAMS, 0.0


def unique_candidate_axioms(onto, batch: List, cid: str) -> Set[Tuple]:
    """某概念在某批上抽取出的候选公理去重键集合（用于估计 LLM 判定量）。"""
    items = [o.as_dict() for o in batch]
    axioms = extract_candidates(cid, items, parent=onto.get(cid).parent if onto.get(cid) else None)
    return {(a.kind, a.pred, a.obj) for a in axioms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=3, help="干跑的时间片数")
    ap.add_argument("--per-slice", type=int, default=1200, help="每片抽样事件数")
    ap.add_argument("--n-slices", type=int, default=24, help="总时间片数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.DIR_RESULTS / "track2_dryrun.json"))
    args = ap.parse_args()

    t_all = time.time()
    params, link_offset = load_calibrated(C.DIR_RESULTS / "exp1_summary.json")

    t0 = time.time()
    data = load_icews()
    slices = make_slices(data, args.n_slices)
    print(f"[data] entities={len(data.entities)} relations={len(data.relations)} "
          f"events={data.n_events} timestamps={len(data.timestamps)} "
          f"slices={len(slices)} ({time.time()-t0:.1f}s)")
    print(f"[data] 每片原始规模: "
          f"{[slice_stat(data, s)['events'] for s in slices[:args.n_slices]]}")

    emb = Embedder(allow_fallback=False, verbose=True)
    t1 = time.time()
    stream = build_icews_stream(data, slices, emb, per_slice=args.per_slice,
                                seed=args.seed, n_slices=args.slices)
    n_embedded = sum(len(b) for b in stream)
    t_embed = time.time() - t1
    print(f"[embed] texts={n_embedded} api_calls={emb.n_api_calls} "
          f"time={t_embed:.1f}s → 外推 {args.n_slices} 片约 "
          f"{n_embedded / max(1, args.slices) * args.n_slices:.0f} 条 / "
          f"{t_embed / max(1, args.slices) * args.n_slices / 60:.1f} min")

    # ---- O_0（第 1 片）------------------------------------------------------
    d0 = stream[0]
    t2 = time.time()
    onto, info = build_from_occurrences(d0, seed=0, k_gt=40)
    t_o0 = time.time() - t2
    print(f"[O_0] concepts={len(onto.concepts)} K*={info.get('K_star')} "
          f"link_p05={info.get('link_sim_p05', float('nan')):.3f} "
          f"time={t_o0:.1f}s")

    n_conc = max(1, len(onto.concepts))
    base_rate = len(d0) / n_conc
    p = params.copy(base_rate=base_rate,
                    link_sim=max(0.40, float(info.get("link_sim_p05", 0.9)) - link_offset))
    print(f"[cfg] 迁移阈值: base_rate={base_rate:.1f} "
          f"(轨道1={params.base_rate}) theta_red={p.theta_red} "
          f"eps_ext={p.eps_ext} W={p.window_w} delay_tol={p.delay_tol}")

    # ---- 检测 + 候选公理统计 ------------------------------------------------
    det = DriftDetector(onto, p)
    alarms: List[Alarm] = []
    seen_axioms: Set[Tuple] = set()
    per_slice_new: List[int] = []
    for t, batch in enumerate(stream[1:], start=2):
        t3 = time.time()
        alarms.extend(det.observe(t, batch))
        new_keys: Set[Tuple] = set()
        # 统计该片产生的“新”候选公理（去重后），用于估计 LLM 判定量
        assigned: Dict[str, List] = {cid: [] for cid in onto.concepts}
        for o in batch:
            v = o.vec
            best, bs = None, -2.0
            for cid, c in onto.concepts.items():
                s = float(np.dot(v, c.centroid))
                if s > bs:
                    bs, best = s, cid
            if best is not None and bs >= p.link_sim:
                assigned[best].append(o)
        for cid, lst in assigned.items():
            if lst:
                new_keys |= unique_candidate_axioms(onto, lst, cid)
        per_slice_new.append(len(new_keys - seen_axioms))
        seen_axioms |= new_keys
        print(f"[slice {t:>2}] alarms={len([a for a in alarms if a.t == t]):>3} "
              f"new_axiom_keys={per_slice_new[-1]:>4} "
              f"cum_unique={len(seen_axioms):>5} ({time.time()-t3:.1f}s)")

    by_type: Dict[str, int] = {}
    for a in alarms:
        by_type[a.dtype] = by_type.get(a.dtype, 0) + 1
    print(f"[detect] 总告警={len(alarms)} {by_type}")

    est = {
        "slices_dryrun": args.slices,
        "slices_total": args.n_slices,
        "per_slice_events": args.per_slice,
        "embedded_texts": n_embedded,
        "embed_api_calls": emb.n_api_calls,
        "embed_seconds": round(t_embed, 1),
        "embed_full_estimate_min": round(
            t_embed / max(1, args.slices) * args.n_slices / 60, 1),
        "concepts": len(onto.concepts),
        "unique_axiom_keys_per_slice": per_slice_new,
        "unique_axiom_keys_cum": len(seen_axioms),
        "llm_calls_full_estimate": round(
            len(seen_axioms) / max(1, len(per_slice_new)) * args.n_slices),
        "alarms": by_type,
        "o0_seconds": round(t_o0, 1),
        "total_seconds": round(time.time() - t_all, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(est, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n[est] 全量 {args.n_slices} 片：嵌入约 "
          f"{est['embed_full_estimate_min']} min；LLM 判定（按去重公理）约 "
          f"{est['llm_calls_full_estimate']} 次")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
