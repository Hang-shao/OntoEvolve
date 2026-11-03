"""命题 B 的补充：增量 vs 全量重构的**双向**概念匹配与“多出概念”的来源。

`run_exp_cost.py` 只报了单向匹配（增量概念 → 全量最近邻），结果是
“增量 27 概念 vs 全量 35 概念、匹配率 1.00”——评审会问：多出的 8 个是哪来的，
反向（全量 → 增量）也匹配吗。这个脚本把两边都算出来，并解释差异来源。

用法：
    python experiments/diag_quality_match.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                     # noqa: E402
from src.build_ontology import (build_from_occurrences,         # noqa: E402
                                build_initial_ontology)
from src.controlled_stream import make_controlled_stream        # noqa: E402
from src.embedding import Embedder                              # noqa: E402

try:                                     # 复用实验脚本里的工具函数
    from run_exp_cost import items_to_occurrences
except Exception:                        # pragma: no cover
    from src.controlled_stream import items_to_occurrences      # type: ignore


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1.0, n)


def bidirectional_match(onto_a, onto_b, thr: float = 0.9) -> dict:
    """双向最近邻匹配：A→B 与 B→A 各自的比例，以及未匹配项的名字。"""
    A = [c for c in onto_a.concepts.values() if c.active]
    B = [c for c in onto_b.concepts.values() if c.active]
    if not A or not B:
        return {"n_a": len(A), "n_b": len(B), "fwd": 0.0, "rev": 0.0,
                "mean_sim": 0.0, "unmatched_a": [], "unmatched_b": []}
    Ma = _l2(np.vstack([np.asarray(c.centroid, dtype=np.float64) for c in A]))
    Mb = _l2(np.vstack([np.asarray(c.centroid, dtype=np.float64) for c in B]))
    S = Ma @ Mb.T
    best_ab, arg_ab = S.max(axis=1), S.argmax(axis=1)
    best_ba = S.max(axis=0)
    hit_a = [i for i in range(len(A)) if best_ab[i] >= thr]
    unmatched_a = [{"name": A[i].name, "sim": round(float(best_ab[i]), 3),
                    "nearest": B[int(arg_ab[i])].name} for i in range(len(A))
                   if best_ab[i] < thr]
    matched_b = {int(arg_ab[i]) for i in hit_a}
    unmatched_b = [{"name": B[j].name, "sim": round(float(best_ba[j]), 3)}
                   for j in range(len(B)) if j not in matched_b]
    return {"n_a": len(A), "n_b": len(B),
            "fwd": round(len(hit_a) / len(A), 3),
            "rev": round(len(matched_b) / len(B), 3),
            "mean_sim": round(float(best_ab.mean()), 3),
            "unmatched_a": unmatched_a, "unmatched_b": unmatched_b}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="diag")
    args = ap.parse_args()

    emb = Embedder(allow_fallback=False, )
    params = C.PARAMS
    stream = make_controlled_stream(emb, seed=args.seed, params=params)
    vocab = {**stream.vocab_base, **stream.vocab_emer}
    k_gt = len(stream.vocab_base)

    # ---------- 1) 增量演化 ----------
    from experiments.run_exp_qa import evolve
    t0 = time.time()
    onto_inc, onto_init, _, _, stats = evolve(args.seed, params, emb)
    inc_s = time.time() - t0

    # ---------- 2) 全量重构（用截止 T 的全部数据） ----------
    occ_all = list(items_to_occurrences(stream.pool, stream.d0_iids))
    for b in stream.batches:
        occ_all.extend(b)
    t0 = time.time()
    onto_full, _ = build_from_occurrences(occ_all, seed=0, k_gt=k_gt,
                                          vocab=vocab)
    full_s = time.time() - t0

    m = bidirectional_match(onto_inc, onto_full, thr=0.9)

    # ---------- 3) 多出概念的来源 ----------
    inc_names = {c.name for c in onto_inc.concepts.values() if c.active}
    init_names = {c.name for c in onto_init.concepts.values() if c.active}
    extra = [u["name"] for u in m["unmatched_b"]]
    from_init = [n for n in extra if n in init_names]      # O_0 里有、增量已淘汰
    rest = [n for n in extra if n not in init_names]

    print("== 命题 B 质量等价补充：双向概念匹配 ==")
    print(f"增量 {m['n_a']} 概念 vs 全量 {m['n_b']} 概念  "
          f"（增量演化 {inc_s:.1f}s，全量重构 {full_s:.1f}s）")
    print(f"  增量 → 全量 匹配率 {m['fwd']:.3f}   "
          f"（增量每个概念都能在全量里找到 cos≥0.9 的对应）")
    print(f"  全量 → 增量 匹配率 {m['rev']:.3f}   "
          f"（全量概念中被增量覆盖的比例）")
    print(f"  平均相似度 {m['mean_sim']:.3f}")
    print(f"  全量多出的 {len(extra)} 个概念："
          f"其中 {len(from_init)} 个来自 O_0（增量已淘汰或合并）、"
          f"{len(rest)} 个是全量重构新产生的")
    if extra:
        print(f"    来自 O_0：{from_init[:12]}")
        print(f"    新增：    {rest[:12]}")
    print(f"  增量演化未匹配（应为空）：{m['unmatched_a']}")

    out = C.DIR_RESULTS / "exp_cost"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"quality_match_{args.tag}.json").write_text(
        json.dumps({"seed": args.seed, **m,
                    "extra_from_init": from_init, "extra_new": rest,
                    "inc_s": round(inc_s, 1), "full_s": round(full_s, 1),
                    "n_ops": stats["n_ops"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out / ('quality_match_' + args.tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
