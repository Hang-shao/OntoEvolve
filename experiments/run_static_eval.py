"""静态结构质量验证：把 README 里的 NMI / SC / DBI 数字固化为可复现的实验。

比较四种“本体 × 数据”组合下，实例被分配到概念后的聚类结构质量：

    O_0   @ D_0        初始本体在初始数据上（起点）
    O_0   @ D_all      冻结本体在全部数据上（不演化的后果）
    O_T   @ D_all      增量演化后的本体在全部数据上（我们的方法）
    O_full@ D_all      全量重构在全部数据上（质量上限参考，可选 --full）

指标（越高越好：NMI、SC；越低越好：DBI）：
    NMI   归一化互信息（分配结果 vs 数据生成时的真值 leaf 标签）
    SC    轮廓系数
    DBI   Davies-Bouldin 指数

用法：
    python experiments/run_static_eval.py
    python experiments/run_static_eval.py --full     # 含全量重构（约 +80s）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.build_ontology import build_from_occurrences          # noqa: E402
from src.controlled_stream import make_controlled_stream       # noqa: E402
from src.embedding import Embedder                             # noqa: E402

try:
    from run_exp_cost import items_to_occurrences
except Exception:                        # pragma: no cover
    from src.controlled_stream import items_to_occurrences     # type: ignore

try:
    from sklearn.metrics import (davies_bouldin_score,
                                 normalized_mutual_info_score,
                                 silhouette_score)
    _SK = True
except Exception:                        # pragma: no cover
    _SK = False


def assign(onto, occs: Sequence, link_sim: float) -> np.ndarray:
    """把实例分配到最近的概念质心；低于阈值记为 -1（未分配）。"""
    act = [c for c in onto.concepts.values() if c.active]
    if not act:
        return np.full(len(occs), -1, dtype=int)
    M = np.vstack([np.asarray(c.centroid, dtype=np.float32) for c in act])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    X = np.vstack([np.asarray(o.vec, dtype=np.float32) for o in occs])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = X @ M.T
    best = S.argmax(axis=1)
    score = S.max(axis=1)
    lab = np.asarray(best, dtype=int)
    lab[score < link_sim] = -1
    return lab


def metrics(X: np.ndarray, truth: Sequence[str],
            lab: np.ndarray, sample: int = 2000) -> Dict[str, float]:
    """NMI / SC / DBI / 纯度 / 真值覆盖率。

    口径要点：
    - **NMI 计入未分配样本**（label = -1 当作一类）。若把未分配的样本排除掉，
      冻结本体只拿“有把握”的部分去算，指标会被系统性高估，与增量不可比。
    - SC / DBI 是纯几何的聚类指标：新概念与父概念在嵌入空间本来就邻近，
      演化把它们拆开反而会让 SC 略降、DBI 略升。这两个指标对“本体是否捕获了
      新知识”不敏感，因此另加**纯度**与**真值覆盖率**——后者才是本体质量的直接体现。
    """
    if not _SK:
        return {k: float("nan") for k in ("nmi", "sc", "dbi")} | {
            "purity": float("nan"), "coverage": float("nan"),
            "assign_rate": round(float(np.mean(lab >= 0)), 3)}

    truth_s = [str(t) for t in truth]
    nmi = float(normalized_mutual_info_score(truth_s, lab))

    # 纯度：每个预测簇里多数真值标签的占比（按簇大小加权）
    tot = correct = 0
    major: Dict[int, str] = {}
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        cnt = Counter(truth_s[i] for i in idx)
        lab_major, n_major = cnt.most_common(1)[0]
        major[int(c)] = lab_major
        correct += n_major
        tot += len(idx)
    purity = correct / max(tot, 1)

    # 真值覆盖率：每个真值概念是否被某个簇“代表”（该簇的多数标签是它）
    represented = {v for k, v in major.items() if k >= 0}
    coverage = len(represented) / max(len(set(truth_s)), 1)

    m = lab >= 0
    idx = np.where(m)[0]
    out = {"nmi": round(nmi, 3), "purity": round(purity, 3),
           "coverage": round(coverage, 3),
           "assign_rate": round(float(np.mean(m)), 3)}
    if len(idx) > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=sample, replace=False)
    Xs, ls = X[idx], lab[idx]
    if len(np.unique(ls)) >= 2:
        out["sc"] = round(float(silhouette_score(Xs, ls)), 3)
        out["dbi"] = round(float(davies_bouldin_score(Xs, ls)), 3)
    else:
        out["sc"] = float("nan")
        out["dbi"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=2000,
                    help="SC/DBI 的抽样规模（O(n^2)，大样本很慢）")
    ap.add_argument("--full", action="store_true",
                    help="额外算全量重构（质量上限参考，约 +80s）")
    args = ap.parse_args()

    if not _SK:
        print("[warn] scikit-learn 不可用，NMI/SC/DBI 无法计算")

    emb = Embedder(allow_fallback=False, )
    params = C.PARAMS
    stream = make_controlled_stream(emb, seed=args.seed, params=params)
    vocab = {**stream.vocab_base, **stream.vocab_emer}
    k_gt = len(stream.vocab_base)

    d0 = list(items_to_occurrences(stream.pool, stream.d0_iids))
    allocc = list(d0)
    for b in stream.batches:
        allocc.extend(b)

    # ---------- O_0 与 O_T ----------
    from experiments.run_exp_qa import evolve
    t0 = time.time()
    onto_t, onto_0, _, _, stats = evolve(args.seed, params, emb)
    link_sim = float(stats["link_sim"])
    print(f"[evolve] {time.time() - t0:.1f}s  link_sim={link_sim:.2f}  "
          f"ops={stats['n_ops']}")

    onto_full = None
    if args.full:
        t0 = time.time()
        onto_full, _ = build_from_occurrences(allocc, seed=0, k_gt=k_gt,
                                              vocab=vocab)
        print(f"[full]   {time.time() - t0:.1f}s")

    combos: List[tuple] = [
        ("O_0 @ D_0", onto_0, d0),
        ("O_0 @ D_all (冻结)", onto_0, allocc),
        ("O_T @ D_all (增量演化)", onto_t, allocc),
    ]
    if onto_full is not None:
        combos.append(("O_full @ D_all (全量重构)", onto_full, allocc))

    rows: List[Dict] = []
    print("\n== 静态结构质量（NMI↑ / 纯度↑ / 覆盖率↑ / SC↑ / DBI↓）==")
    print(f"{'组合':<26}{'NMI':<8}{'纯度':<8}{'覆盖率':<9}{'SC':<8}"
          f"{'DBI':<8}{'分配率':<8}")
    for name, onto, data in combos:
        lab = assign(onto, data, link_sim)
        X = np.vstack([np.asarray(o.vec, dtype=np.float32) for o in data])
        truth = [o.leaf for o in data]
        m = metrics(X, truth, lab, sample=args.sample)
        rows.append({"combo": name, "n_data": len(data),
                     "n_concepts": sum(1 for c in onto.concepts.values()
                                       if c.active), **m})
        print(f"{name:<26}{m['nmi']:<8}{m['purity']:<8}{m['coverage']:<9}"
              f"{m['sc']:<8}{m['dbi']:<8}{m['assign_rate']:<8}")

    out = C.DIR_RESULTS / "static_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "static_eval.json").write_text(
        json.dumps({"seed": args.seed, "sample": args.sample,
                    "link_sim": round(link_sim, 3), "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "static_eval.csv").open("w", newline="",
                                        encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
