"""ontoevolve.src.build_ontology - 模块 A/B/C 构建初始本体 O_0。

调用 legacy/（DRAC 局部密度聚类 + 自动 K + 动态剪切），把聚类结果转成
Ontology（概念质心 + 公理集），并维护“叶子概念 → 本体概念”的映射，
用于把真值事件对齐到检测器所见的本体概念上。
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .controlled_stream import Occurrence, PoolItem
from .embedding import l2_normalize
from .extract import extract_candidates
from .ontology_model import Concept, Ontology


def _legacy_package_dir() -> Path:
    """返回模块 A/B/C 的包目录。

    默认是仓库内的 `legacy/`；若设了 ONTOEVOLVE_LEGACY_EXP，则用它指向的目录，
    并兼容旧的 `<dir>/src` 布局（早期把该实现放在外部工程的 src 下）。
    """
    base = Path(C.LEGACY_EXP)
    if (base / "__init__.py").exists():
        return base
    if (base / "src" / "__init__.py").exists():
        return base / "src"
    raise RuntimeError(
        f"模块 A/B/C 未找到：{base} 与 {base / 'src'} 下都没有 __init__.py；"
        "请检查 ONTOEVOLVE_LEGACY_EXP")


def load_legacy():
    """以别名加载模块 A/B/C 的包，避免与本项目 src 包重名。"""
    name = "ontoevolve_legacy"
    if name in sys.modules:
        return sys.modules[name]
    path = _legacy_package_dir()
    spec = importlib.util.spec_from_file_location(
        name, str(path / "__init__.py"), submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for sub in ("drac", "metrics", "utils", "methods"):
        importlib.import_module(f"{name}.{sub}")
    return mod


def s_star_dendrogram_gap(vectors: np.ndarray) -> float:
    """用树状图合并高度的最大跳跃估计目标簇规模 s*。

    原始公式 s* = n·(d_med/d_global)² 在近各向同性的高维嵌入空间中会严重高估
    （实测：n=300、真实簇规模 10 时估计为 59，导致模块 B 选到 K_min）。
    这里改用树状图合并高度的最大相对跳跃确定自然簇数 K_gap，再令 s* = n/K_gap。
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import pdist
    v = np.asarray(vectors, dtype=np.float64)
    n = len(v)
    if n < 6:
        return max(C.HP_S_STAR_MIN, float(n))
    Z = scipy_linkage(pdist(v, metric="cosine"), method="average")
    h = Z[:, 2]
    if len(h) < 3:
        return max(C.HP_S_STAR_MIN, float(n))
    gaps = np.diff(h)
    k_star = int(np.argmax(gaps))
    k_gap = int(np.clip(n - (k_star + 1), 2, max(2, n // 2)))
    return max(C.HP_S_STAR_MIN, float(n) / float(k_gap))


def average_linkage_result(legacy, vectors: np.ndarray):
    """把 scipy average-linkage 包装成 DRACResult，供模块 B/C 直接使用。

    模块 A（DRAC 局部密度自适应 eps）在 n≥450 时退化：d_hat 非单调且异常点被
    留到最后合并，剪切后得到“30 个单点簇 + 1 个巨型簇”（实测 NMI 0.095），
    而同一数据上 average-linkage 的 NMI=1.0。因此默认改用 average-linkage，
    模块 B（自动 K）与模块 C（动态剪切）保持不变。
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import pdist

    X = np.asarray(vectors, dtype=np.float64)
    n = len(X)
    Z = scipy_linkage(pdist(X, metric="cosine"), method="average")
    members: List = [np.array([i], dtype=int) for i in range(n)] + [None] * (n - 1)
    centres = np.zeros((2 * n - 1, X.shape[1]), dtype=np.float64)
    centres[:n] = X
    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        new = n + i
        members[new] = np.concatenate([members[a], members[b]])
        centres[new] = X[members[new]].mean(axis=0)
    return legacy.drac.DRACResult(linkage=Z, n=n, eps_dict={},
                                  members=members, centres=centres)


def run_abc(vectors: np.ndarray, k_gt: int, *, seed: int = 0,
            s_star_mode: str = "gap",
            linkage_mode: str = "average",
            k_mode: str = "auto") -> Tuple[np.ndarray, Dict]:
    """模块 A（层次连接）+ B（自动 K）+ C（动态剪切）。

    linkage_mode='average' 用 scipy average-linkage（默认，见
    average_linkage_result 的说明）；'drac' 用 DRAC 的局部 eps 连接。
    s_star_mode='gap' 用树状图跳跃估计替换原始密度公式。
    k_mode='fixed' 时关闭模块 B（直接按 k_gt 水平剪切），用于消融对照。
    """
    legacy = load_legacy()
    original = legacy.methods._target_cluster_size
    if s_star_mode == "gap":
        legacy.methods._target_cluster_size = s_star_dendrogram_gap
    try:
        return _run_abc(legacy, vectors, k_gt, seed=seed,
                        linkage_mode=linkage_mode, k_mode=k_mode)
    finally:
        legacy.methods._target_cluster_size = original


def _run_abc(legacy, vectors: np.ndarray, k_gt: int, *, seed: int = 0,
             linkage_mode: str = "average",
             k_mode: str = "auto") -> Tuple[np.ndarray, Dict]:
    X = np.asarray(vectors, dtype=np.float64)
    if linkage_mode == "average":
        drac_res = average_linkage_result(legacy, X)
        if k_mode == "fixed":
            labels = legacy.drac.horizontal_cut(drac_res.linkage, k_gt,
                                                len(X))
            return np.asarray(labels, dtype=int), {
                "k_pred": int(len(set(labels))), "K_star": k_gt,
                "timings": {}, "info_c": {}, "info_b": {}, "k_mode": "fixed"}
        k_star, info_b = legacy.methods.module_b_select_k(
            drac_res.linkage, X, drac_res)
        rng = np.random.default_rng(seed)
        labels, info_c = legacy.methods.module_c_dynamic_cut(
            drac_res, X, k_star, rng=rng)
        return np.asarray(labels, dtype=int), {
            "k_pred": int(len(set(labels))), "K_star": k_star,
            "timings": {}, "info_c": info_c, "info_b": info_b}
    res = legacy.methods.run_with_options(
        np.asarray(vectors, dtype=np.float64), k_gt,
        eps_mode="local", use_auto_k=True, use_dynamic_cut=True, seed=seed)
    info = {"k_pred": int(res.k_pred), "K_star": res.notes.get("K_star"),
            "timings": res.timings, "info_c": res.notes.get("info_c", {})}
    return np.asarray(res.labels, dtype=int), info


def items_to_occurrences(pool: Sequence[PoolItem], iids: Sequence[int]) -> List[Occurrence]:
    idx = {p.iid: p for p in pool}
    out = []
    for i in iids:
        p = idx[int(i)]
        out.append(Occurrence(iid=p.iid, leaf=p.leaf, variant=p.variant,
                              text=p.text, vec=p.vec, op_slots=list(p.op_slots),
                              dp_slots=list(p.dp_slots), sup=p.sup))
    return out


def build_initial_ontology(pool: Sequence[PoolItem], d0_iids: Sequence[int], *,
                           seed: int = 0, k_gt: int = 30,
                           run_abc_clustering: bool = True,
                           vocab: Optional[Dict] = None,
                           k_mode: str = "auto",
                           k_fixed: Optional[int] = None,
                           s_star_mode: str = "gap") -> Tuple[Ontology, Dict]:
    """由 D_0 构建 O_0。返回 (本体, 元信息)。

    元信息包含 labels / leaf_to_cid / centroids / abc 运行信息。
    """
    occ = items_to_occurrences(pool, d0_iids)
    return build_from_occurrences(occ, seed=seed, k_gt=k_gt,
                                  run_abc_clustering=run_abc_clustering,
                                  vocab=vocab, k_mode=k_mode,
                                  k_fixed=k_fixed, s_star_mode=s_star_mode)


def build_from_occurrences(occ: Sequence[Occurrence], *, seed: int = 0,
                           k_gt: int = 30, run_abc_clustering: bool = True,
                           vocab: Optional[Dict] = None, k_mode: str = "auto",
                           k_fixed: Optional[int] = None,
                           s_star_mode: str = "gap",
                           op_vocab: Optional[Sequence[str]] = None,
                           dp_vocab: Optional[Sequence[str]] = None
                           ) -> Tuple[Ontology, Dict]:
    """由一组已嵌入的 Occurrence 构建本体（轨道 1/2 共用）。

    公理签名是定长向量，其维度取决于领域属性/关系词表：轨道 2 必须显式传入
    ICEWS 的关系词表，否则真实关系会落在默认（合成）词表之外，
    导致所有签名的关系分量恒为 0、重定义永远检不出。
    """
    X = np.vstack([o.vec for o in occ]).astype(np.float64)
    if op_vocab is None:
        op_vocab = sorted({str(p) for o in occ for p, _ in o.op_slots})
    if dp_vocab is None:
        dp_vocab = sorted({str(p) for o in occ for p, _ in o.dp_slots})
    leaves = [o.leaf for o in occ]
    if run_abc_clustering:
        k_gt_use = int(k_fixed) if k_fixed is not None else k_gt
        labels, info = run_abc(X, k_gt_use, seed=seed, s_star_mode=s_star_mode,
                               k_mode=("fixed" if k_fixed is not None else k_mode))
    else:
        uniq = {lf: i for i, lf in enumerate(sorted(set(leaves)))}
        labels = np.asarray([uniq[l] for l in leaves], dtype=int)
        info = {"k_pred": len(uniq), "K_star": len(uniq), "timings": {},
                "info_c": {}, "ground_truth_labels": True}

    onto = Ontology(version=0, op_vocab=list(op_vocab), dp_vocab=list(dp_vocab))
    leaf_best: Dict[str, Tuple[str, int]] = {}
    by_label: Dict[int, List[int]] = {}
    member_sims: List[float] = []
    for i, lab in enumerate(labels):
        by_label.setdefault(int(lab), []).append(i)

    for lab, member_pos in sorted(by_label.items()):
        mem = [occ[i] for i in member_pos]
        vecs = np.vstack([o.vec for o in mem])
        centroid = l2_normalize(vecs.mean(axis=0))
        member_sims.extend(float(np.dot(o.vec, centroid)) for o in mem)
        cnt = Counter(o.leaf for o in mem)
        major_leaf, _ = cnt.most_common(1)[0]
        sup_cnt = Counter(o.sup for o in mem)
        major_sup = sup_cnt.most_common(1)[0][0]
        cid = f"C{lab}"
        schema = (vocab or {}).get(major_leaf)
        transitive_ops: List[str] = []
        disjoint: List[str] = []
        if schema is not None:
            if schema.transitive and schema.ops:
                transitive_ops = [schema.ops[0][0]]
            disjoint = list(schema.disjoint)
        axioms = extract_candidates(
            cid, [o.as_dict() for o in mem], parent=major_sup,
            transitive_ops=transitive_ops, disjoint=disjoint)
        for ax in axioms:
            # D_0 属于静态构建阶段，证据充分：置信度下限设为 tau_conf 之上
            ax.confidence = max(ax.confidence, 0.75)
        onto.add(Concept(cid=cid, name=major_leaf, parent=major_sup,
                         centroid=centroid, axioms=axioms,
                         counts=[len(mem)], weights=[1.0],
                         meta={"leaf": major_leaf, "members": member_pos}))
        for lf, n in cnt.items():
            prev = leaf_best.get(lf)
            if prev is None or n > prev[1]:
                leaf_best[lf] = (cid, n)

    leaf_to_cid = {lf: cid for lf, (cid, _) in leaf_best.items()}
    info.update({"leaf_to_cid": leaf_to_cid, "n_concepts": len(onto.concepts),
                 "n_d0": len(occ)})
    if member_sims:
        # Qwen3 嵌入的绝对余弦值整体上移（无关文本也在 0.6~0.7），
        # 实例链接阈值必须由 D_0 自身标定，而不能取固定常数。
        arr = np.asarray(member_sims, dtype=float)
        info["link_sim_p05"] = float(np.quantile(arr, 0.05))
        info["link_sim_p50"] = float(np.quantile(arr, 0.50))
        info["member_sim_mean"] = float(arr.mean())
    return onto, info
