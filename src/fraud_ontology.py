"""ontoevolve.src.fraud_ontology - 轨道 4 的 O_0 骨架构造（类型—步骤两层）。

为什么不复用模块 A/B/C 的嵌入聚类：
    在中文电诈话术上实测（data/track4/audit/d4_ksweep.txt），平均连接聚类在任何
    K（7—18）下都退化为"一个巨型簇（~200/240 成员）+ 若干 ≤2 成员的离群簇"，
    K=18 时 13 个簇只有 1—2 个成员。此时簇质心几乎等于单个句子的向量，
    实例链接与淘汰判定全部失真。该退化作为实验事实记录，不作为轨道 4 的构建方式。

轨道 4 因此用**确定性骨架**构造 O_0（可复现、不依赖随机种子、可解释）：
    概念 = 类型 × 主要话术步骤（b01 中该类型出现次数 ≥ min_members 的步骤）
    公理 = OP(该概念成员覆盖到的其余步骤) + DP(media/audience) + DOMAIN(诈骗事件)
           + RANGE(话术步骤) + CARD(步骤数区间) + DISJOINT(不同诈骗类型互斥)
这一构造只替换"概念的来源"，抽取与一致性检查仍复用 src.extract / src.consistency。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .controlled_stream import Occurrence
from .extract import extract_candidates
from .fraud_vocab import PARENT_EVENT, steps_of
from .ontology_model import Axiom, Concept, Ontology


def _type_of_leaf(leaf: str) -> str:
    return leaf.split("·")[0]


def build_skeleton_ontology(occ: Sequence[Occurrence], *,
                           min_members: int = 8,
                           op_vocab: Sequence[str] = (),
                           dp_vocab: Sequence[str] = (),
                           version: int = 0,
                           disjoint_types: bool = True,
                           granularity: str = "type") -> Tuple[Ontology, Dict]:
    """由一批实例构造"类型×步骤"骨架本体。"""
    # 每条实例的 (类型, 步骤集合)
    per_item: List[Tuple[Occurrence, str, List[str]]] = []
    for o in occ:
        t = _type_of_leaf(o.leaf)
        st = [s for s in steps_of(o.text)]
        per_item.append((o, t, st))

    # 只对"诈骗类型"建概念（噪声类型不建）
    # 按"主步骤"做**互斥划分**：每条实例只进入一个概念。
    # 若允许一条实例同时进入多个 (类型, 步骤) 概念（多标签），概念质心会高度重叠，
    # 最近质心分配下大量概念长期拿不到实例（"概念饥饿"），淘汰判定会把饥饿误判为
    # 淘汰。互斥划分 + 公理仍按实例的全部步骤抽取，可同时避免饥饿与信息损失。
    # granularity="type"（默认）：概念 = 类型，公理覆盖该类型实例的全部步骤；
    # granularity="type_step"：概念 = 类型 × 主步骤（更细，作为消融对照）。
    # 采用类型级的理由：类型×步骤会切出 10—78 个成员的窄概念，同一类型的其余实例
    # 大量落不到任何概念上（实测 b02—b03 未链接实例 363 条，触发簇规模 363 的假涌现），
    # 淘汰判定随之失去分母。两种粒度均已在实验记录中报告。
    groups: Dict[Tuple[str, str], List[Occurrence]] = defaultdict(list)
    tcount: Counter = Counter()
    for o, t, st in per_item:
        tcount[t] += 1
        if granularity == "type":
            groups[(t, "")].append(o)
            continue
        if not st:
            continue                     # 无步骤命中的实例不进入 O_0（保持未链接）
        groups[(t, st[0])].append(o)

    kept = {k: v for k, v in groups.items() if len(v) >= min_members}
    # 每个类型至少保证 1 个概念（取成员最多者）
    by_type: Dict[str, List[Tuple[str, List[Occurrence]]]] = defaultdict(list)
    for (t, s), v in kept.items():
        by_type[t].append((s, v))
    # b01 中出现的每个类型都必须在本体里有落点：若该类型的所有主步骤组都被
    # min_members 滤掉，退回成员最多的那一组。原实现把该判断写成 `if not lst`，
    # 而 by_type 里收录的类型 lst 恒非空，条件恒假，回退从不生效。本轮反诈数据上
    # 每个类型都已有 ≥min_members 的主步骤组，故修正前后结果相同；保留修正是为了
    # 在类型成员数不足时不让整类丢失（整类丢失会使该类实例长期"未链接"，被涌现
    # 检测当成新概念反复告警）。
    for t in {t for (t, _s) in groups}:
        if by_type.get(t):
            continue
        best = max(((s, v) for (tt, s), v in groups.items() if tt == t),
                   key=lambda kv: len(kv[1]))
        by_type[t] = [best]

    onto = Ontology(version=version, op_vocab=op_vocab, dp_vocab=dp_vocab)
    type_cids: Dict[str, List[str]] = defaultdict(list)
    for t, lst in sorted(by_type.items()):
        for step, members in sorted(lst, key=lambda kv: -len(kv[1])):
            cid = f"C{len(onto.concepts):02d}"
            vecs = np.vstack([o.vec for o in members])
            cen = vecs.mean(axis=0)
            cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
            axioms = extract_candidates(cid, [o.as_dict() for o in members],
                                        parent=PARENT_EVENT)
            for ax in axioms:
                ax.confidence = max(ax.confidence, 0.75)   # 静态构建阶段：证据充分
            _name = f"{t}·{step}" if step else t
            # 成员到质心的相似度分布：供"按概念自适应链接阈值"使用（不同概念的
            # 成员紧致度差异大，全局单一阈值会让松散概念吞掉邻近新类型的实例）。
            _sims = np.asarray([float(np.dot(o.vec, cen)) for o in members])
            onto.add(Concept(cid=cid, name=_name, parent=PARENT_EVENT,
                             centroid=cen.astype(np.float32), axioms=axioms,
                             counts=[len(members)], weights=[1.0],
                             meta={"type": t, "step": step,
                                   "n_members": len(members),
                                   "sim_mu": float(_sims.mean()),
                                   "sim_sigma": float(_sims.std())}))
            type_cids[t].append(cid)

    if disjoint_types:
        # 只在不同类型、且**共享同一步骤**的概念之间声明互斥：这正是语义上最容易
        # 混淆的一对（如"冒充公检法·索要银行账户"与"冒充客服·索要银行账户"），
        # 全对偶声明会使 DISJOINT 占到公理总数的一半以上，失去信息量。
        by_step: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for t, cids in type_cids.items():
            for cid in cids:
                by_step[onto.get(cid).meta.get("step")].append((t, cid))
        for step, lst in by_step.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    t1, cid1 = lst[i]
                    t2, cid2 = lst[j]
                    if t1 == t2:
                        continue
                    for cid, ocid in ((cid1, cid2), (cid2, cid1)):
                        c = onto.get(cid)
                        if c is None:
                            continue
                        c.axioms.append(
                            Axiom(kind="DISJOINT", subj=cid, obj=ocid,
                                  confidence=0.9, evidence_count=1,
                                  source_doc_count=1, batch_cover_count=1))

    # 实例到最近概念的相似度分位（链接阈值标定依据，与 build_from_occurrences 同口径）
    sims = []
    for o in occ:
        best = -2.0
        for c in onto.concepts.values():
            best = max(best, float(np.dot(o.vec, np.asarray(c.centroid))))
        if best > -1:
            sims.append(best)
    info = {"build": f"skeleton:{granularity}", "n_d0": len(occ),
            "n_concepts": len(onto.concepts),
            "n_types": len(type_cids),
            "type_cids": {t: v for t, v in type_cids.items()},
            "link_sim_p05": float(np.quantile(sims, 0.05)) if sims else 0.5,
            "link_sim_p50": float(np.quantile(sims, 0.50)) if sims else 0.6}
    return onto, info


def skeleton_table(onto: Ontology) -> List[Dict]:
    """骨架本体清单（附录表）：概念名、类型、成员数、公理数。"""
    return [{"cid": cid, "name": c.name,
             "type": c.meta.get("type"), "step": c.meta.get("step"),
             "members": c.meta.get("n_members"),
             "n_axioms": len(c.axioms)}
            for cid, c in sorted(onto.concepts.items())]
