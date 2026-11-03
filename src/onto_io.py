"""ontoevolve.src.onto_io - 本体的可复现序列化（JSON），供跨脚本复用同一 O_T。

只序列化评测与作答需要的字段（概念名、父类、质心、公理、活跃标志、计数），
不引入额外依赖，保证 data/track4 下的产物可被第三方脚本直接读取核对。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .ontology_model import Axiom, Concept, Ontology


def dump_onto(onto: Ontology, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": onto.version,
        "M": onto.M,
        "op_vocab": list(onto.op_vocab),
        "dp_vocab": list(onto.dp_vocab),
        "concepts": [
            {
                "cid": c.cid, "name": c.name, "parent": c.parent,
                "active": bool(c.active),
                "centroid": [float(x) for x in np.asarray(c.centroid).ravel().tolist()],
                "counts": list(c.counts or []),
                "weights": [float(w) for w in (c.weights or [])],
                "axioms": [
                    {"kind": a.kind, "pred": a.pred, "obj": a.obj,
                     "card_min": int(a.card_min), "card_max": int(a.card_max),
                     "confidence": float(a.confidence),
                     "evidence_count": int(a.evidence_count),
                     "source_doc_count": int(a.source_doc_count),
                     "batch_cover_count": int(a.batch_cover_count)}
                    for a in c.axioms],
            }
            for c in onto.concepts.values()
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def load_onto(path) -> Ontology:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    onto = Ontology(op_vocab=d["op_vocab"], dp_vocab=d["dp_vocab"],
                    version=int(d.get("version", 0)))
    for cd in d["concepts"]:
        ax = [Axiom(kind=a["kind"], subj=cd["cid"], pred=a.get("pred", ""),
                    obj=a.get("obj", ""), card_min=a.get("card_min", 0),
                    card_max=a.get("card_max", 0),
                    confidence=a.get("confidence", 1.0),
                    evidence_count=a.get("evidence_count", 0),
                    source_doc_count=a.get("source_doc_count", 0),
                    batch_cover_count=a.get("batch_cover_count", 0))
              for a in cd["axioms"]]
        onto.add(Concept(cid=cd["cid"], name=cd["name"], parent=cd.get("parent"),
                         centroid=np.asarray(cd["centroid"], dtype=np.float32),
                         axioms=ax, active=bool(cd["active"]),
                         counts=list(cd.get("counts") or []),
                         weights=[float(w) for w in (cd.get("weights") or [])]))
    return onto


def concepts_table(onto: Ontology) -> List[Dict]:
    """概念清单（附录用）：cid / 名称 / 父类 / 活跃 / 公理数 / 七类分布。"""
    out = []
    for cid, c in sorted(onto.concepts.items()):
        kinds: Dict[str, int] = {}
        for a in c.axioms:
            kinds[a.kind] = kinds.get(a.kind, 0) + 1
        out.append({"cid": cid, "name": c.name, "parent": c.parent,
                    "active": bool(c.active), "n_axioms": len(c.axioms),
                    "kinds": kinds})
    return out
