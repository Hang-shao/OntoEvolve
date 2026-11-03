"""ontoevolve.src.owl_export - 本体 → OWL 2 DL 序列化（供 HermiT/Pellet 推理）。

映射约定（公理归属于概念，因此全部写成类上的受限公理，而不是全局的
rdfs:domain / rdfs:range —— 后者会把同一属性在不同概念上的用法互相污染，
导致大量假不一致）：

    OP(p, range)      → C ⊑ ∃p.RangeClass
    DP(p)             → p 声明为 DatatypeProperty（不带类限制）
    DOMAIN(obj)       → C ⊑ obj
    RANGE(obj)        → 值域类声明（与 OP 配合）
    CARD(min,max,p)   → C ⊑ (≥min p) ⊓ (≤max p)
    TRANSITIVE(p)     → p a owl:TransitiveProperty
    DISJOINT(obj)     → C ⊓ obj ⊑ ⊥
    parent            → C ⊑ parent

与 z3 替代方案的对齐：为每个**活跃**概念断言一个具名个体（i_C : C），
使“类不可满足”直接体现为“本体不一致”，从而与
consistency.satisfiable 的“每个活跃概念至少 1 个实例”语义一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from . import config as C
from .ontology_model import Ontology


def _base_ns(iri_base: Optional[str] = None) -> Namespace:
    base = iri_base or C.IRI_BASE
    return Namespace(base if base.endswith(("#", "/")) else base + "#")


def _cls(g: Graph, ns: Namespace, name: str) -> URIRef:
    """声明并返回一个类 IRI。"""
    uri = ns[_safe_local(name)]
    g.add((uri, RDF.type, OWL.Class))
    return uri


def _safe_local(name: str) -> str:
    """把概念 id / 名称转成合法的 IRI 本地名。"""
    out = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(name))
    return out or "X"


def ontology_to_graph(onto: Ontology, *, iri_base: Optional[str] = None,
                      with_individuals: bool = True,
                      notes: Optional[list] = None) -> Graph:
    """把 Ontology 转成 rdflib 图（OWL 2 DL）。

    OWL 2 DL 要求基数限制只作用于 **simple property**，而传递属性（及其子属性、
   逆）不是 simple。自动抽取的公理常在同一属性上同时给出 TRANSITIVE 与 CARD，
    直接导出会让推理机抛 IllegalArgumentException。冲突时保留基数限制
   （对一致性判定更敏感），不声明传递性，并在 notes 中记录。
    """
    ns = _base_ns(iri_base)
    g = Graph()
    g.bind("", ns)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    onto_iri = URIRef(str(ns).rstrip("#/"))
    g.add((onto_iri, RDF.type, OWL.Ontology))

    cids = {cid: _cls(g, ns, cid) for cid in onto.concepts}
    # parent 可能被引用但未在概念表中（构建阶段使用词表名）
    for cid, c in onto.concepts.items():
        if c.parent and c.parent not in cids:
            cids[c.parent] = _cls(g, ns, c.parent)

    card_used: Set[str] = set()
    for cid, c in onto.concepts.items():
        curi = cids[cid]
        if c.parent:
            g.add((curi, RDFS.subClassOf, cids[c.parent]))

        op_props: Dict[str, URIRef] = {}
        for ax in c.axioms:
            if ax.kind == "OP":
                p = ns[_safe_local(ax.pred or "rel")]
                op_props[ax.pred] = p
                g.add((p, RDF.type, OWL.ObjectProperty))
                rng = _cls(g, ns, ax.obj) if ax.obj else None
                restr = BNode()
                g.add((restr, RDF.type, OWL.Restriction))
                g.add((restr, OWL.onProperty, p))
                if rng is not None:
                    g.add((restr, OWL.someValuesFrom, rng))
                else:
                    g.add((restr, OWL.someValuesFrom, OWL.Thing))
                g.add((curi, RDFS.subClassOf, restr))
                if rng is not None:
                    ranges: Set[URIRef] = set()
                    ranges.add(rng)
            elif ax.kind == "RANGE" and ax.obj:
                _cls(g, ns, ax.obj)
            elif ax.kind == "DP":
                p = ns[_safe_local(ax.pred or "attr")]
                g.add((p, RDF.type, OWL.DatatypeProperty))
            elif ax.kind == "DOMAIN" and ax.obj:
                g.add((curi, RDFS.subClassOf, cids.get(ax.obj)
                       or _cls(g, ns, ax.obj)))
            elif ax.kind == "DISJOINT" and ax.obj:
                other = cids.get(ax.obj) or _cls(g, ns, ax.obj)
                g.add((curi, OWL.disjointWith, other))

        # 基数限制：作用于 CARD 指定的属性，缺省时取该概念首个对象属性
        for ax in c.axioms:
            if ax.kind != "CARD":
                continue
            # CARD 未指定属性且概念无对象属性时，落到默认属性上，
            # 否则“空基数区间”这类矛盾在导出的 OWL 里会消失
            pname = ax.pred or next(iter(op_props), None) or "rel"
            p = op_props.get(pname) or ns[_safe_local(pname)]
            card_used.add(pname)
            g.add((p, RDF.type, OWL.ObjectProperty))
            if ax.card_min > 0:
                mn = BNode()
                g.add((mn, RDF.type, OWL.Restriction))
                g.add((mn, OWL.onProperty, p))
                # OWL 2 的限定基数必须有 onClass，这里显式取 owl:Thing
                g.add((mn, OWL.onClass, OWL.Thing))
                g.add((mn, OWL.minQualifiedCardinality,
                       Literal(ax.card_min, datatype=XSD.nonNegativeInteger)))
                g.add((curi, RDFS.subClassOf, mn))
            if ax.card_max > 0:
                mx = BNode()
                g.add((mx, RDF.type, OWL.Restriction))
                g.add((mx, OWL.onProperty, p))
                g.add((mx, OWL.onClass, OWL.Thing))
                g.add((mx, OWL.maxQualifiedCardinality,
                       Literal(ax.card_max, datatype=XSD.nonNegativeInteger)))
                g.add((curi, RDFS.subClassOf, mx))

    # 第二遍：传递属性声明（避开已用于基数限制的属性）
    for cid, c in onto.concepts.items():
        for ax in c.axioms:
            if ax.kind != "TRANSITIVE":
                continue
            pname = ax.pred or "rel"
            if pname in card_used:
                if notes is not None:
                    notes.append(f"skip TransitiveProperty '{pname}': "
                                 "已在基数限制中使用（OWL 2 DL simple property）")
                continue
            p = ns[_safe_local(pname)]
            g.add((p, RDF.type, OWL.ObjectProperty))
            g.add((p, RDF.type, OWL.TransitiveProperty))

    if with_individuals:
        for cid, c in onto.concepts.items():
            if not c.active:
                continue
            ind = ns[f"i_{_safe_local(cid)}"]
            g.add((ind, RDF.type, OWL.NamedIndividual))
            g.add((ind, RDF.type, cids[cid]))

    return g


def owl_text(onto: Ontology, **kw) -> str:
    """序列化为 RDF/XML 文本（OWLAPI / HermiT 可直接读取）。"""
    return ontology_to_graph(onto, **kw).serialize(format="pretty-xml")


def export_owl(onto: Ontology, path, **kw) -> Path:
    """导出 OWL 文件，返回路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(owl_text(onto, **kw), encoding="utf-8")
    return p


def export_versions(versions: Iterable, out_dir, **kw) -> Dict[str, Path]:
    """批量导出一批 (name, ontology) 版本，返回 {name: 路径}。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for name, onto in versions:
        paths[str(name)] = export_owl(onto, out_dir / f"{_safe_local(str(name))}.owl",
                                      **kw)
    return paths
