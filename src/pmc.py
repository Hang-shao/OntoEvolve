"""ontoevolve.src.pmc - PMC 开放获取全文语料（OAI-PMH 导出）的解析与时间切片。

数据格式（PMC OAI 导出）：
    每个 .xml 是一个 OAI-PMH ``ListRecords`` 响应，含 10 条 ``<record>``；
    每条 record 的 metadata 下是一篇 JATS ``<article>``：
        ``<article-meta>`` : ``<pub-date><year>``、``<article-title>``、
                             ``<kwd>``、``<journal-title>``
        ``<body>``         : ``<sec>`` / ``<p>`` 正文段落

与轨道 2（ICEWS18）的对照：
    ICEWS：实例 = 一条事件三元组，时间轴 = 事件时间戳，槽位 = 关系 + 参与者角色
    PMC  ：实例 = 一个正文句子，时间轴 = 发表年份，槽位 = 领域术语 + 段落语义类型

目标是给漂移检测提供一个**与 ICEWS 完全不同领域**的真实数据流，用于验证
方法不是只在时序知识图谱上成立。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from . import config as C

PMC_DIRNAME = "03_领域语料/PubMed_Central_PMC_300"

# 段落语义类型：正文不同部分承载的陈述类型不同，类比 ICEWS 的“参与者角色”
SEC_TYPES = ("background", "method", "result", "conclusion", "other")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WS = re.compile(r"\s+")


def _local(tag: str) -> str:
    """去掉 XML 命名空间，返回本地标签名。"""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(el: ET.Element) -> str:
    """取元素及其子元素的全部文本（JATS 的行内标签如 <italic> 需保留）。"""
    return _WS.sub(" ", "".join(el.itertext())).strip()


@dataclass
class PMCArticle:
    pmid: str
    year: int
    title: str
    journal: str
    paragraphs: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    # 完整发表日期 yyyymmdd（缺失的月/日记 0）；用于细粒度时间轴
    date_ymd: int = 0

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)


def _date_of(art: ET.Element) -> Tuple[int, int, int, int]:
    """取发表日期：优先 ppub（正式出版），否则 epub，再退化为年份。

    返回 (y, m, d, pubtype_rank)；缺失的月/日记 0。
    """
    best = (0, 0, 0, 9)
    for pd in art.iter():
        if _local(pd.tag) != "pub-date":
            continue
        y = m = d = 0
        for ch in pd.iter():
            ln = _local(ch.tag)
            if ln == "year" and (ch.text or "").strip().isdigit():
                y = int(ch.text.strip())
            elif ln == "month" and (ch.text or "").strip().isdigit():
                m = int(ch.text.strip())
            elif ln == "day" and (ch.text or "").strip().isdigit():
                d = int(ch.text.strip())
        if not y:
            continue
        rank = 0 if pd.get("pub-type") == "ppub" else 1
        cand = (y, m, d, rank)
        if rank < best[3] or (rank == best[3] and (y, m, d) > best[:3]):
            best = cand
    return best


def _pmid_of(art: ET.Element, header: Optional[ET.Element]) -> str:
    for aid in art.iter():
        if _local(aid.tag) == "article-id":
            if aid.get("pub-id-type") in ("pmid", "pmc", "doi"):
                if (aid.text or "").strip():
                    return aid.text.strip()
    if header is not None:
        for ch in header.iter():
            if _local(ch.tag) == "identifier" and (ch.text or "").strip():
                return ch.text.strip()
    return ""


def _sec_hint(sec: ET.Element) -> str:
    """根据 <sec> 的 <title> 猜测段落语义类型。"""
    for ch in sec.iter():
        if _local(ch.tag) == "title" and (ch.text or "").strip():
            t = ch.text.strip().lower()
            for kw, lab in (("method", "method"), ("result", "result"),
                            ("discussion", "conclusion"), ("conclusion", "conclusion"),
                            ("intro", "background"), ("background", "background")):
                if kw in t:
                    return lab
            break
    return "other"


def parse_file(path: Path) -> List[PMCArticle]:
    """解析一个 OAI-PMH 响应文件，返回其中的全部文章。"""
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return _parse_lenient(path)
    out: List[PMCArticle] = []
    # record -> header / metadata -> pmc-articleset -> article
    records = [e for e in root.iter() if _local(e.tag) == "record"]
    if not records:
        records = [root]
    for rec in records:
        header = None
        for ch in rec.iter():
            if _local(ch.tag) == "header":
                header = ch
                break
        for art in rec.iter():
            if _local(art.tag) != "article":
                continue
            paras: List[str] = []
            for sec in art.iter():
                if _local(sec.tag) not in ("sec", "body"):
                    continue
                hint = _sec_hint(sec) if _local(sec.tag) == "sec" else "other"
                for p in sec.iter():
                    if _local(p.tag) != "p":
                        continue
                    txt = _text(p)
                    if len(txt) >= 60:          # 过短的段落多为图注/声明
                        paras.append(f"[{hint}] {txt}")
            title = ""
            for ch in art.iter():
                if _local(ch.tag) == "article-title":
                    title = _text(ch)
                    break
            journal = ""
            for ch in art.iter():
                if _local(ch.tag) == "journal-title":
                    journal = _text(ch)
                    break
            kws: List[str] = []
            for ch in art.iter():
                if _local(ch.tag) == "kwd" and (ch.text or "").strip():
                    kws.append(ch.text.strip().lower())
            y, mo, dd, _ = _date_of(art)
            out.append(PMCArticle(pmid=_pmid_of(art, header), year=y,
                                  title=title, journal=journal,
                                  paragraphs=paras, keywords=kws,
                                  date_ymd=y * 10000 + mo * 100 + dd))
    return out


def _parse_lenient(path: Path) -> List[PMCArticle]:
    """逐块解析：个别畸形 XML 时尽量抢救出完整 <article> 子树。"""
    out: List[PMCArticle] = []
    try:
        for _, el in ET.iterparse(str(path), events=("end",)):
            if _local(el.tag) == "article":
                try:
                    out.extend(_article_from(el))
                except Exception:
                    pass
                el.clear()
    except ET.ParseError:
        pass
    return out


def _article_from(art: ET.Element) -> List[PMCArticle]:
    paras = []
    for p in art.iter():
        if _local(p.tag) == "p":
            txt = _text(p)
            if len(txt) >= 60:
                paras.append(f"[other] {txt}")
    title = ""
    for ch in art.iter():
        if _local(ch.tag) == "article-title":
            title = _text(ch)
            break
    y, mo, dd, _ = _date_of(art)
    return [PMCArticle(pmid=_pmid_of(art, None), year=y, title=title,
                       journal="", paragraphs=paras,
                       date_ymd=y * 10000 + mo * 100 + dd)]


def load_pmc(root: Optional[Path] = None, *,
             max_files: Optional[int] = None) -> List[PMCArticle]:
    """加载全部 PMC 文章（默认目录下所有 .xml）。"""
    d = Path(root or (C.DATA_ROOT / PMC_DIRNAME))
    if not d.exists():
        raise FileNotFoundError(f"PMC corpus not found: {d}")
    files = sorted(d.glob("*.xml"))
    if max_files:
        files = files[:max_files]
    arts: List[PMCArticle] = []
    for f in files:
        arts.extend(parse_file(f))
    return arts


def year_hist(arts: Sequence[PMCArticle]) -> Dict[int, int]:
    h: Dict[int, int] = {}
    for a in arts:
        h[a.year] = h.get(a.year, 0) + 1
    return dict(sorted(h.items()))


def sentences_of(paras: Sequence[str], *, min_len: int = 60,
                 max_len: int = 400) -> List[str]:
    """把段落切成句子；段落前缀 ``[type]`` 作为该句的语义类型。"""
    out: List[str] = []
    for p in paras:
        m = re.match(r"\[([a-z]+)\]\s*", p)
        hint = m.group(1) if m else "other"
        body = p[m.end():] if m else p
        for s in _SENT_SPLIT.split(body):
            s = _WS.sub(" ", s).strip()
            if min_len <= len(s) <= max_len:
                out.append(f"[{hint}] {s}")
    return out


def slice_by_year(arts: Sequence[PMCArticle], n_slices: int = 12,
                  *, min_year: int = 1990) -> List[List[PMCArticle]]:
    """按发表日期等分成 n_slices 个时间片（保持时序顺序）。

    ⚠ 本语料是 PMC 早期开放获取子集，年份跨度仅 1997–2001，且 2001 年占
    绝大多数。若按年份等宽切片，会有一半时间片同属 2001 年、时间轴退化。
    因此这里按**文章数等分**、排序键用完整发表日期（yyyymmdd），让时间片
    内部保持年月有序；代价是各片的“时间宽度”不等，这一点需在论文中说明。
    """
    use = sorted([a for a in arts if a.year >= min_year],
                 key=lambda a: (a.date_ymd, a.pmid))
    if not use or n_slices <= 0:
        return []
    return [list(c) for c in _chunk(use, n_slices)]


def _chunk(seq: Sequence[PMCArticle], n: int) -> List[Sequence[PMCArticle]]:
    size = len(seq) / n
    out: List[Sequence[PMCArticle]] = []
    for i in range(n):
        lo = int(round(i * size))
        hi = int(round((i + 1) * size))
        out.append(seq[lo:hi])
    return out
