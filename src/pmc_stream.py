"""ontoevolve.src.pmc_stream - 把 PMC 全文语料按发表日期切成检测器可消费的实例流。

映射方式（轨道 3，真实数据·生物医学领域，与轨道 2 的 ICEWS18 对照）：

    ==========  ==============================  ============================
    角色        轨道 2（ICEWS18，政治事件）      轨道 3（PMC，生物医学全文）
    ==========  ==============================  ============================
    实例        一条事件三元组                   一个正文句子
    时间轴      事件时间戳                       发表日期（yyyymmdd）
    对象属性槽  (关系名, 客体角色)                (主术语, 共现术语)
    数据属性槽  (参与者角色, 值)                  (术语, 段落语义类型)
    概念        事件类型                         研究主题
    ==========  ==============================  ============================

术语词表**从语料统计得到**（n-gram 文档频次 + 子串剪枝），不使用人工词表，
以免把外部知识注入到"跨领域迁移"的验证里。

漂移语义：涌现=新研究主题；淘汰=某主题连续多片不再出现；
          重定义=某主题的术语共现模式或语义类型分布发生变化。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .controlled_stream import Occurrence
from .embedding import Embedder
from .pmc import PMCArticle, sentences_of

# 精简英文停用词：术语抽取时过滤无信息量的组合
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "we", "our",
    "they", "their", "he", "she", "his", "her", "not", "no", "can", "may",
    "could", "should", "would", "will", "have", "has", "had", "do", "does",
    "did", "than", "then", "there", "here", "when", "which", "who", "whom",
    "also", "however", "thus", "therefore", "more", "most", "less", "least",
    "both", "each", "other", "such", "same", "only", "very", "all", "any",
    "some", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "using", "used", "use", "based", "due", "between", "among",
    "within", "without", "about", "into", "over", "under", "after", "before",
    "during", "while", "since", "because", "although", "whereas", "respectively",
}

_WORD = re.compile(r"[a-z][a-z\-]+")
_PREFIX = re.compile(r"^\[([a-z]+)\]\s*")


def _tokens(s: str) -> List[str]:
    return _WORD.findall(s.lower())


def strip_hint(sent: str) -> Tuple[str, str]:
    """拆开 ``[type] 正文``，返回 (语义类型, 正文)。"""
    m = _PREFIX.match(sent)
    if m:
        return m.group(1), sent[m.end():]
    return "other", sent


def build_term_vocab(arts: Sequence[PMCArticle], *, top_k: int = 300,
                     min_df: int = 6, max_art_ratio: float = 0.45,
                     ngram_max: int = 3, use_keywords: bool = True,
                     kw_min_df: int = 5, kw_top: int = 100,
                     max_sents_per_article: int = 300,
                     verbose: bool = False) -> List[str]:
    """从语料统计领域术语词表。

    判据说明（避免抽到学术通用词）：

    * **df**：出现于多少个句子。太低是拼写/罕见噪声。
    * **文章覆盖率上界 ``max_art_ratio``**：出现在超过 ``max_art_ratio``
      比例文章中的短语视为通用词（如 patients / analysis / results，
      几乎所有生物医学论文都会写），直接剔除——这是区分"领域术语"与
      "学术通用词"的主要手段，比手工枚举停用词更客观。
    * **长短语偏好**：打分 ``df × (1 + 0.35·(n-1))``，让 "breast cancer"
      排在 "cancer" 前面。
    * **子串剪枝**：短术语 t 若是已保留长术语 T 的子串且 df(t) 不超过
      df(T) 的 1.2 倍，说明 t 只是片段，去掉 t。
    * **作者关键词**：文章自带的 ``<kwd>`` 直接纳入（质量最高的领域术语）。
    """
    df: Counter = Counter()          # 句子频次（句内去重）
    df_art: Counter = Counter()      # 文章频次
    n_art = 0
    for a in arts:
        n_art += 1
        sents = sentences_of(a.paragraphs)
        if max_sents_per_article and len(sents) > max_sents_per_article:
            sents = sents[:max_sents_per_article]
        seen_art: set = set()
        for s in sents:
            _, body = strip_hint(s)
            toks = _tokens(body)
            seen: set = set()
            for n in range(1, ngram_max + 1):
                for i in range(len(toks) - n + 1):
                    if len(toks) - i < n:
                        break
                    g = toks[i: i + n]
                    if g[0] in STOPWORDS or g[-1] in STOPWORDS:
                        continue
                    if any(len(w) < 3 for w in g):
                        continue
                    phrase = " ".join(g)
                    if len(phrase) < 8:
                        continue
                    seen.add(phrase)
            df.update(seen)
            seen_art |= seen
        df_art.update(seen_art)
    if n_art == 0:
        return []

    cap = max_art_ratio * n_art
    cand = [t for t in cand_df(df) if df[t] >= min_df and df_art[t] <= cap]
    cand.sort(key=lambda t: (-_score(t, df), t))

    kept: List[str] = []
    for t in cand:
        if any(t in h and df[t] <= df[h] * 1.2 for h in kept):
            continue
        kept.append(t)
        if len(kept) >= top_k:
            break

    n_stat = len(kept)
    if use_keywords:
        # 作者关键词质量最高，但多数只出现在 1 篇文章里；这里只收能在句子中
        # 稳定观察到的（df≥2），否则会把签名维度撑大而只带来噪声
        kws = Counter()
        for a in arts:
            kws.update(a.keywords)
        n_kw = 0
        for k, c in kws.most_common():
            if n_kw >= kw_top:
                break
            if k in kept or df.get(k, 0) < kw_min_df:
                continue
            kept.append(k)
            n_kw += 1
    if verbose:
        print(f"[vocab] 候选 {len(df)} → 过覆盖率/频次 {len(cand)} → "
              f"剪枝后 {len(kept)}（统计 {n_stat} + 关键词 {len(kept) - n_stat}）")
        print(f"[vocab] top12: {kept[:12]}")
    return kept


def cand_df(df: Counter) -> List[str]:
    return list(df.keys())


def _score(t: str, df: Counter) -> float:
    n = t.count(" ") + 1
    return df[t] * (1.0 + 0.35 * (n - 1))


def _compile(vocab: Sequence[str]) -> re.Pattern:
    """把词表编译成一个正则，一次性找出句子中所有术语。"""
    parts = sorted((re.escape(t) for t in vocab), key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(parts) + r")(?![a-z])")


def sentence_slots(sent: str, matcher: re.Pattern, *, max_dp: int = 4
                   ) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """把一个句子映射成 (leaf, op_slots, dp_slots)。

    leaf     : 最长匹配术语（信息量最大者），类比 ICEWS 的关系名
    op_slots : [(leaf, 共现术语)]，类比 (关系, 客体角色)
    dp_slots : [(段落语义类型, 术语)]，类比 (参与者角色, 值)

    注意 dp 槽的**谓词是语义类型**而非术语：若谓词也用术语，公理签名维度
    M = |术语| + |术语| 会翻倍（实测 970），与 ICEWS 的 264、受控流的 18
    不可比，重定义阈值的跨轨道迁移就失去意义。
    """
    hint, body = strip_hint(sent)
    low = body.lower()
    hits = matcher.findall(low)
    # 去重保序，按长度降序（最具体的排前面）
    uniq: List[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    uniq.sort(key=lambda t: (-len(t), t))
    if not uniq:
        return "misc", [], [(hint, "none")]
    leaf = uniq[0]
    second = uniq[1] if len(uniq) > 1 else "none"
    op = [(leaf, second)]
    dp = [(hint, t) for t in uniq[:max_dp]]
    return leaf, op, dp


def slice_occurrences(arts: Sequence[PMCArticle], matcher: re.Pattern,
                      embedder: Embedder, *, t: int, per_slice: int,
                      seed: int = 0, start_iid: int = 0,
                      min_len: int = 60, max_len: int = 400
                      ) -> List[Occurrence]:
    """把一个时间片的文章转成 Occurrence 列表（一次性嵌入）。"""
    sents: List[str] = []
    for a in arts:
        sents.extend(sentences_of(a.paragraphs, min_len=min_len, max_len=max_len))
    if not sents:
        return []
    rng = np.random.default_rng(seed + t)
    if per_slice and len(sents) > per_slice:
        idx = rng.choice(len(sents), size=per_slice, replace=False)
        idx.sort()
        sents = [sents[i] for i in idx]
    texts = [strip_hint(s)[1] for s in sents]
    vecs = embedder.embed(texts)
    out: List[Occurrence] = []
    for i, (s, v) in enumerate(zip(sents, vecs)):
        leaf, op, dp = sentence_slots(s, matcher)
        out.append(Occurrence(
            iid=start_iid + i,
            leaf=leaf,
            variant="base",
            text=texts[i],
            vec=np.asarray(v, dtype=np.float32),
            op_slots=op,
            dp_slots=dp,
            sup="",
        ))
    return out


def build_pmc_stream(arts: Sequence[PMCArticle], n_slices: int,
                     embedder: Embedder, *, per_slice: int = 600,
                     seed: int = 0, top_k: int = 300, min_df: int = 6,
                     verbose: bool = True) -> Tuple[List[List[Occurrence]], List[str]]:
    """切片 → 术语词表 → 逐片嵌入，返回 (每片实例列表, 术语词表)。"""
    from .pmc import slice_by_year

    slices = slice_by_year(arts, n_slices)
    # 词表在**全部语料**上统计（不按片），否则各片词表不同、公理签名不可比
    vocab = build_term_vocab(arts, top_k=top_k, min_df=min_df,
                             verbose=verbose)
    matcher = _compile(vocab)

    stream: List[List[Occurrence]] = []
    iid = 0
    for si, sl in enumerate(slices, start=1):
        occ = slice_occurrences(sl, matcher, embedder, t=si,
                                per_slice=per_slice, seed=seed, start_iid=iid)
        iid += len(occ)
        stream.append(occ)
        if verbose:
            leaves = Counter(o.leaf for o in occ)
            print(f"  slice {si:>2}: 文章={len(sl):>3} 句子={len(occ):>4} "
                  f"不同术语={len(leaves):>3} top={leaves.most_common(3)}")
    return stream, vocab
