"""llm_annotate_alarms - 用大模型对真实轨道告警做首轮标注（人工全量盲复核）。

定位：模型只做"第一遍标注 + 证据摘录"，最终真值以人工复核结果为准。
因此输出文件与人工复核文件分开保存，最后由 analyze_alarm_annotation.py
合并计算一致性与精确率。

Run:
    python tools/llm_annotate_alarms.py --pack annotations/alarm_pack_shuffled.csv \
        --out annotations/alarm_pack_llm.csv [--limit 5] [--workers 4]

环境变量：
    ONTOEVOLVE_ANNOT_API_URL / ONTOEVOLVE_ANNOT_API_KEY / ONTOEVOLVE_ANNOT_MODEL
    缺省回退到 ONTOEVOLVE_CHAT_*（同一 OpenAI 兼容网关），模型缺省 deepseek-chat。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

from src import config as C

SYSTEM_PROMPT = """你是一名本体工程领域的标注员。给定一条"概念漂移告警"及其证据，\
判断它是否对应本体中真实发生的概念层变化。严格按判据判断，宁可判"无法判定"也不要臆测。\
只输出一个 JSON 对象，不要输出任何其他文字。

判据：
1) 涌现：出现了本体中没有对应概念的新概念（不是新实体、不是已有概念的别名或拼写变体），\
且有至少 2 条证据句描述了该概念的关系或属性。
2) 淘汰：该概念近期不再被提及，且没有上位/下位概念承接其语义。单批缺失不算淘汰；\
若标记"从未提及"，需判断该概念本身是否为有效概念（有效聚合→是；明显是构建噪声或无意义聚合→否）。
3) 重定义：同一概念的属性、关系、基数或不相交约束发生语义变化，而非仅名称或同义表达变化。\
若提供了公理前后对照，以其为依据。
4) 判"否"的典型情形：别名/拼写变体、纯数值分布漂移、仅新增实体实例、抽取模板错配产生的伪概念、\
证据句与该概念无关。
5) 判"无法判定"的情形：证据为空且无历史、证据句过短或无法体现概念层变化。

输出格式（严格 JSON，不得增删字段）：
{"is_true_drift": "是|否|无法判定", "dtype_true": "涌现|淘汰|重定义|不适用", \
"confidence": "高|中|低", "evidence_quote": "从证据句原文摘录不超过80字的片段", \
"rationale": "不超过60字的判定理由"}"""


def _domain_hint(track: str) -> str:
    if track.upper().startswith("ICEWS"):
        return ("领域：ICEWS18 政治事件编码数据。概念对应事件类型"
                "（如 Make statement、Accuse），证据句是事件记录文本。")
    return ("领域：PMC 生物医学文献。概念对应文献主题/术语聚类，"
            "证据句是论文原句。")


def _build_user_msg(row: Dict) -> str:
    lines = [
        f"轨道：{row['track']}",
        _domain_hint(row["track"]),
        f"时间批：第 {row['t']} 批",
        f"系统判定类型：{row['dtype']}（仅表示告警来源，须独立判断真实类型）",
        f"本批匹配实例数：{row['n_members']}",
        f"信号值：语义 {row['sem']}，统计 {row['stat']}，公理 {row['ax']}，"
        f"平均置信度 {row['mean_conf']}",
    ]
    name = row.get("concept", "")
    if name.startswith(("new_concept_", "topic_", "NEW")):
        lines.append("概念名：涌现候选（系统以占位符命名，名称不参与判断，"
                     "请依据证据句判断是否构成新概念）")
    else:
        lines.append(f"概念名：{name}")
    if row.get("nearest_concept"):
        lines.append(f"已有本体中最相近的概念：{row['nearest_concept']}"
                     f"（余弦距离 {row.get('score', '')} 不参与判断，仅供参照）")
    if row.get("parent_concept"):
        lines.append(f"该概念在当前本体中的父概念：{row['parent_concept']}")
    defs = [s for s in (row.get("definition") or "").split(" || ") if s.strip()]
    if defs:
        lines.append("该概念在当前本体中的定义：")
        for s in defs:
            lines.append(f"  - {s}")
    if row.get("never_mentioned"):
        lines.append("该概念自建立后从未在任何批次被匹配到实例（历史证据为空属事实）。")
    elif row.get("last_seen_t"):
        lines.append(f"该概念最近一次出现于第 {row['last_seen_t']} 批。")
    ev = [s for s in (row.get("evidence") or "").split(" || ") if s.strip()]
    if ev:
        lines.append("证据句：")
        for i, s in enumerate(ev, 1):
            lines.append(f"  [{i}] {s}")
    else:
        lines.append("证据句：无")
    for key, label in (("axioms_before", "重定义前公理"),
                       ("axioms_after", "重定义后公理")):
        vals = [s for s in (row.get(key) or "").split(" || ") if s.strip()]
        if vals:
            lines.append(f"{label}：")
            for s in vals[:8]:
                lines.append(f"  - {s}")
    return "\n".join(lines)


def _parse_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    if not text:
        return None
    # 去掉 ```json 围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 末路：逐字段正则抽取（应对输出被截断导致 JSON 不完整）
    g = lambda k: (re.search(rf'"{k}"\s*:\s*"([^"]*)"', text) or [None, None])[1]
    if g("is_true_drift"):
        return {"is_true_drift": g("is_true_drift"),
                "dtype_true": g("dtype_true") or "",
                "confidence": g("confidence") or "",
                "evidence_quote": g("evidence_quote") or "",
                "rationale": g("rationale") or ""}
    return None


def _one(session: requests.Session, url: str, key: str, model: str,
         row: Dict, retries: int = 3) -> Dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_msg(row)},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    last = ""
    for k in range(retries):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                content = (ch.get("message") or {}).get("content") or ""
                if not content.strip():
                    last = f"empty_content(finish={ch.get('finish_reason')})"
                    time.sleep(1.5 * (k + 1))
                    continue
                obj = _parse_json(content) or {}
                return {
                    "alarm_id": row["alarm_id"],
                    "llm_is_true_drift": obj.get("is_true_drift", ""),
                    "llm_dtype": obj.get("dtype_true", ""),
                    "llm_confidence": obj.get("confidence", ""),
                    "llm_evidence_quote": obj.get("evidence_quote", ""),
                    "llm_rationale": obj.get("rationale", ""),
                    "error": "" if obj else "unparsable_json",
                    "raw": content[:500],
                }
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (k + 1))
    return {"alarm_id": row["alarm_id"], "llm_is_true_drift": "",
            "llm_dtype": "", "llm_confidence": "", "llm_evidence_quote": "",
            "llm_rationale": "", "error": last, "raw": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / "annotations" / "alarm_pack_shuffled.csv"))
    ap.add_argument("--out", default=str(ROOT / "annotations" / "alarm_pack_llm.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="只标注指定 alarm_id，逗号分隔")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=os.getenv("ONTOEVOLVE_ANNOT_MODEL", "deepseek-chat"))
    ap.add_argument("--base-url", default=os.getenv(
        "ONTOEVOLVE_ANNOT_API_URL", C.CHAT_API_URL))
    ap.add_argument("--api-key", default=os.getenv(
        "ONTOEVOLVE_ANNOT_API_KEY", C.CHAT_API_KEY))
    args = ap.parse_args()

    if not args.base_url:
        sys.exit("[error] 未配置标注模型接口（ONTOEVOLVE_ANNOT_API_URL）")
    url = args.base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    with open(args.pack, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        rows = [r for r in rows if r["alarm_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[annot] {len(rows)} 条，model={args.model}  url={url}")

    done = set()
    outp = Path(args.out)
    prev = []
    if outp.exists():
        with outp.open(encoding="utf-8-sig") as f:
            prev = [r for r in csv.DictReader(f)
                    if r.get("llm_is_true_drift") and not r.get("error")]
        done = {r["alarm_id"] for r in prev}
        print(f"[annot] 断点续跑：已完成 {len(done)} 条")
    rows = [r for r in rows if r["alarm_id"] not in done]

    out_fields = ["alarm_id", "llm_is_true_drift", "llm_dtype", "llm_confidence",
                  "llm_evidence_quote", "llm_rationale", "error", "raw"]
    lock = threading.Lock()
    results = prev[:]
    session = requests.Session()
    if rows:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(_one, session, url, args.api_key, args.model, r): r
                    for r in rows}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                with lock:
                    results.append(res)
                flag = res["error"][:60] if res["error"] else res["llm_is_true_drift"]
                print(f"  [{i}/{len(rows)}] {res['alarm_id']}: {flag}")

    with outp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    err = sum(1 for r in results if r.get("error"))
    print(f"[annot] 完成 {len(results)} 条，失败 {err} -> {outp}")
    if err:
        print("[annot] 存在失败条目，可用同一命令重跑（自动续跑未完成项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
