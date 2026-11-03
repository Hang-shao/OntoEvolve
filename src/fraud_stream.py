"""ontoevolve.src.fraud_stream - 把反诈公开材料组织成"准时序"数据流（轨道 4）。

定位：**半合成应用轨道**。批次顺序由作者依公开材料记载的类型演化次序人工编排，
不是数据自带时间戳；真值事件 4—5 个，仅用于评测，不参与阈值标定。

数据来源（均为公开数据集，见 data/track4/audit）：
    Telescam（ScamGen.zip，句子级 5 分类，模板扩充生成）
    ZhCCVi（KoCCVi2 及平台收集的中文译本，仅作噪声子集）

批次规格由 BATCH_SPEC 给出（类型 → 条数），第四条起引入注入事件：
    R1 重定义（b04，利诱代办：步骤签名偏移）
    E1 涌现（b05，冒充熟人：留出类型首次出现，真实）
    X1 淘汰（b06 起，冒充客服：停止出现）
    E2 涌现（b07，复合话术：两类型句子拼接，半合成）
    R2 重定义（b08，冒充公检法：步骤签名偏移）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import fraud_vocab as FV
from .controlled_stream import Occurrence
from .embedding import l2_normalize

POOL_DIR = Path(__file__).resolve().parents[1] / "data" / "track4" / "pool"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "track4"

# 每批条数（各类型按比例拆分；最后按最小公倍数对齐到 per_batch）
PER_BATCH = 240
N_BATCHES = 10

# 类型 → 每批条数（None 表示该批不出现）
# 说明：重定义事件（R2）是"持续"的——b08 起冒充公检法一律改取受限子集，
# 否则 b09/b10 又回到原始构成，会使 O_T 的该概念内部出现自相矛盾的证据。
BATCH_SPEC: List[Dict[str, float]] = [
    {"冒充公检法": 0.50, "利诱代办": 0.30, "冒充客服": 0.20},            # b01  冷启动
    {"冒充公检法": 0.47, "利诱代办": 0.30, "冒充客服": 0.18, "_noise": 0.05},
    {"冒充公检法": 0.42, "利诱代办": 0.35, "冒充客服": 0.18, "_noise": 0.05},
    {"利诱代办": 0.60, "冒充公检法": 0.20, "冒充客服": 0.15, "_noise": 0.05},
    {"冒充熟人": 0.42, "冒充公检法": 0.30, "利诱代办": 0.18,
     "其他干扰": 0.05, "_noise": 0.05},                                  # E1 涌现
    {"冒充公检法": 0.40, "利诱代办": 0.30, "冒充熟人": 0.20, "其他干扰": 0.05,
     "_noise": 0.05},                                                    # X1 淘汰起
    {"利诱代办": 0.35, "冒充熟人": 0.40, "_composite": 0.10, "其他干扰": 0.10,
     "_noise": 0.05},                                                    # E2 涌现（半合成）
    {"冒充公检法": 0.25, "冒充熟人": 0.50, "_composite": 0.08, "其他干扰": 0.12,
     "_noise": 0.05},                                                    # R2 重定义起
    {"冒充公检法": 0.25, "冒充熟人": 0.55, "其他干扰": 0.15, "_noise": 0.05},
    {"冒充公检法": 0.24, "冒充熟人": 0.55, "利诱代办": 0.11, "其他干扰": 0.05,
     "_noise": 0.05},
]

# 重定义构造：类型 → 允许出现的步骤（keep_any）+ 必须消失的步骤（drop_any）
REDEF_SPEC = {
    "冒充公检法": {"keep_any": ("引导下载软件", "索要验证码", "要求转账汇款"),
                   "drop_any": ("虚构涉案",)},
}
# 重定义生效的批次（自该批起持续）
REDEF_FROM = {"冒充公检法": 8}


@dataclass
class FraudItem:
    iid: str
    text: str
    ftype: str                 # 真实类型（评测用，非本体标签）
    src: str                   # telescam | zhccvi | composite | noise
    steps: Tuple[str, ...] = ()
    synthetic: bool = False

    def as_dict(self) -> Dict:
        return {"iid": self.iid, "text": self.text, "type": self.ftype,
                "src": self.src, "steps": list(self.steps),
                "synthetic": self.synthetic}


def load_pools(pool_dir: Optional[Path] = None) -> Dict[str, List[Dict]]:
    pool_dir = Path(pool_dir or POOL_DIR)
    out = {}
    for name in ("telescam", "zhccvi"):
        p = pool_dir / f"{name}.jsonl"
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[name] = rows
    return out


class _Cursor:
    """按类型维护"取到第几条"，保证同一句子不在多个批次重复出现。"""

    def __init__(self, pools: Dict[str, List[Dict]], seed: int):
        rng = np.random.default_rng(seed)
        self.by_type: Dict[str, List[Dict]] = {}
        for r in pools["telescam"]:
            t = FV.type_of_label(str(r["label"]))
            if t == FV.TYPE_NOISE:
                continue                      # L4 单独按"其他干扰"取用
            self.by_type.setdefault(t, []).append(r)
        self.noise_pool = [dict(r, label="4") for r in pools["telescam"]
                           if FV.type_of_label(str(r["label"])) == FV.TYPE_NOISE]
        self.zh = pools["zhccvi"]
        for t in list(self.by_type):
            idx = rng.permutation(len(self.by_type[t]))
            self.by_type[t] = [self.by_type[t][int(i)] for i in idx]
        idx = rng.permutation(len(self.noise_pool))
        self.noise_pool = [self.noise_pool[int(i)] for i in idx]
        self.zh_nonfraud = [r for r in self.zh if str(r.get("label")) == "0"]
        idx = rng.permutation(len(self.zh_nonfraud))
        self.zh_nonfraud = [self.zh_nonfraud[int(i)] for i in idx]
        self.pos = {t: 0 for t in self.by_type}
        self.pos_noise = 0
        self.pos_zh = 0
        self.used: set = set()          # 已进入任何批次的 id，防止跨批重复

    def _take_from(self, pool: List[Dict], key: str, n: int, idpfx: str,
                   ftype: str, src: str = "telescam") -> List[FraudItem]:
        """从池中顺序取 n 条**未使用过**的记录（用尽则回绕并允许重复后告警）。"""
        out: List[FraudItem] = []
        p = self.pos.get(key, 0)
        scanned = 0
        while len(out) < n and scanned < len(pool) * 2:
            r = pool[p % len(pool)]
            p += 1
            scanned += 1
            iid = f"{idpfx}:{r['id']}"
            if iid in self.used:
                continue
            self.used.add(iid)
            out.append(FraudItem(iid=iid, text=r["text"], ftype=ftype, src=src,
                                 steps=tuple(FV.steps_of(r["text"]))))
        self.pos[key] = p
        return out

    def take(self, ftype: str, n: int) -> List[FraudItem]:
        if ftype == "_noise":
            out = self._take_from(self.zh_nonfraud, "noise_zh", n, "zh",
                                  "非诈骗转写", src="zhccvi")
            return out
        if ftype == "其他干扰":
            return self._take_from(self.noise_pool, "noise_ts", n, "ts", ftype)
        return self._take_from(self.by_type.get(ftype, []), ftype, n, "ts", ftype)

    def take_filtered(self, ftype: str, n: int) -> List[FraudItem]:
        """重定义构造：只取步骤集合受限的句子（确定性、不重复、同 seed 可复现）。"""
        spec = REDEF_SPEC[ftype]
        pool = self.by_type[ftype]
        picked: List[FraudItem] = []
        p = self.pos.get(ftype, 0)
        scanned = 0
        while len(picked) < n and scanned < len(pool) * 2:
            r = pool[p % len(pool)]
            p += 1
            scanned += 1
            steps = FV.steps_of(r["text"])
            if not (any(s in steps for s in spec["keep_any"]) and
                    not any(s in steps for s in spec["drop_any"])):
                continue
            iid = f"ts:{r['id']}"
            if iid in self.used:
                continue
            self.used.add(iid)
            picked.append(FraudItem(iid=iid, text=r["text"], ftype=ftype,
                                    src="telescam", steps=tuple(steps)))
        self.pos[ftype] = p
        return picked

    def composite(self, n: int) -> List[FraudItem]:
        """复合话术（半合成）：客服类句式 + 返利/垫付类句式拼接。"""
        a = self.by_type.get("冒充客服", [])
        b = self.by_type.get("利诱代办", [])
        if not a or not b:
            return []
        out: List[FraudItem] = []
        for k in range(n):
            ta = a[(self.pos_zh + k) % len(a)]["text"] if a else ""
            tb = b[(self.pos_zh * 3 + k) % len(b)]["text"] if b else ""
            text = f"{ta} 另外{tb}"
            out.append(FraudItem(iid=f"cmp:{k}", text=text, ftype="复合话术",
                                 src="composite", steps=tuple(FV.steps_of(text)),
                                 synthetic=True))
        return out


def build_batches(pools: Optional[Dict[str, List[Dict]]] = None, *,
                  seed: int = 0, n_batches: int = N_BATCHES,
                  per_batch: int = PER_BATCH
                  ) -> Tuple[List[List[FraudItem]], List[Dict], Dict]:
    """构造批次与真值记录。返回 (batches, truth, manifest)。"""
    pools = pools or load_pools()
    cur = _Cursor(pools, seed)
    rng = np.random.default_rng(seed + 17)
    batches: List[List[FraudItem]] = []
    type_trace: List[Dict] = []

    for bi, spec in enumerate(BATCH_SPEC[:n_batches], start=1):
        items: List[FraudItem] = []
        for key, ratio in spec.items():
            n = int(round(ratio * per_batch))
            if key in ("_noise", "_composite"):
                if key == "_noise":
                    items.extend(cur.take("_noise", n))
                else:
                    items.extend(cur.composite(n))
                continue
            if key == "其他干扰":
                items.extend(cur.take(key, n))
                continue
            # 重定义批次：该类型改取受限子集（自 REDEF_FROM 起持续）
            if key in REDEF_FROM and bi >= REDEF_FROM[key]:
                items.extend(cur.take_filtered(key, n))
            else:
                items.extend(cur.take(key, n))
        rng.shuffle(items)
        batches.append(items)
        cnt: Dict[str, int] = {}
        for it in items:
            cnt[it.ftype] = cnt.get(it.ftype, 0) + 1
        type_trace.append({"batch": bi, "n": len(items), "types": cnt})

    truth = [
        {"event_id": "E1", "dtype": "emergence", "batch_declare": 5,
         "target": FV.TYPE_HELD_OUT, "win_before": 1, "win_after": 1,
         "synthetic": False,
         "note": "留出类型'冒充熟人'首次成批出现（真实分类子集，O_0 未包含）"},
        {"event_id": "X1", "dtype": "extinction", "batch_declare": 5,
         "target": "冒充客服", "win_before": 1, "win_after": 5,
         "synthetic": False,
         "note": "冒充客服自 b05 起停止出现（b04 为最后一批）；淘汰判定需连续 "
                 "W=4 批低活跃，故允许 +5 批延迟并如实报告延迟"},
        {"event_id": "E2", "dtype": "emergence", "batch_declare": 7,
         "target": "复合话术", "win_before": 1, "win_after": 2,
         "synthetic": True,
         "note": "客服句式+返利/垫付句式拼接构造（半合成），b07—b08 连续出现；"
                 "涌现判定需连续 kappa=2 批，故最早在 b08 触发"},
        {"event_id": "R2", "dtype": "redefinition", "batch_declare": 8,
         "target": "冒充公检法", "win_before": 1, "win_after": 2,
         "synthetic": False,
         "note": "冒充公检法自 b08 起改取'下载/验证码/转账'子集，'虚构涉案'话术退出"},
    ]
    manifest = {"seed": seed, "n_batches": n_batches, "per_batch": per_batch,
                "spec": BATCH_SPEC[:n_batches], "trace": type_trace,
                "vocab": FV.vocab_report(), "truth": truth}
    return batches, truth, manifest


def to_occurrences(items: Sequence[FraudItem], embedder, *,
                   start_iid: int = 0) -> List[Occurrence]:
    """FraudItem → Occurrence（一次性批量嵌入）。"""
    texts = [it.text for it in items]
    vecs = embedder.embed(texts)
    out: List[Occurrence] = []
    for k, (it, v) in enumerate(zip(items, vecs)):
        ops, dps = FV.extract_slots(it.text)
        out.append(Occurrence(
            iid=start_iid + k, leaf=it.ftype, variant="base", text=it.text,
            vec=l2_normalize(np.asarray(v, dtype=np.float32)),
            op_slots=ops, dp_slots=dps, sup=FV.PARENT_EVENT,
            noise=(it.src in ("zhccvi",))))
    return out


def write_materialized(batches: Sequence[Sequence[FraudItem]],
                       truth: Sequence[Dict], manifest: Dict,
                       out_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_dir or OUT_DIR)
    bdir = out_dir / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for bi, items in enumerate(batches, start=1):
        with (bdir / f"b{bi:02d}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.as_dict(), ensure_ascii=False) + "\n")
    with (out_dir / "truth.jsonl").open("w", encoding="utf-8") as f:
        for r in truth:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "stream_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def read_materialized(out_dir: Optional[Path] = None
                      ) -> Tuple[List[List[FraudItem]], List[Dict], Dict]:
    out_dir = Path(out_dir or OUT_DIR)
    batches: List[List[FraudItem]] = []
    for bi in range(1, N_BATCHES + 1):
        p = out_dir / "batches" / f"b{bi:02d}.jsonl"
        if not p.exists():
            break
        items = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                items.append(FraudItem(iid=d["iid"], text=d["text"],
                                       ftype=d["type"], src=d["src"],
                                       steps=tuple(d.get("steps") or ()),
                                       synthetic=bool(d.get("synthetic"))))
        batches.append(items)
    truth = []
    tp = out_dir / "truth.jsonl"
    if tp.exists():
        with tp.open(encoding="utf-8") as f:
            truth = [json.loads(l) for l in f if l.strip()]
    manifest = json.loads((out_dir / "stream_manifest.json").read_text(encoding="utf-8"))
    return batches, truth, manifest
