"""ontoevolve.src.fraud_vocab - 轨道 4（反诈公开材料）的中文词表与槽位抽取。

设计约束（论文必须能自证）：
  * 纯数据 + 纯函数，**不调用任何 LLM**；
  * 词表来源可追溯：①《面向智能应用的领域本体构建研究》等公开文献的话术步骤术语；
    ②Telescam 语料的字符 n-gram 高频片段（见 data/track4/audit/d1_report.txt）。
    二者均由人工筛选后写死在本文件，任何改动需在论文附录登记；
  * 槽位到七类公理模板的映射：步骤→OP、媒介/受众→DP、上位→DOMAIN、步骤数→CARD、
    步骤客体→RANGE、类型互斥→DISJOINT。

槽位约定：
    op_slots = [(步骤名, "话术步骤")]         → OP(概念, 步骤名, 话术步骤)
    dp_slots = [("media", 媒介), ("audience", 受众)]
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

# ---- 1. 诈骗类型（数据集标签 → 类型名；L3 为留出类型，不进入 O_0）------------
TYPE_BY_LABEL: Dict[str, str] = {
    "0": "冒充公检法",
    "1": "利诱代办",
    "2": "冒充客服",
    "3": "冒充熟人",
    "4": "其他干扰",
}

# O_0 覆盖的类型（b01 冷启动）与留出类型（后段涌现）
TYPES_IN_O0: Tuple[str, ...] = ("冒充公检法", "利诱代办", "冒充客服")
TYPE_HELD_OUT: str = "冒充熟人"
TYPE_NOISE: str = "其他干扰"

# 上位概念（DOMAIN 公理的取值范围，对应论文"事件—流程—场景"框架）
PARENT_EVENT = "诈骗事件"
PARENT_FLOW = "话术步骤"
PARENT_SCENE = "诈骗场景"

# ---- 2. 话术步骤词表（12 个，触发词为正则）----------------------------------
STEP_RULES: List[Tuple[str, str]] = [
    ("虚构涉案", r"破获|刑事案|窝点|涉嫌|通缉|洗钱|涉案|犯罪|违法|拘留|逮捕"),
    ("身份被冒用", r"身份信息|名下|名义|冒用|被盗用|异地|别人用"),
    ("要求报案自证", r"报案|自证|交代|说明.{0,4}情况|证明.{0,6}清白"),
    ("核验个人信息", r"核实|核验|验证|确认.{0,4}信息|个人信息|资料"),
    ("索要验证码", r"验证码|校验码|报.{0,2}码|短信码|动态码"),
    ("引导下载软件", r"下载|安装|APP|App|软件|会议|钉钉|腾讯会议|程序"),
    ("发送或点击链接", r"链接|网址|点击|扫码|二维码"),
    ("添加社交账号", r"微信|QQ|qq|加.{0,2}好友|添加我"),
    ("索要银行账户", r"银行卡|卡号|账号|存折|农信社|信用社|信用卡|银联|账户"),
    ("要求转账汇款", r"转账|汇款|转一笔|转给|打过来|打到|转出|转一点"),
    ("垫付或缴纳费用", r"垫付|缴纳|定金|手续费|保证金|缴税|充值|垫一下"),
    ("承诺退税返利", r"退税|退款|赔偿|返利|奖金|奖学金|中奖|理赔|返现|红包|补偿"),
    ("提出借款求助", r"借.{0,3}(点|一笔|些)?钱|借给我|借钱|急需|急用|周转|赔钱|"
                     r"保命|要你好看|摆平|私了|捞人|放.{0,4}出来"),
    ("制造紧迫情境", r"停机|欠费|关闭|注销|冻结|过期|失效|逾期|风险|损失|"
                     r"影响.{0,3}(征信|信誉)|黑名单|今天之内|尽快|马上"),
]

# ---- 3. 媒介与受众（DP 公理的取值）------------------------------------------
MEDIA_RULES: List[Tuple[str, str]] = [
    ("电话", r"打电话|来电|致电|通话|给你打"),
    ("短信", r"短信|发条信息|发信息"),
    ("社交软件", r"微信|QQ|qq"),
    ("App", r"下载|软件|APP|App|会议|钉钉"),
    ("链接", r"链接|网址|点击|扫码"),
]

AUDIENCE_RULES: List[Tuple[str, str]] = [
    ("学生", r"学生|校园|学长|同学|本科|毕业|奖学金|教务"),
    ("企业财务", r"财务|会计|出纳|总经理|王总|公司"),
    ("老年", r"老人|退休|爷爷|奶奶|大叔|大妈|阿姨"),
]

OP_VOCAB: List[str] = [s for s, _ in STEP_RULES]
DP_VOCAB: List[str] = ["media", "audience"]

_STEP_RE = [(name, re.compile(pat)) for name, pat in STEP_RULES]
_MEDIA_RE = [(name, re.compile(pat)) for name, pat in MEDIA_RULES]
_AUD_RE = [(name, re.compile(pat)) for name, pat in AUDIENCE_RULES]


def extract_slots(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """从一条中文文本抽取 op_slots / dp_slots（确定性，无 LLM）。"""
    ops = [(name, PARENT_FLOW) for name, rx in _STEP_RE if rx.search(text)]
    dps: List[Tuple[str, str]] = []
    for name, rx in _MEDIA_RE:
        if rx.search(text):
            dps.append(("media", name))
            break                      # 一条文本只记主媒介，避免多值虚增
    for name, rx in _AUD_RE:
        if rx.search(text):
            dps.append(("audience", name))
            break                      # 受众同理只记最匹配的一个
    return ops, dps


def steps_of(text: str) -> List[str]:
    return [name for name, rx in _STEP_RE if rx.search(text)]


def has_step(text: str, step: str) -> bool:
    for name, rx in _STEP_RE:
        if name == step:
            return bool(rx.search(text))
    return False


def type_of_label(label: str) -> str:
    return TYPE_BY_LABEL.get(str(label).strip(), TYPE_NOISE)


def vocab_report() -> Dict[str, object]:
    """给附录用的词表自述（词条数、正则数、来源）。"""
    return {
        "n_types": len(TYPE_BY_LABEL),
        "types": list(TYPE_BY_LABEL.values()),
        "types_in_o0": list(TYPES_IN_O0),
        "type_held_out": TYPE_HELD_OUT,
        "n_steps": len(STEP_RULES),
        "steps": OP_VOCAB,
        "dp_vocab": DP_VOCAB,
        "n_media": len(MEDIA_RULES),
        "n_audience": len(AUDIENCE_RULES),
        "source": ["公开反诈本体文献的话术步骤术语", "Telescam 语料字符 n-gram 高频片段（人工筛选）"],
    }


def step_signature(texts: Sequence[str]) -> Dict[str, int]:
    """一批文本的步骤分布（用于重定义构造与题集金标，不参与公理生成之外的用途）。"""
    out: Dict[str, int] = {}
    for t in texts:
        for s in steps_of(t):
            out[s] = out.get(s, 0) + 1
    return out
