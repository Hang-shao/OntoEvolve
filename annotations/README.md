# annotations — 告警人工复核包

对应论文正文 Section 4.6.2 的告警精度估计（0.1585，13/82）和 Section 5.3 的"单标注者"披露。

## 一、去原文处理（重要）

论文的 Data availability 声明：Track 2 (ICEWS18) 与 Track 3 (PMC) 依各自许可**不再分发**。
为与此一致，本目录在发布前清空了以下列（原始文件备份在 `dist/annotations_backup-<时间戳>/`，
不入库）：

| 列 | 内容 | 处理 |
|---|---|---|
| `evidence` | PMC 原文句子 / ICEWS18 事件文本 | 清空 |
| `definition` | ICEWS18 的实体-角色链 | 清空 |
| `axioms_before` / `axioms_after` | 候选公理内容 | 清空 |
| `llm_evidence_quote` / `raw` | LLM 复核时的逐字引文与原始响应 | 清空 |

处理脚本：`scripts/redact_annotations.py`。复核统计所需的字段全部保留：
`alarm_id`、`track`、`t`、`dtype`/`system_dtype`、`concept`、`score`、`n_members`、
`last_seen_t`、`never_mentioned`、`sem`/`stat`/`ax`、`mean_conf`、`nearest_concept`、
`parent_concept`，以及判定列 `is_true_drift`、`dtype_true`、`confidence`、
`llm_*`（判定与置信度）与 `note`/`llm_rationale`（标注者的转述判断，不是原文摘录）。

因此本目录可以复现"85 条告警中 ICEWS18 有 1 真 / 15 假 / 3 未定、PMC 有 12 真 / 54 假"
这一计数，但**不能**直接复核每条告警的证据是否成立——那需要回到原始语料，
按 ICEWS18 与 PMC 各自的许可获取。论文正文已把这一估计写成
"exploratory, non-blind single-rater review"，而不是正式的双人标注精度。

## 二、文件

| 文件 | 行 | 用途 |
|---|---|---|
| `alarm_pack.csv` | 85 | 告警复核表（seed 0，两轨道），含判定列 |
| `alarm_pack_review.csv` | 85 | 同上，附单标注者的最终判定与理由 |
| `alarm_pack_wide.csv` | 85 | 同上，按轨道拆列的宽表，含 `agree`（人工与 LLM 是否一致） |
| `alarm_pack_llm.csv` | 85 | LLM 复核结论（对应论文中"LLM 作为旁证"的用法） |
| `alarm_pack_shuffled.csv` | 85 | 打乱顺序的版本，用于盲评 |
| `human_eval_for_scoring.csv` | 125 | 下游问答的人工评分样本；`human_score` 列**为空**，双人盲评未进行 |

## 三、已知缺口

`human_eval_for_scoring.csv` 的 `human_score` 全部为空，说明"人工评分 + κ"这条线尚未做。
`compute_human_kappa.py` 已就绪，补齐两列后即可给出 Cohen's κ 与混淆矩阵。
论文目前不依赖这条线：Section 4.5.1 用的是两个 LLM judge 与确定性判据的交叉验证。
