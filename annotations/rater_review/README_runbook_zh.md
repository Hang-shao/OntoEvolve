# 双人盲评执行手册（作者用）

对应审稿意见 **Major 5**：把 §4.6.2 的告警精度（0.1585 = 13/82）从“单人、非盲、单次”升级为
“两人独立盲评 + 一致性统计”。**它不会改写 0.1585**，只新增一份可复核的一致性证据。

打分表本身在 `dist/rater_pack/`（`dist/` 已被 `.gitignore` 排除：表里含 PMC 摘要原句与 ICEWS 事件记录，不能随仓库分发）；判定规则、邮件模板与本手册放在 `annotations/rater_review/`，可入库、可分发给审稿人。**剩下的只有两个人**。

---

## 0. 现在已经就绪的东西

| 文件 | 用途 | 给谁 |
|---|---|---|
| `dist/rater_pack/rater_R2.csv` | 打分表 A（85 行，行序 A） | 标注者 1 |
| `dist/rater_pack/rater_R3.csv` | 打分表 B（同一批 85 条，行序 B） | 标注者 2 |
| `dist/rater_pack/rater_sheet_manifest.json` | 两份表的 sha256、行数、生成时间、源文件哈希 | 你自己留档 |
| `rater_instructions_EN.md` / `_ZH.md` | 判定规则、决策树、5 个虚构示例、纪律 | 随表发出 |
| `email_request_template.md` | 中英双语邀请邮件 | 你自己改 |

两份表的差异只有行序（种子 20261106 / 20261107），条目内容与列完全相同；判定在三列上留空：
`label`、`confidence`、`note`。

**可见列**：`seq, alarm_id, track, t, dtype, concept, n_members, last_seen_t, evidence`。
**不可见列**（脚本会断言它们没进表）：`score, sem, stat, ax, mean_conf, definition,
axioms_before/after, never_mentioned, nearest_concept, parent_concept`，以及所有判定列
（`is_true_drift, dtype_true, confidence, note`）与 `llm_*`。

重跑生成（会覆盖，先备份）：
```
cd ontoevolve
python scripts/build_rater_sheets.py --names R2 R3      # 默认输出 dist/rater_pack/
python scripts/build_rater_sheets.py --check            # 只看统计，不写文件
```

## 1. 你需要现在就决定的五件事

| 决定 | 建议 | 理由 |
|---|---|---|
| 找谁 | 两位**不是作者**的人，能读英文科研文本 | 作者自评不构成独立证据 |
| 领域匹配 | 一位偏生物医学（PMC 66 条）、一位偏国际关系/事件数据（ICEWS 18 条），**但两人都评全部 85 条** | 只有都评全部才有一致性可算；脚本另给分轨道精度 |
| 是否付费 | 按小时付酬或以劳务致谢，二者择一，邮件里写明 | 2–3 小时的工作，纯人情难催第二次 |
| 时间窗 | 两周，允许分两次做 | 一次做 85 条，后半段准确率会掉 |
| 是否练手 | **不设练习集** | 85 条里挑“示范条目”会把答案漏给标注者；说明里已给 5 个虚构示例 |

找不到同领域的人时：仍按“两人都评 85 条”执行，只在文中说明两位标注者均为通用科学读者，
并把**分轨道一致性**作为主结果（PMC 66 条的一致性比合并 85 条更可信）。

## 2. 发送什么、绝不发送什么

**发**：`rater_R2.csv`（或 `rater_R3.csv`）+ `rater_instructions_EN.md`（中文标注者则用 `_ZH`）。

**绝不发**：
- `annotations/alarm_pack_review.csv`（首位标注者的判定与理由）
- `annotations/alarm_pack_llm.csv`（LLM 结论）
- `annotations/alarm_pack_wide.csv`（含 `agree` 列）
- 论文草稿、代码仓库地址

说明文件第 5 节已写明“若手上已有文件名带 `llm` 或 `review` 的文件，请告知并忽略”。发出前
请确认你只附了那两份。

## 3. 收表后：检查 → 评分 → 仲裁

### 3.1 完整性检查

评分脚本会报告未填条目：
```
cd ontoevolve
python scripts/score_rater_agreement.py --a dist/rater_pack/rater_R2.csv \
        --b dist/rater_pack/rater_R3.csv
```
若输出里 `未填：A N 条 [...]`，把行号发回给对应标注者补填；**不要替他们填**。

### 3.2 评分

```
python scripts/score_rater_agreement.py \
    --a dist/rater_pack/rater_R2.csv --b dist/rater_pack/rater_R3.csv \
    --a-name R2 --b-name R3 \
    --r1 dist/annotations_backup-20261102-235759/alarm_pack_review.csv \
    --boot 2000 --out dist/rater_pack/agreement_report.md
```
报告含：判定分布、原始一致率、Cohen's κ、Krippendorff's α、PABAK（三者都带按条目重抽的
95% bootstrap 区间）、混淆矩阵、阳性/阴性一致率、各自与共同可判定子集的精度（Wilson 区间）、
分轨道精度、与首位标注者 R1 的对照、置信等级分布、待仲裁清单（只给 ID 与两方标签）。

`--r1` 那一段会把 R1 的历史判定重算一遍并核对 13/82 = 0.1585，用来证明分母没错。R1 非盲，
它的 κ 只作参考，**不进结论**。

### 3.3 仲裁

对报告第七节列出的分歧条目做独立裁定（第三人或你自己都可以，但要在文中说明是谁、是否看过两方
理由）。`rater_instructions` 已声明 **`note` 不交给仲裁人**，所以仲裁人只看到条目与两个标签。

仲裁表两列即可，然后：
```
# dist/rater_pack/arbitration.csv
# alarm_id,ruling
# PMC-07,true
python scripts/score_rater_agreement.py --a ... --b ... --arbitration dist/rater_pack/arbitration.csv
```
`ruling` 接受 `true/false/undecidable` 或 `真漂移/误报/无法判断`。

## 4. 论文里怎么写（模板，等结果出来再填空）

正文 §4.6.2 末句之后加两句：

> Two raters, neither of them an author, independently re-labelled all 85 alarms from the
> packet of claim plus quoted evidence, without the model's scores, without the first
> rater's labels and without access to the corpora. Agreement was
> Krippendorff's α = __ (95% CI __–__) and Cohen's κ = __ (__–__) on the three-way
> task, with positive agreement __ (__/__) and negative agreement __ (__/__); the __
> disagreements were adjudicated by a third party and are reported separately. The
> resulting precision on the same denominator is __ (__/82, 95% CI __–__).

ESM 加一张表（建议 Table S17）：条目数、未填、α/κ/PABAK 及其区间、正/负一致率、混淆矩阵、
分轨道精度、仲裁后的精度、以及“证据为空 4 条”与“10 条淘汰告警无批次上下文”的处理说明。

**必须写进正文或 ESM 的三条披露**：

1. R1 见过模型分数且非盲，R2/R3 是盲评，三者信息量不同，因此这是一次**盲评复核**，
   不能写成 0.1585 的“复现”。
2. 盲评包只含“主张 + 引句”，不含原文全文（语料许可不允许再分发），所以 10 条淘汰告警中
   没有证据的那些只能判 `无法判断`；这条限制同时解释了为什么分轨道一致性比合并值更可信。
3. 发布仓库里的 `annotations/` 已脱敏，第三方要复现这次盲评，须自行按 PMC/ICEWS18 的条款
   取得语料；可复现的是**流程与脚本**，不是判定本身。

措辞边界（与既有 V24 草案一致）：只可写 `a limited human review of detected alarms`，
不可写 `comprehensive validation` / `human-validated precision`。

## 5. 六个坑

1. **不要把 LLM 结论当“第二标注者”**。同一模型家族出两份“判断”不是独立证据，审稿人会看穿，
   而且与论文自己“LLM 作旁证”的定位冲突。
2. **不要自己再评一遍当第二人**。作者 = 非独立。
3. **不要改 0.1585**。新的盲评结果另列；如果两者差别大，那是发现，不是错误。
4. **阳性只有 13 条，κ 不稳**。13/85 的阳性率下 κ 对边际分布极敏感（干跑中 κ = 0.54 而
   PABAK = 0.65，阳性一致率只有 0.40）。必须同时给 α、PABAK、阳性/阴性一致率与计数，
   只报 κ 会被认为选了个好看的数。
5. **Excel 会改行序或把 CSV 存成 GBK**。表头带 BOM，用“CSV UTF-8”另存；收表后先比对
   `alarm_id` 集合是否与 `rater_sheet_manifest.json` 的行数一致。
6. **不要公开分发盲评包**。里面有 PMC 摘要原句与 ICEWS 事件记录，与 Data availability 的
   “不再分发”声明冲突。

## 6. 干跑（仅验证脚本可跑，不是结果）

`scratchpad/…/dryrun/` 下有两份用**历史判定**填好的假表（R2 = R1 原样，R3 扰动 12%），
用来确认评分脚本的数学与报告格式：R1 复算得 13/82 = 0.159 [0.095, 0.253]，与论文一致；
假 R3 给出 α = 0.54、阳性一致率 0.40。**这批数字不得进入论文**，它们只证明管线通。

## 7. 成本与工期

| 项 | 量 |
|---|---|
| 每人工作量 | 85 条 × 1–2 分钟 ≈ 2–3 小时（含阅读说明） |
| 仲裁 | 分歧通常 10–20 条，30–45 分钟 |
| 日历工期 | 发出 → 两周内回收 → 评分与仲裁 1 天 → 写回函 1 天 |
| 费用 | 主要在找人；如需付酬，按当地小时费率 × 3 小时 × 2 人估 |
