# 邀请标注者的邮件模板

## 中文版

主题：请求帮忙做一次 2–3 小时的独立文本判定（告警复核，署名致谢）

XX 老师 / XX：

您好。我们有一篇关于「流式文本上自动构建本体并检测概念漂移」的论文正在审稿修改，审稿人要求把
告警精度的人工复核从单人非盲升级为**双人独立盲评**。想请您做其中一位标注者。

需要您做的：打开一个 85 行的 CSV，每行是一段被判为“概念漂移”的文本证据，判断这段证据是否支持
系统所声称的漂移类型（新概念出现 / 概念消失 / 概念含义改变）。三选一 + 一个置信度，逐行填。
随附的 `rater_instructions_*.md` 写清了判定规则和 5 个虚构示例。**不需要任何代码或环境**，
预计 2–3 小时，可分两次做。

三点说明：

1. **独立**。另一位标注者拿到的条目相同但顺序不同；在您交回之前，请不要与任何人讨论具体条目。
2. **只依据表内信息判断**：不上网、不查原始语料、不读论文正文。两个语料（PMC、ICEWS18）的许可
   不允许再分发，所以我们只能给您短引句。
3. **不必求全**。有 4 行没有证据、若干行证据很碎，`无法判断` 是合法的答案，也是我们最需要的诚实结果。

时间上不急，两周内给我就够。数据包请勿转发；交回并确认后请删除副本。如果您同意，我们会在致谢中
具名（也可以匿名，请示下）。若需要，我们可以按小时支付评审费 / 以劳务形式致谢。

如愿意，我这边可以直接把 `rater_R2.csv`（或 `rater_R3.csv`）+ 说明发给您。

祝好
XXX
2026-XX-XX

---

## English version

Subject: Request for a 2–3 hour independent text-judgement task (alert review; acknowledgement offered)

Dear XX,

We have a paper under revision on incremental ontology construction and concept-drift detection over
text streams. A reviewer asked us to upgrade the human check of alert precision from a single
non-blind pass to **two independent, blind ratings**. I would like to ask you to be one of the two
raters.

What it involves: an 85-row CSV. Each row contains the stream evidence behind one alarm, together
with the drift type the system claims (a new concept appeared / a concept disappeared / a concept
changed meaning). For each row you record one of three verdicts plus a confidence level. The attached
`rater_instructions_EN.md` gives the decision rules and five synthetic examples. **No code, no
environment, no data access** — about 2–3 hours, splittable across two sittings.

Three points:

1. **Independence.** The other rater receives the same items in a different order. Please do not
   discuss individual rows with anyone until you have returned your sheet.
2. **Judge from the packet only**: no web search, no corpus lookup, no reading the paper. The two
   corpora (PMC, ICEWS18) are used under terms that do not permit redistribution, so we can only
   pass on short quoted excerpts.
3. **You do not have to be complete.** Four rows carry no evidence and several are fragmentary;
   `undecidable` is a legitimate verdict and is more useful to us than a guess.

There is no rush — within two weeks would be fine. Please do not forward the packet, and delete your
copy after returning it. We will name you in the Acknowledgements unless you prefer otherwise; we can
also pay an honorarium or acknowledge the work formally.

Happy to send `rater_R2.csv` (or `rater_R3.csv`) plus the instructions whenever you say go.

Best wishes,
XXX
2026-XX-XX
