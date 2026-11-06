# Rating instructions — alert review for a drift-detection study

Thank you for helping. This note tells you exactly what to judge, how to record it, and
what not to do. It should take 2–3 hours for 85 rows; you can split it across sittings.

If anything is unclear, ask the person who sent you this packet **before** you start
rating, and not about individual rows.

---

## 1. What you are rating

The study monitors a text stream that arrives in batches and builds an ontology (a set of
concepts and typed axioms) as it goes. A detector raises an **alarm** when it believes one
of three kinds of change has happened to a concept:

| Alarm type | The claim the system makes |
|---|---|
| `emergence` | a **new** concept appeared at batch `t`, and the listed stream items are its first members |
| `extinction` | a concept that existed **stopped appearing** |
| `redefinition` | an existing concept is still in use but its **meaning or structure changed** |

Your job is one question per row: **does the quoted evidence support the claim as stated?**

You are not asked whether the system should have raised the alarm, whether the threshold is
right, or whether the ontology is a good ontology. Only whether this alarm, as claimed, is
supported by what is quoted.

## 2. What is in the sheet

One row per alarm, 85 rows. You fill three columns and nothing else:

| Column | Meaning |
|---|---|
| `label` | your verdict: `true drift`, `false alarm` or `undecidable` (Chinese accepted too: `真漂移` / `误报` / `无法判断`) |
| `confidence` | your confidence in **that verdict**: `high`, `medium`, `low` (`高` / `中` / `低`) |
| `note` | optional, one line, English or Chinese. Skip it if the verdict is obvious |

Context columns, for your information only:

| Column | Meaning |
|---|---|
| `seq` | row number in your sheet. The two raters get different orders; do not re-sort |
| `alarm_id` | stable identifier. **Use this to talk about an item**, not `seq` |
| `track` | `PMC` = biomedical abstracts (1997–2001); `ICEWS18` = coded political-event records |
| `t` | batch index at which the alarm was raised |
| `dtype` | the claimed change, as in the table above |
| `concept` | the concept the alarm is about. On the PMC track these are anonymous internal labels (`topic_NEW905`), because the study does not name its clusters; on ICEWS18 they are event predicates (`Make an appeal or request`). **An opaque label means "no prior knowledge available", not "suspicious"** |
| `n_members` | how many stream items the system attached to the concept at that batch |
| `last_seen_t` | last batch in which the concept had been seen before (may be empty) |
| `evidence` | the quoted stream text the system relied on. Sentences are separated by line breaks. On PMC these are abstract sentences; on ICEWS18 they are coded event records of the form *actor — verb — target* |

Four rows carry **no evidence at all**. That is a property of the alarm type, not a mistake
in the packet: for those rows `undecidable` is the expected answer unless the other columns
already settle it.

## 3. Decision rules

Use these rules literally. When two rules seem to apply, the earlier one wins.

### `emergence` — is this a genuinely new *kind*?

1. **`true drift`** — the quotes describe a coherent type-level entity, process, relation or
   property that the text itself treats as a distinct kind, and that is not merely one more
   instance of an ordinary category. You are judging the quoted material only; you do not
   need to know what the ontology already contains.
2. **`false alarm`** — the quotes are any of: (a) a single instance's data point or
   measurement; (b) figure or table captions, methods boilerplate, or procedural sentences;
   (c) a restatement or paraphrase of a category the text already treats as ordinary;
   (d) a tokenisation artefact — fragments that name nothing; (e) several mentions of the
   same ordinary event or entity under different names.
3. **`undecidable`** — the quote is too short or too fragmentary to classify, or
   `n_members` is 1 and the single quote could be read either way.

### `extinction` — is the disappearance supported?

1. **`true drift`** — the quoted material shows the entity or type being withdrawn,
   renamed, superseded, retracted, or explicitly ceasing.
2. **`false alarm`** — the quote is an ordinary mention (so the concept was still in use),
   or the concept is a generic category for which "stopping" is not meaningful.
3. **`undecidable`** — no evidence, and nothing else in the row settles it. This is the
   usual answer for an extinction row with an empty `evidence` cell: **absence cannot be
   observed from a quote**, and we would rather have an honest `undecidable` than a guess.

### `redefinition` — did the meaning change?

1. **`true drift`** — the quotes show the same named concept used with a materially
   different definition, participants or structure than before.
2. **`false alarm`** — the quotes show an ordinary use consistent with the existing
   meaning, or they are about a different concept altogether.
3. **`undecidable`** — there is no contrast to judge from.

### General

- Do not use a quota. The packet contains many more false alarms than true ones; answer row
  by row and let the totals fall where they fall.
- `undecidable` is a legitimate answer, not a failure to perform. Use it for empty or
  fragmentary evidence and for genuinely balanced readings — not for "I would have to look
  at the corpus".
- Judge the row, not the system.

## 4. Worked examples (synthetic, not from your packet)

**A. `emergence`, evidence: three sentences describing reverse transcription as a mechanism
that copies RNA into DNA, contrasted with the text's ordinary categories "DNA replication"
and "transcription".**
→ `true drift`, high. A type-level mechanism, described as distinct from the ordinary ones.

**B. `emergence`, evidence: one sentence reporting that a protein's half-life was 42 min in
the treated group.**
→ `false alarm`, high. A single measurement, instance level.

**C. `emergence`, evidence: "Figure 3. Effect of treatment on expression (n = 6)." and "Cells
were washed twice before lysis."**
→ `false alarm`, high. Caption and methods boilerplate; nothing names a kind.

**D. `emergence`, evidence: "…consistent with the earlier observation described above."**
(single truncated fragment, `n_members` = 1)
→ `undecidable`, medium. Too little to classify.

**E. `extinction`, no evidence, `n_members` = 0.**
→ `undecidable`, high — unless `last_seen_t` plus the concept name make the claim plausible
on its own, in which case say so in `note`.

## 5. Rules of the exercise

- **Rate alone.** Do not discuss any row with the other rater, with the authors, or in a
  group, until both sheets have been returned. The value of the exercise is that the two
  opinions are independent.
- **Do not look anything up.** Judge from the packet only: no web search, no corpus lookup,
  no reading the paper or the repository. This keeps your sheet comparable to the other
  rater's, and the source corpora are used under terms that do not permit redistribution.
- **Do not consult any other file** the sender may have shared earlier (in particular any
  file whose name contains `llm` or `review`). If you have such a file, tell the sender and
  ignore it.
- **Do not change the sheet's structure**: no reordering, no deleting rows, no new columns.
  Fill only `label`, `confidence`, `note`. If you use a spreadsheet program, save back as
  CSV UTF-8; keep the header row unchanged.
- **Take breaks.** Accuracy falls after about an hour of this. Two sittings of 45 minutes is
  better than one of 2.5 hours.

## 6. What happens to your sheet

- Both sheets are merged by `alarm_id` and scored for agreement (Krippendorff's α,
  Cohen's κ, positive and negative agreement, with bootstrap intervals).
- Rows where the two of you disagree are put to a third person for adjudication. Your
  `note` is **not** passed to that person, so that wording cannot sway the ruling.
- Your ratings are reported as **unsigned aggregate statistics**. You are named in the
  Acknowledgements unless you ask not to be. The paper describes the review as
  "a limited human review of detected alarms", never as a validation of the system.
- Do not redistribute the packet. Delete your copy once the sheet has been returned and the
  ratings have been confirmed.

## 7. How to return it

Send back the same CSV file, with `label` filled on every row, `confidence` filled on every
row, and the filename unchanged (`rater_R2.csv` or `rater_R3.csv`). If a row feels
impossible, still give it a label — `undecidable` — and say why in one line if you wish.
