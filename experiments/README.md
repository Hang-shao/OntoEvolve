# 实验脚本索引

全部脚本都在**仓库根目录**下运行：`python experiments/<script>.py ...`。
结果一律写到 `results/`（`src/config.py` 的 `DIR_RESULTS`），日志写到 `logs/`。

## 一、与论文的对应关系

| 论文位置 | 内容 | 脚本 | 产物 |
|---|---|---|---|
| Table 2 | 数据集与角色 | — | 见 `data/README.md` |
| Table 3、Table 4、Fig. 2 | 检测精度、告警质量（10 次重复） | `run_exp1.py` | `results/exp1_main.csv`、`results/exp1_summary.json` |
| Table 5、Fig. 3(a) | 组件消融 | `run_ablation.py` | `results/exp4_ablation.csv/.json` |
| Fig. 3(b) | 融合门 vs 优先级分派 | `run_fusion_gate_ablation.py` | `results/exp_fusion_gate/fusion_gate_main.json` |
| Table 6、Fig. 4 | 成本：增量 vs 逐批 / 周期重构（T = 36，五种子） | `run_cost_full5.py` | `results/exp_cost/exp_cost_t36_full5.json/.csv` |
| Table 7、Fig. 5 | 三类知识问答（五种子 × 两版本） | `run_exp_qa.py` | `results/exp_qa/`（含 `summary.json`、`answers.jsonl`） |
| 正文 4.6.2（散文，不设表） | 两个真实语料的结构性与告警级检查 | `run_exp_pmc.py`、`run_track2.py`、`tools/make_review_sheet.py` → `tools/analyze_alarm_annotation.py` | `results/exp_pmc/`、`results/track2_*.json`、`results/_annot/` |
| Table 8 | 反诈轨道维护 / 检测 / 问答三层 | `build_track4_stream.py` → `run_track4.py` → `run_track4_qa.py` | `results/track4/` |
| Table 9 | 可分性消融（只压缩类内相似度） | `run_track1_lowsep.py` | `results/track1_lowsep/` |
| Table 10、Fig. 6 | 第四基线：LLM 一次性重生成 | `run_track4_qa.py` + `run_track4_llm_regen.py` | `results/track4_llm_regen/` |
| ESM 1 Table S3 | 后验告警上限对已抑制流的再评分 | `analyze_selection.py` 的 `suppressed_*` 行 + `run_exp1.py --dump-alarms` | `results/alarms_cd*.jsonl`、`results/budget_analysis.json` |
| ESM 1 Table S5 | 统一选择规则（对无抑制原始告警流） | `analyze_selection.py` | `results/uniform_selection.json` |
| ESM 1 Table S6 | 匹配偏移三分与因果口径（分别以 `t_declare` 与 `t_start` 为参照） | `analyze_offsets.py` | `results/offset_detail.json` |

**补充材料 ESM 1 Table S1**（六项鲁棒性与消融检查）由多条轨道的旁证汇总，不是单个脚本：

| 检查 | 脚本 |
|---|---|
| 重定义阈值扫描（z ∈ {1.0, 1.5, 2.0, 3.0}） | `run_pmc_threshold_scan.py`（内部调 `run_exp_pmc.py --z`）→ `results/exp_pmc_ps1200_z*/` |
| 标定组合、AxiomJudge 开 / 关 | `run_track2.py --theta-red-mode {stat,transfer} --z ..`、`--llm-topk` |
| 公理有效性与规模（15 个 OWL 版本、HermiT） | `run_exp3.py` → `results/exp3.json` |
| 模板仲裁 / 支持衰减 | `run_ablation.py` 的 `no_support_decay` 等变体 |

> 上表列的是产出这些数字的脚本；推理机依赖的版本配对见 `scripts/fetch_hermit_deps.py`。

## 二、入口脚本

| 脚本 | 作用 | 主要参数 | 前提 |
|---|---|---|---|
| `run_exp1.py` | 阈值标定 + 五基线对比 + 配对检验 | `--repeats 10`、`--quick`、`--cooldown`（告警抑制窗口，标定与测试同设）、`--dump-alarms`（落盘逐条告警，供离线重算） | 嵌入服务 |
| `run_ablation.py` | 逐组件移除的消融，复用 exp1 的标定结果 | `--summary-in` | 先跑 `run_exp1.py` |
| `run_fusion_gate_ablation.py` | 打开 3.2 节的融合门，θ_score ∈ {0.35, 0.45, 0.55} | `--repeats 10` | 先跑 `run_exp1.py` |
| `run_exp3.py` | OWL 导出 + HermiT 一致性 + 注入矛盾对照 | `--engine hermit|z3|none`、`--quick` | JDK + `vendor/` 下 jar |
| `run_exp_cost.py` | 单种子成本实验（增量 / 逐批 / 周期） | `--tag` | 嵌入服务 |
| `run_cost_multiseed.py` | 多种子串行 + bootstrap 汇总 | `--seeds 0,1,2,3,4` | 嵌入服务 |
| `run_cost_full5.py` | **表 6 / 图 4 的口径**：单会话五种子、全批实测 + CPU 探针 | `--seeds`、`--tag`、`--smoke` | 嵌入服务；约 25 min/种子 |
| `run_exp5.py` | 资源消耗与回退率；`--engines` 的 z3/structural/none 分支是对照，按各策略自身裁决提交（关闭 fail-closed） | `--seeds`、`--engines`、`--alphas` | 嵌入服务 + 推理机 |
| `run_exp_qa.py` | 命题 C：三类知识出题、作答与评分 | `--seeds 0`、`--tag` | 嵌入服务（+ LLM judge 可选） |
| `run_static_eval.py` | 静态结构质量（NMI / 纯度 / SC / DBI） | — | 嵌入服务 |
| `run_exp_pmc.py` | 轨道 3（PMC）跨域验证 | `--per-slice 1200 --n-slices 12 --z 2.0` | PMC 语料 |
| `run_pmc_threshold_scan.py` | 轨道 3 重定义阈值敏感性（固定 per-slice = 1200） | — | PMC 语料 |
| `run_track2.py` | 轨道 2（ICEWS18）演化闭环 | `--per-slice 1200 --n-slices 24`、`--llm-topk`、`--theta-red-mode` | ICEWS18 |
| `build_track4_stream.py` | 物化轨道 4 的批次、问题与真值 | `--per-batch`、`--n-batches` | Telescam / ZhCCVi |
| `run_track4.py` | 轨道 4 的演化闭环与检测层评测 | — | 先 `build_track4_stream.py` |
| `run_track4_qa.py` | 轨道 4 的应用层问答（三种维护策略） | — | 先 `run_track4.py` |
| `run_track4_llm_regen.py` | 第四基线：LLM 一次性重生成整份本体 | `--per-type 2 --timeout 900` | 对话模型端点 |
| `run_track1_lowsep.py` | 可分性消融（只压缩类内相似度） | `--repeats 10` | 嵌入服务 |
| `run_exp_gowl.py` | 模块 G 的 OWL 2 DL 合法性与公理增长（ESM 1 Table S1 的"公理有效性与规模"行） | `--seed 0`、`--quick` | 嵌入服务 + 推理机 |
| `analyze_selection.py` | 在无抑制 / 已抑制的归档告警上施加统一选择规则（raw、前 N 条、每概念 1 条、去重加截断），重算 macro-F1 | `--results`、`--repeats` | 先 `run_exp1.py --dump-alarms` |
| `analyze_offsets.py` | 匹配偏移三分（早于漂移起点 / 有效提前 / 同步或滞后）与因果口径重算，因果窗口按 `t_declare`、`t_start` 两个参照各算一次 | `--results`、`--repeats` | 先 `run_exp1.py --dump-alarms` |
| `run_stage2.py` | 串行补跑：真实轨道阈值敏感性（ICEWS/PMC）、模块 G 的 OWL 合法性、一致性检查开销对照；串行是为避免计时互相污染 | — | 嵌入服务 + 推理机 |

## 三、诊断脚本（不产出论文数字，用于定位问题）

| 脚本 | 诊断对象 |
|---|---|
| `diag_abc.py` | 模块 A/B/C 在 D₀ 上的粒度与聚类质量（含 PCA 对照） |
| `diag_scale.py` | D₀ 规模对聚类的影响，与 scipy average-linkage 对照 |
| `diag_cut.py` | 动态剪切 vs 水平剪切 |
| `diag_emergence.py` | 涌现检测链路（未链接实例数、候选簇规模、与锚点距离） |
| `diag_full.py` | 全尺寸（T = 36）下 O₀ 质量与可检测性 |
| `diag_reasoner.py` | JDK / HermiT 是否可用，OWL 导出 + 一致性判定行为 |
| `diag_quality_match.py` | 增量 vs 全量重构的双向概念匹配 |
| `probe_rebuild_seed.py` | 单次全量重建的计时（判断某轮异常是数据效应还是机器降速） |
| `smoke_track1.py` | 轨道 1 全链路冒烟（建池 → O₀ → 检测 → 评估） |
| `dryrun_track2.py` | 轨道 2 前 3 个时间片干跑，实测成本 |
| `time_track4_rebuild.py` | 反诈轨道单次全量重构成本 |
| `run_cost_guarded.py` | 带 CPU 探针与反降速的逐种子成本运行器（夜间/怀疑降速时用） |
| `merge_cost.py` | **已废弃**：把分属两轮的成本结果拼成五种子（现口径用 `run_cost_full5.py`） |

## 四、图脚本（`experiments/figures/`）

| 脚本 | 产出 | 数据依赖 |
|---|---|---|
| `make_fig1_architecture.py` | `fig1_architecture.png/pdf` | 无 |
| `make_fig2_drift.py` | `fig2_drift_f1.*` | `results/exp1_summary.json` |
| `make_fig3_5_6.py` | `fig3_component_ablation.*`、`fig5_qa_comparison.*`、`fig6_antifraud_comparison.*` | `exp4_ablation.json`、`exp_fusion_gate/fusion_gate_main.json`；Fig. 5/6 的数值按正文表硬编码 |
| `make_fig4_cost.py` | `fig4_cost_speedup.*` | `results/exp_cost/exp_cost_t36_full5.json` |
| `export_submission.py` | `Fig1…Fig6.tif/.pdf`（上传命名） | 上面四张图 |

输出目录默认 `<repo>/figures`，可用 `ONTOEVOLVE_FIG_DIR` 覆盖（本仓库投稿时指向稿件目录）。

## 五、通用约定

- 所有入口脚本支持 `--quick` 或 `--smoke`（若有），用于小规模验证；
- 需要嵌入服务的脚本在 `ONTOEVOLVE_EMBED_API_*` 为空且没配本地模型时会直接报错，
  不会静默返回随机向量（`Embedder(allow_fallback=False)`）；
- 结果文件多数带 `tag` 后缀以便重跑留档；`run_exp1.py`、`run_ablation.py`、`run_static_eval.py`
  的输出文件名固定，重跑会覆盖旧文件，需要留档就先把 `results/` 里的文件改名；
- 计时类实验（成本、资源）不要并行跑，也不要与其它重负载进程同时跑；
  Windows 下还必须关掉进程级执行速度限制，否则单次计时会被拉长数倍。

## 六、最短复现路径

```bash
python -m pytest -q                                   # 单元测试
python experiments/run_exp1.py --quick                # 轨道 1 冒烟
python experiments/run_fusion_gate_ablation.py --repeats 3
python experiments/run_cost_full5.py --smoke          # 成本实验冒烟
```
