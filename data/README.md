# 数据说明

本仓库与随论文提交的代码包**均不附带任何语料**，只给出获取位置与来源。
四条轨道如下；轨道 1 完全由代码生成，不需要数据。

## 轨道 1（受控合成流）——无需数据

`src/vocab.py` + `src/controlled_stream.py` 按 `DriftParams` 生成：T = 36 批、600 个初始个体、
每批注入 9 个真值事件（涌现/淘汰/重定义各 3）、噪声比 0.15，漂移形态 gradual/periodic/lowfreq 各 0.34。
所有实验（表 3—7、图 2—5）都从这里来。**换机器不需要拷任何文件**。

## 轨道 2（ICEWS18）——公开数据集，需自行下载

- 论文口径：24 个时间片 × 1 200 条（末片不满），共 28 748 条事件三元组。
- 代码读取位置（`src/icews.py`）：

  ```
  <DATA_ROOT>/01_带时间戳的时序数据/TemporalKG_RE-Net/data/ICEWS18/{train,valid,test}
  ```

  `DATA_ROOT` 默认是**仓库父目录**下的 `data/`，可用环境变量 `ONTOEVOLVE_DATA_ROOT` 覆盖；
  也可以给 `run_track2.py` 传 `--root`。
- **来源**：ICEWS Coded Event Data，Harvard Dataverse，doi:10.7910/DVN/28075
  （ICEWS 事件编码数据；本文用的 18 个月切片即通称的 ICEWS18，目录结构沿用 RE-Net 一系的发布形式）。
  请按其原始许可使用。
- 本仓库**不重新分发**该数据。

## 轨道 3（PMC 生物医学全文）——公开语料，需自行下载

- 论文口径：14 400 句、1997—2001 年、per-slice = 1 200（基线口径）。
- 代码读取位置（`src/pmc.py`）：

  ```
  <DATA_ROOT>/03_领域语料/PubMed_Central_PMC_300
  ```

  `DATA_ROOT` 同上；`run_exp_pmc.py` / `run_pmc_threshold_scan.py` 支持 `--root`。
- **来源**：PubMed Central Open Access Subset，U.S. National Library of Medicine
  （https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ ）。请遵守 PMC 的使用条款；本仓库不重新分发。

## 轨道 4（中文反诈）——半合成，派生物不入库

- 材料：Telescam（`ScamGen.zip`，句子级 5 分类）与 ZhCCVi（中文译本，仅作噪声子集），
  均为公开数据集；批次顺序由作者按材料记载的演化次序人工编排，因此是**半合成闭环**。
- **来源**：两者的原始发布说明随材料自带（含发布方与使用条款），本仓库不复制其内容；
  论文若需要正式著录（作者、年份、DOI），按各自发布页填写。
- 构建链：

  ```bash
  python experiments/build_track4_stream.py     # 由原始材料物化 data/track4/{pool,batches,questions,truth.jsonl}
  python experiments/track4_audit.py            # 体检与类别语义核定
  ```

- 物化结果默认写在 `data/track4/`（约 40 MB，不入库，也不随稿件提交）。若只在本地复现，
  请把原始材料按 `src/fraud_stream.py` 顶部注释的说明放好。

## 目录约定

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `data/track4/` | 反诈轨道的派生物料 | 否 |
| `data/ontology/`、`data/processed/`、`data/temp/` | 中间产物（OWL 导出、切片缓存） | 否 |
| `data/cache*/` | 嵌入向量缓存（按批缓存，可达数百 MB） | 否 |
| `data/annotation/` | 告警人工复核标注 | 视许可 |

`src/config.py` 会在启动时自动创建缺的目录；缓存按"本地模型 / API 模型"分目录存放，
**不要合并**——两种向量维度同为 1024，共目录会静默复用错向量（见 `src/config.py` 注释）。
