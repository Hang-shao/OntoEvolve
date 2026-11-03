# OntoEvolve

围绕「LLM 建好的本体怎么持续演化」的实验代码：把流式语料里的变化判成**涌现 / 淘汰 / 重定义**三类漂移，
用增量算子更新本体，并在每次提交前做全本体一致性检查、失败即回滚。

论文：*Incremental evolution and typed drift detection for automatically built ontologies*（Discover Computing，投稿中）。

## 做什么

- **模块 D｜漂移检测**：语义、统计、公理签名三路信号按优先级分派，而不是加权融合；三类漂移各有自己的判据。
- **模块 E｜一致性与回滚**：候选版本依次过结构预检查 → OWL-DL 推理（HermiT，Pellet 为备选）→ 受限 SMT（z3），任一层报不一致即回滚；只有「无结论」才降级到下一层。
- **模块 G｜模板抽取**：七类模板确定性产出候选公理并附证据置信度，LLM 只给置信度判定、可关闭。
- **模块 A/B/C（`legacy/`）**：冷启动本体 O₀ 的聚类基座（局部密度聚类 + 自动簇数 + 动态剪切）。

## 仓库结构

```
ontoevolve/
├── src/            方法实现（模块 D/E/G 及其评估）
├── legacy/         模块 A/B/C：冷启动聚类基座
├── experiments/    实验与诊断入口（见 experiments/README.md 的表/图→脚本映射）
├── tests/          pytest 单元测试
├── tools/java/     HermiT 检查入口（绕开 OWLAPI 的 parser SPI）
├── scripts/        依赖下载、打包发布
├── data/           数据目录（语料不入库，见 data/README.md）
└── results/ logs/  运行产物（不入库）
```

## 安装

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # 填入嵌入服务端点与密钥
```

`requirements-optional.txt` 只在用本地嵌入模型（torch/transformers）时需要。推理机相关：

```bash
python scripts/fetch_hermit_deps.py --owlapi 5.1.20
python scripts/build_hermit_cli.py
```

## 跑起来

```bash
python -m pytest -q                                   # 单元测试
python experiments/run_exp1.py --quick                # 小规模冒烟
python experiments/run_exp1.py --repeats 10           # 检测实验正式运行
python experiments/run_exp_cost.py --full-every 3 --period 5   # 增量 vs 全量重构
```

每个 `experiments/*.py` 都支持 `--help`。完整的实验→脚本→产物对照见 `experiments/README.md`。

## 环境依赖与跳过

单元测试不依赖本机数据：缺 JDK/HermiT、缺嵌入服务、缺 PMC 语料时，对应用例自动跳过而不是失败
（`pytest -rs` 显示原因）。干净检出应得 `104 passed, 15 skipped`，配齐环境后为 `119 passed`。

## 数据

ICEWS18、PMC、Telescam、ZhCCVi 等语料按各自许可**不随仓库分发**；获取方式、目录约定与派生文件的说明见
`data/README.md`。`ONTOEVOLVE_DATA_ROOT` 可把数据根指到仓库外。

## 发布打包

```bash
python scripts/package_release.py --check    # 密钥 / 绝对路径扫描
python scripts/package_release.py            # 生成 dist/ontoevolve-<日期>.zip
```

排除 `.env`、`vendor/`、`data/`、`results/`、`logs/` 与缓存。

## 引用与许可

- 论文录用后在此补 citation 与 DOI。
- 许可证：**MIT**，全文见 `LICENSE`。选择 MIT 而非 Apache-2.0 的理由是不含专利授权条款，在作者与所在单位的权利归属书面确认之前不再扩大授权范围；若后续需要专利明示授权，可换成 Apache-2.0 并同步更新本行。
- 模块 A/B/C 来自作者前期的聚类工作，随本仓库一并发布。
