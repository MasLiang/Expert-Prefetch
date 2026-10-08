# 当前主模拟器的真实模型结构复测

## 范围

本次复测使用当前 `UnifiedSequenceSimulator`，重新执行此前最近三套模型对照矩阵。
不复用历史 checkpoint，不覆盖历史结果，不调用已经弃用的 `M10AlignmentSimulator`。
这里的“真实模型”表示使用真实模型的 MoE 结构尺寸，并非执行下载的 checkpoint 或完整 Transformer。

| 模型 | H | F | Expert 数量 | Top-K | Expert 结构 | INT8 单 Expert 权重 |
|---|---:|---:|---:|---:|---|---:|
| Switch-base-8 | 768 | 3072 | 8 | 1 | 两矩阵 ReLU | 4.5 MiB |
| Switch-base-16 | 768 | 3072 | 16 | 1 | 两矩阵 ReLU | 4.5 MiB |
| OLMoE-1B-7B | 2048 | 1024 | 64 | 8 | fused Gate-Up SwiGLU | 6 MiB |
| Mixtral-8×7B | 4096 | 14336 | 8 | 2 | fused Gate-Up SwiGLU | 168 MiB |
| Mixtral-8×22B | 6144 | 16384 | 8 | 2 | fused Gate-Up SwiGLU | 288 MiB |

每次样本是独立冷启动的单 token、六个 MoE 层。OLMoE/Mixtral 采用六层截取，不代表这些模型的全部层数。
权重和激活 INT8、人工 routing/predictor trace、配置性能参数均是实验假设，不是板级测量。
没有 N:M pruning，也没有复现或测量 CEPR。

## 三套矩阵

| 子目录 | 原配置 | 样本模拟次数 | 配置点 |
|---|---|---:|---:|
| `switch_four_cases` | `switch_real_model_four_cases.yaml` | 2,400 | 24 |
| `switch_wide` | `switch_wide_tile_prefetch.yaml` | 2,400 | 24 |
| `olmoe_mixtral` | `olmoe_mixtral_cases.yaml` | 4,500 | 90 |
| 总计 | 最近三套对照，非全部历史 DSE | 9,300 | 138 |

Switch 四场景包含 A 整 Expert 无预取、B streaming 无预取、C 整 Expert 预取、D streaming tile 预取，
另保留 C/D 的上一层权重 barrier-on 消融。主要配置是 barrier-off，不重新引入已经移除的硬约束。

Switch 宽候选矩阵覆盖完整 Expert、Uniform tile 和 Ranked tile 预取。
候选覆盖全部 Expert 不等于置信度排序正确；完整 Expert 分配与 tile 分配使用同一配置的预取预算。

OLMoE 矩阵包含 streaming 与非 streaming 对照。Mixtral 的完整 Expert 大于配置的 8/16/24 MiB OCM，
因此只执行 streaming；其中 `full_expert_prefetch` 表示完整 tile 集合的预取目标，可受容量截断，
不能解释为“整个 Expert 同时驻留后才开始计算”。Mixtral-8×22B 的全部 INT8 routed 权重也超出 U200 外存，
本矩阵不构成单卡部署可行性证明。

## 硬件参数与可比性

Switch：四通道、每通道名义带宽 8/16 byte/cycle、32 ops/cycle、16 MiB OCM、4 KiB burst、
128 的 F 维 tile、100 个固定 seed 样本。原配置有 payload multiplier=2，名义带宽不能直接当成有效逻辑权重带宽。

OLMoE/Mixtral：四通道、每通道名义 16 byte/cycle、32/512 ops/cycle、8/16/24 MiB OCM、
128 的 F 维 tile、50 个固定 seed 样本。原 profile 使用 matrix-fragment 非抢占 DMA，没有启用 burst 仲裁。

以上保持原研究 profile 的实际 byte OCM 和 `expert_locked` 执行规则，不强制替换为固定 slot 或 GU/DOWN pair RTL profile。
本次目标是在相同硬件假设下检查更新后的模拟器；不是改硬件参数后与旧结果比较。
不同 backend 的绝对周期不能直接横比；加速比必须明确分母。

## 运行与输出

从工程根目录执行：

```bash
PYTHONPATH=src python examples/retest_real_models.py --workers 4 --parallel-experiments 3
pytest -q
```

`workers` 是每套实验的主机进程数，不是硬件 PE 数。默认结果根目录是：
`results/current_model_retest_20261007/`。若保留多个复测版本，可使用 `--output` 指定一个新的目录。

每套目录保留原始输入 JSON、逐样本与聚合 CSV、summary、PNG/PDF 图。
根目录的 `REPORT.md` 是生成的实际结果报告；`headline_comparison.csv` 包含代表配置；
`sample_deltas.csv` 包含完整旧/新配对差值；`summary.json` 包含总数、输入一致性与变化范围。
通过入口驱动运行时，根目录还有三份子实验日志。

若三套实验已分别运行完成，只汇总而不重复模拟：

```bash
PYTHONPATH=src python examples/retest_real_models.py --collect-only
```

汇总会拒绝未完成结果、重复或缺失样本、变化的激励、非主模拟器结果、
不匹配的当前源文件 SHA-256，以及缺失的 PNG/PDF 图。周期差定义为“当前减历史”，负值代表节省。
投机 byte 差单独记录，不作为周期差使用。

## 代码回归

2026-10-07 在当前工作区执行全量测试：`547 passed, 14 warnings`。
警告来自已有 Python 依赖的弃用提示，无失败测试。新增的四个测试检查配对差值以及重复、
身份不一致、缺失样本的拒绝行为。本次没有修改主模拟器、RTL 或 XDC 的执行语义。
