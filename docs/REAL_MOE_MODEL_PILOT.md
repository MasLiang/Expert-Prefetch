# 真实 MoE 配置的主模拟器实验

## 范围

这是首次使用真实模型的 H/F/E/K 运行 `UnifiedSequenceSimulator` 的规模化 pilot，保留现有调度算法和模拟器语义，不使用已失效的 M10AlignmentSimulator。不下载模型权重，也不运行语言模型。

必须区分：真实架构尺寸、假设部署精度、合成路由和预测、假设硬件性能。此次输出不是完整 LLM 的 tokens/s、准确率或 U200 板上实测。

## 模型与配置来源

| 模型 | 发布总参数量（近似） | H | 单 Expert F | 每层 E | Top-K | MoE 层数 | 单 Expert 参数量 | 单 Expert INT8 权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMoE-1B-7B-0924 | 7B | 2048 | 1024 | 64 | 8 | 16 | 6,291,456 | 6 MiB |
| Mixtral-8×7B-v0.1 | 46.7B | 4096 | 14336 | 8 | 2 | 32 | 176,160,768 | 168 MiB |
| Mixtral-8×22B-v0.1 | 141B | 6144 | 16384 | 8 | 2 | 56 | 301,989,888 | 288 MiB |

结构来自官方发布者仓库的固定版本配置：

- [OLMoE config.json](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/83792c65a757954d4dfbd677732dd914397f5d69/config.json)
- [Mixtral-8×7B config.json](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/58301445dc1378584211722b7ebf8743ec4e192b/config.json)
- [Mixtral-8×22B config.json](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/76fa59362cd95eb633a4e419c24364c90674966c/config.json)

总参数量为发布的舍入值，不把它当作 tensor 精确统计。单 Expert 参数量精确按 `3*H*F` 推导；全模型 routed Expert 参数量按 `L*E*3*H*F` 推导，分别为 6,442,450,944、45,097,156,608、135,291,469,824。不能把发布名称中的“8×7B”理解成八个 7B FFN Expert。

总量参考：[OLMoE 官方模型卡](https://huggingface.co/allenai/OLMoE-1B-7B-0924)、[Mixtral-8×7B 发布说明](https://mistral.ai/news/mixtral-of-experts/)、[Mixtral-8×22B 官方模型页](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)。`model_geometry.csv` 的 active_expert_parameters_per_layer 仅包含该层被激活的 routed Expert 权重，不等于全模型 active parameters。

源配置为 BF16；本次假设权重与 activation 使用 INT8，未验证量化准确率，也未计入 scale/zero-point 等元数据。OLMoE 的 gating normalization 与 Mixtral 不同；本次只使用 Top-K Expert 身份建模时序，不验证 Router 数值等价。

Mixtral-8×22B 的 routed Expert 权重在 INT8 下已占 126 GiB，超过单卡 U200 四个 16 GB DIMM 的容量，因此它是调度压力案例，不是单卡完整部署方案。[U200 DDR 容量来源](https://docs.amd.com/r/en-US/ug1289-u200-u250-reconfig-accel/DDR4-DIMM-Memory)

没有纳入 Qwen/DeepSeek 的 shared Expert 或其他异构结构，避免静默忽略未覆盖的计算依赖。适配器会拒绝未知架构或带 shared Expert 的配置。

## 实验配置

完整输入见 `configs/experiments/real_moe_pilot.yaml` 与 `configs/models/real_moe_catalogue.yaml`。

- 一个 decode token，默认模拟四个连续 MoE 层，而非全部 Transformer 层。
- 模型的所有真实 MoE 层数保留为元数据；可以显式开启全 MoE 深度。
- 每层随机选择 K 个不同 Expert，固定 seed=20261005；所有策略共享完全相同的真实路由。
- 候选数为 2K：OLMoE 为 16，两个 Mixtral 为 4。
- perfect：候选覆盖全部真实 Expert；partial：覆盖一半真实 Expert，候选 recall=0.5。命中候选排在错误候选之前，因此它是偏乐观的缺失候选实验，不是随机排名退化或真实 predictor。
- source Router 完成后 20 cycles 产生下一层预测；等待前一层权重读取完成后才能释放 future prefetch，并受有限 OCM、descriptor 和 FIFO 约束。
- 不预测首层。基线是 no_prediction 的精确 demand tile streaming，不是弱串行 baseline。
- 比较现有 uniform_prefix 与 ranked_prefix，prefix_depth=2。ranked 的请求深度为 2、1、0…；两种策略有相同容量上限，但不保证申请或传输 byte 相同，不能作为等预算 allocator 最优性证明。

### 硬件假设

以 `u200_ingress_alignment.yaml` 的控制规则为基础：单共享逻辑计算 Engine、四内存通道、非抢占 DMA、GU/DOWN pair admission、tile-ready-first、有限 descriptor/FIFO、动态 ingress retirement。

core clock 为 200 MHz；每通道配置 3.2 GB/s，四路总接口配置为 12.8 GB/s；保留快速 RTL 后端的 `dma_payload_cycle_multiplier=2`，因此不是 12.8 GB/s 持续有效带宽，更不是板上 DDR 带宽实测。

比较 32 和 512 ops/cycle。前者沿用小型 RTL 的吞吐输入；后者只是扩展 PL 计算吞吐的假设，不代表已综合实现的 PE 阵列。历史字段名 `aie` 在 U200 study 中表示逻辑 PL 计算资源，U200 本身没有 AIE。

F tile=128；单 GU slot 根据 `2*H*F_tile*weight_bits/8` 扩大，DOWN 仍占同尺寸 slot。requested OCM 为 8/16 MiB，向下取整到完整 bank/slot，不能将 requested capacity 当作实际可用 byte。例如 Mixtral-8×22B 的实际容量为 6/15 MiB。

OCM 是专用于 Expert 权重的研究预算，不包含 Router 权重、activation、accumulator 或其他片上资源。Router 权重按模型默认视为常驻，不产生加载 DMA。

逻辑 burst 从现有 RTL 的 256 byte 放大到 16 KiB，以减少规模化实验事件数。保留 burst 仲裁，但不宣称现有 RTL 支持这个配置，也未用 DDR4 controller 样本校准服务时间。因此这次属于可配置 scale-study，不属于已完成逐周期 RTL 对齐的配置范围。未来若作硬件结论，需对 burst、计算吞吐和实际 buffer 配置做 sensitivity/RTL 验证。

## 输出与指标解释

`results/real_moe_pilot/`：

- `model_geometry.csv`：官方维度、参数推导、部署大小和来源 URL。
- `runs.csv`：60 组运行，即 3 模型 × 2 容量 × 2 吞吐 ×（1 基线 + 2 quality × 2 policy）。
- `per_layer.csv`：240 行层级 start/Router resolve/done 时间。
- `<model>_<quality>_input.json`：6 份固定 seed 的输入路由与候选记录。
- `summary.json`：主模型名称、完整配置、源码 SHA-256 和限制声明。
- `speedup.png/pdf`：requested OCM=8 MiB 下的吞吐敏感性图。

`simulated_moe_ms` 仅是这四层 MoE 子图的时长，不是单 token 完整延迟；attention、KV cache、norm、embedding、lm_head 和外部通信均未计入。inter-layer gap 未加入真实 attention 工作。

`aie_starvation_cycles` 沿用主模型的逐 Expert 等待求和口径，等待区间可以重叠；它不是单共享 Engine 的互斥 idle 区间总长，不能与 busy cycles 相加作整段时间守恒。

## 复现

```bash
python examples/real_moe_pilot.py --plot
pytest -o addopts= -q
```

`--plot` 需要额外安装 matplotlib；未安装会明确报错，不静默跳过。模拟器自身不新增大型框架或模型推理依赖。

只跑一个模型：

```bash
python examples/real_moe_pilot.py --models olmoe_1b_7b --output results/olmoe_pilot
```

显式模拟全部 MoE 层：

```bash
python examples/real_moe_pilot.py --full-moe-depth --output results/real_moe_full_depth
```

最后一个命令只增加 MoE 层数量，仍不是完整 LLM 推理，本轮未将它作为已完成结果。

## 本轮完成的结果

60 组全部完成，240 行 per-layer 时间记录已生成；在首轮完成时核验了所有记录的源码 SHA-256，所有 peak OCM 均不超过实际可用容量。7 个新增测试及当时全库共 349 个测试通过。

下表为 requested OCM=8 MiB、perfect 候选、uniform_prefix，参照同一尺寸/硬件/路由的 no_prediction 基线。Mixtral-8×22B 实际可用容量为 6 MiB，不把表中的 requested 8 MiB 误当作 usable。

| 模型 | ops/cycle | 无预测 cycles | uniform cycles | speedup |
| --- | ---: | ---: | ---: | ---: |
| OLMoE-1B-7B | 32 | 12,689,469 | 12,639,981 | 1.003915× |
| OLMoE-1B-7B | 512 | 6,336,409 | 6,340,840 | 0.999301× |
| Mixtral-8×7B | 32 | 88,246,797 | 88,147,869 | 1.001122× |
| Mixtral-8×7B | 512 | 44,312,729 | 44,324,456 | 0.999735× |
| Mixtral-8×22B | 32 | 151,235,085 | 151,522,263 | 0.998105× |
| Mixtral-8×22B | 512 | 75,959,449 | 75,986,347 | 0.999646× |

整个矩阵的 speedup 范围为 0.9981047×–1.0039152×。真实 Expert 很大、只请求前两个 tiles、有限 credits 及单共享 Engine 的条件下，没有观察到显著延迟收益；部分点发生小幅退化。不能将这些不到 1% 的差异称为已经 RTL 验证的优势，尤其本次配置已超出既有小尺寸对齐范围。

所有匹配条件下 uniform/ranked 的总周期一致，perfect/partial 的总周期也一致；这不是对算法等价或 predictor 无关性的证明。partial 将命中排在前面，前缀请求、有限资源及关键路径会掩盖候选后半部分的差异。不同策略/quality 仍可能产生不同 byte 成本：例如 Mixtral-8×7B、16 MiB requested、32 ops/cycle、partial 的错误 speculative bytes 为 uniform 9,437,184，ranked 4,718,592，而总周期相同。因此下一轮必须分别评价延迟与 byte 浪费，并增加排名退化、深度/预算和 burst 敏感性条件。

`speedup.png/pdf` 中多条曲线重合是实际数据相同，不是缺少运行；图只显示 requested 8 MiB 切片，16 MiB 结果在 CSV 中。

随后新增完整 Expert 目标预取的容量/完成度诊断，未改变主模型语义；历史 60 组保留其原始 wrapper hash，不把新增诊断的当前版本说成原运行快照。补跑命令、容量口径和独立结果见 [完整 Expert 目标比较](REAL_MOE_FULL_EXPERT_COMPARISON.md)。

## 后续真实 trace 阶段的建议

当前模型维度已真实，但 predictor 效果并非来自模型。若要评价真实工作负载，需要运行模型收集每 token、每 layer 的 Top-K truth，以及独立 predictor 的候选排名、置信度和产生时机。应加入多个 prompt/token、多个种子、错误候选前置或排名退化，而不是仅依赖本次命中前置的 0.5 recall 条件。

对接真实 trace 后才能评价实际模型路由分布、跨层相关性、预测质量及 confidence calibration；再选代表性点，用可实现的 RTL 尺寸/带宽及 DDR4 后端验证。
