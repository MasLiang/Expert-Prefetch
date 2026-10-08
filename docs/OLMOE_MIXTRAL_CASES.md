# OLMoE 与 Mixtral 多 case 对照

## 配置与范围

运行 `python examples/olmoe_mixtral_cases.py`；回归 `pytest -q`。
实验配置为 `configs/experiments/olmoe_mixtral_cases.yaml`，基础硬件继承
`configs/experiments/switch_prefetch.yaml`。结果保存至 `results/olmoe_mixtral_cases/`，
不覆盖 Switch 实验。

默认四个主机 worker，并非四个硬件 PE；样本独立，父进程按确定顺序汇总。
`--workers 1` 可单进程复现。`--resume` 用于同一配置、同一代码下的中断运行：
复用无预测样本，预测样本重跑后必须与 checkpoint 每字段一致，同时补齐时序。
不要修改硬件参数后复用旧 checkpoint；应使用新的输出目录重新运行。

每模型 50 个独立单 token、六个连续 MoE 层序列，非完整模型深度。
OCM 预算 8/16/24 MiB，单逻辑计算引擎配置 32/512 ops/cycle，
200 MHz、四通道各 3.2 GB/s、效率1、DMA setup8 cycles、计算 startup8 cycles，
F tile128，非线性吞吐16 elements/cycle；均为可配置假设，不是 U200 实测。
按实际 byte 计 OCM，无全权重 barrier；DMA 仍按整个矩阵 tile 不可抢占，
未启用 16 KiB burst 边界仲裁。没有新增 pruning、CEPR 或调度算法。

| 模型 | H | F | E | K | 候选数 | INT8 单 Expert | GU/Down tile | F groups |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| OLMoE-1B-7B | 2048 | 1024 | 64 | 8 | 16 | 6 MiB | 512/256 KiB | 8 |
| Mixtral-8×7B | 4096 | 14336 | 8 | 2 | 4 | 168 MiB | 1024/512 KiB | 112 |
| Mixtral-8×22B | 6144 | 16384 | 8 | 2 | 4 | 288 MiB | 1536/768 KiB | 128 |

三模型均为 fused Gate-Up SwiGLU，不是 Switch 两矩阵 ReLU。
源结构固定 revision 见 `configs/models/real_moe_catalogue.yaml` 和输出 geometry。
INT8 是部署假设，不代表原 checkpoint 精度或已验证量化准确率。

## 预测与公平性

所有策略共享同一 truth、候选集合、候选排序、score 和 ready 时序。
覆盖率按 `|candidate ∩ true|/K` 计算，不把 any-hit 当作 recall。
OLMoE 250 个目标共2000次真实 Expert 决策，1688次候选命中；
Mixtral 每模型共500次决策，422次命中，均为84.4%。
该比例只是受控合成质量假设，不是这些模型上的真实 CEPR 表现。
命中位置随机混入候选，score 随 rank 递减，不强制将真 Expert 排在最前。

预测在前一个 MoE 层 Router 完成后20 cycles可用，立即尝试 admission，
不等待前一层全部权重加载完成。第一层无预测；无跨 token cache 复用。

Uniform 与 Ranked 采用相同 prefix_depth=2 的既有算法：
Uniform 各候选最多2个 GU tiles，Ranked 前两候选2/1个，其他0个。
因此它们不是等 speculative byte 预算的 allocator 对照；容量和总预算一致，
实际请求量可能不同。相同 recall 也不代表相同 ready recall。

## Case

OLMoE 七种：完整按需加载、串行 tile 按需、demand tile streaming、
非流水完整目标预取、流水 all-tiles 目标预取、Uniform prefix、Ranked prefix。
非流水模式要求当前真实 Expert 全权重 ready 后才计算；Top-8 顺序执行，
并不要求八个真实 Expert 同时驻留。

Mixtral 仅四种 streaming：无预测、all-tiles 预取目标、Uniform、Ranked。
168/288 MiB Expert 无法在当前预算完整驻留；两个 whole-residency case 明确
列为不可行，不伪装成 tile case，也不为其计算 speedup。
CSV 中 `full_expert_prefetch` 是既有 policy 键，在 Mixtral 中只能解释为
“All-Tiles Target + Streaming”：计划会被 byte budget/容量截断，绝不意味着
整个 Expert 已驻留。实际完整加载完成计数可查询 CSV。

## 输出和分母

- `comparison.csv`：90个聚合点，OLMoE42、每个Mixtral24。
- `per_sample.csv`：4500次模拟，OLMoE2100、每个Mixtral1200。
- `per_prediction.csv`：各预测目标的 ready、release、Router 和 byte metrics。
- 每模型输入 JSON，model_geometry.csv，summary/config/source hashes。
- 两组 PNG/PDF：相对 streaming，以及 OLMoE 相对完整按需 baseline。

`speedup_vs_streaming` 始终以本模型相同硬件的 demand tile streaming 为分母对应
baseline（其cycles / 当前cycles）。`speedup_vs_whole_demand` 仅OLMoE有值，
Mixtral为空，不使用别的baseline补数。`speedup_vs_full_target_streaming`
对比 all-tiles streaming目标，不是 FLAME-inspired whole-residency baseline。

## 部署边界

本实验不包含 attention、KV、embedding、共享/非MoE计算或通信。
Mixtral-8×22B 的全模型 routed INT8 权重约126 GiB，超过U200的64 GB外部DDR；
这里仅模拟六层性能，不声称完整模型可部署单卡。
所有速度比是配置下的趋势，不是芯片实测或 FLAME 精确复现。

## 完整实验结果

本轮全部完成：90个配置点、4500条独立配置/样本结果、15000条预测目标诊断。
中断续跑复用了1000条已完成的非预测样本；已缓存预测样本重新执行并逐字段
核对一致，避免丢失预测诊断记录。这里的4500是最终唯一结果数，不是续跑新增数。

下表选取16 MiB OCM；速度比均相对同配置的**无预测 tile streaming**。
数值小于1表示变慢。All-Tiles是预取目标+streaming，不是全Expert就绪门槛。

| 模型 | ops/cycle | All-Tiles | Uniform | Ranked |
|---|---:|---:|---:|---:|
| OLMoE | 32 | 1.00498× | 1.01294× | 1.00470× |
| OLMoE | 512 | 0.96479× | 0.93199× | 0.98563× |
| Mixtral-8×7B | 32 | 1.00000× | 1.00061× | 1.00046× |
| Mixtral-8×7B | 512 | 0.99718× | 0.98957× | 0.99676× |
| Mixtral-8×22B | 32 | 1.00000× | 1.00043× | 1.00041× |
| Mixtral-8×22B | 512 | 0.99803× | 0.99391× | 0.99716× |

OLMoE的另一组比较，以**完整按需加载后计算**为弱baseline：

| OCM | ops/cycle | 非流水完整Expert预取 | 无预测streaming | Uniform streaming预取 |
|---|---:|---:|---:|---:|
| 16 MiB | 32 | 1.01649× | 1.22721× | 1.24309× |
| 16 MiB | 512 | 1.02870× | 1.23848× | 1.15426× |

不能将相对弱baseline的约1.24×解释成预测本身的收益：其中主要包含允许
tile DMA/计算重叠带来的收益。非流水完整Expert预取在24 MiB时相对弱baseline
分别为1.03208×（32 ops/cycle）和1.05486×（512 ops/cycle）。

OLMoE Top-8每层真实Expert总权重48 MiB，超过全部测试OCM容量；单个6 MiB
Expert可以完整加载，但不代表可以同时缓存全部真实或候选Expert。
Mixtral单Expert远大于OCM，只运行streaming；这轮固定浅prefix的额外收益很小，
不能据此否定其他tile深度、预测时序或allocator的可能收益。
错误预取会占用不可抢占DMA和OCM，因此当前策略没有“最坏自动等于不预取”的保证。
这些约束解释了负收益的可能来源，但本轮没有做逐项因果消融，不将其当作已量化归因。

## 验证与复现

全套测试：382 passed。额外产物检查通过：

- 配置点和样本键无重复，模型/策略范围正确。
- 所有样本peak OCM不超过配置容量。
- speculative bytes等于useful+wasted；无预测/按需case speculative bytes为0。
- Mixtral没有whole-residency case，whole-demand速度比分母保持空值。
- 所有记录的源码/配置SHA256与当前文件一致；两组PNG/PDF均已生成。

```bash
python examples/olmoe_mixtral_cases.py --workers 4
pytest -q -o addopts=''
```

输出目录：`results/olmoe_mixtral_cases/`。源码和实验配置不变时可使用
`--resume`继续同一输出目录的中断运行；若修改配置或模拟器，使用新的输出目录
完整重跑，不能复用旧缓存。本轮未修改模拟器核心、RTL或既有调度算法。
