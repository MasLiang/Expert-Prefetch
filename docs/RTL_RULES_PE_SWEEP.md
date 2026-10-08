# RTL架构规则对照与单Engine MAC规模扫描

## 本轮范围

2026-10-06：先记录DSP MAC扩展方案，再使用主模拟器
`UnifiedSequenceSimulator`已有的`ExecutionRules`进行配置对照和规模扫描。
**没有修改RTL，没有修改主模型默认行为，没有运行综合或RTL仿真。**
DSP映射、OCM宽读口与参数化归约流水尚待实施，见
[待实施方案](CONFIGURABLE_DSP_MAC_PLAN.md)。本轮扩大的是单Engine的假想有效计算速率，
不是增加同时执行的Expert数，也不是声称当前16-lane RTL已支持256 lanes。

## 对照规则

| 模式 | 本轮启用的规则 |
|---|---|
| `legacy` | 实际byte容量、原有Expert锁定、整tile DMA、灵活跨层放行 |
| `tile_scheduler` | GU大小固定slot；DOWN也占一个slot；4 banks；每bank保护2个demand slots；GU/DOWN原子pair admission；pair FIFO；未来pair重试；tile-ready-first；当前RTL上一层权重完成条件 |
| `burst_fifo` | 上述规则，加4096-byte burst边界仲裁、64个descriptor、深度16的DMA FIFO |
| `rtl_resources` | 上述规则，加配置中的控制/ingress/descriptor退休、payload可见性、请求与返回延迟；层间等待实际退休资源事件；区分FIFO reserved credit与actual occupancy |

GU一旦resident、DOWN slot已保留，即可开始GU计算；不要求DOWN payload已返回。
同一tile的GU/非线性/DOWN保持依赖，tile之间可切换Expert；计算资源仍只有一个。

不设置“仅slot/pair、但仍锁定整个Expert”的中间消融：在Top-8、部分预取和容量饱和时，
非Router游标Expert可能持有计算所有权，Router游标的pair又占满slot，出现互相等待。
该混搭不是当前RTL策略。本轮最初尝试暴露了该问题，随后去掉无效混搭，
增加Top-8部分驻留前进性回归；没有通过跳过失败样本获得最终结果。
因此第一步是联合规则变化，不能解释为隔离slot这一单因素的因果实验。

当前`cross_layer_prefetch.sv`仍检查上一层`weights_done`，所以非legacy模式保留此条件。
先前研究版本的灵活放行不等于目前RTL已放宽；此差异在本轮显式声明，未修改RTL。

## 输入及硬件假设

| 项目 | 合成SwiGLU | OLMoE结构 |
|---|---:|---:|
| H / F | 128 / 512 | 2048 / 1024 |
| E / K | 16 / 2 | 64 / 8 |
| F tile宽度 | 64 | 128 |
| GU slot bytes | 16,384 | 524,288 |
| OCM bytes（4 banks×12 slots） | 786,432（0.75 MiB） | 25,165,824（24 MiB） |
| 单seed输入序列数 | 5 | 1 |
| 实际candidate recall | 27/30=90% | 22/24≈91.67% |

每序列单token、4个MoE层；3个固定seed；权重/激活8-bit；候选数量2K；
随机candidate ranking；Uniform prefix depth=2。所有模式和MAC档位复用保存的相同输入。
OLMoE采用真实结构，不是实际模型路由数据；91.67%来自整数hit数取整，不能称为精确90%。

公共配置：200 MHz；4 memory channels；每channel最多一个in-flight；DMA startup=8 cycles；
GEMM startup=8 cycles；非线性16 elements/cycle。配置值不是板级实测。
合成模型扫描每channel 4/16/64 bytes/cycle；OLMoE只扫描16 bytes/cycle。
高档带宽是供数敏感性假设，不代表U200真实DDR可提供该速率。

`rtl_resources`的`dma_payload_cycle_multiplier=2`显式模拟当前payload路径额外周期，
因此仅payload窗口的有效速率是标称的一半；burst startup、选择及返回进一步降低速率。
例如标称16 bytes/cycle/channel并不是此模式下16 bytes/cycle/channel的有效payload速率。
legacy不加此倍率，因此跨模式差值混合了硬件规则和供数成本，不能全部归因于scheduler。

MAC groups=1/2/4/8/16对应16/32/64/128/256假想lanes；
有效计算速率=32×groups operations/cycle，MAC按乘+加两次operation计数。
`total_aie_units=1`、`aie_units_per_expert=1`，名称中的AIE是主模型既有资源接口，
在这里代表PL逻辑GEMM Engine，不是声称U200具有AIE。

Router控制周期采用结构估计`E*K + 2*K + 34*(K-1) + 3`；allocator周期采用
`candidate_count + 5 + ceil(log2(candidate_count))`。其余控制参数全部保存于YAML。
这些不是针对大模型实测校准后的微架构延迟，不应用于绝对板级吞吐结论。

## 结果：扩大MAC是否有用

以下均为`rtl_resources`，cycle是3个seed的平均每序列makespan。
prefetch speedup是同档位不预测baseline与预测版本的配对速度比的seed均值。
这与“更大MAC相对16 lanes的加速”是不同分母。

| 负载 / 标称每channel带宽 | lanes | 不预取cycles | Uniform预取cycles | 预取speedup |
|---|---:|---:|---:|---:|
| 合成 / 4 B/cycle | 16 | 202,441 | 227,560.7 | 0.88984× |
| 合成 / 4 B/cycle | 64 | 200,521 | 208,653.1 | 0.96103× |
| 合成 / 4 B/cycle | 256 | 200,041 | 202,475.6 | 0.98798× |
| 合成 / 16 B/cycle | 16 | 103,245 | 101,453.5 | 1.01766× |
| 合成 / 16 B/cycle | 64 | 53,065 | 59,958.3 | 0.88519× |
| 合成 / 16 B/cycle | 256 | 52,585 | 54,453.1 | 0.96572× |
| 合成 / 64 B/cycle | 16 | 101,709 | 101,069.5 | 1.00633× |
| 合成 / 64 B/cycle | 64 | 27,597 | 26,957.5 | 1.02372× |
| 合成 / 64 B/cycle | 256 | 15,721 | 17,434.6 | 0.90200× |
| OLMoE / 16 B/cycle | 16 | 12,693,741 | 12,643,253 | 1.00399× |
| OLMoE / 16 B/cycle | 64 | 6,472,617 | 6,767,903.7 | 0.95649× |
| OLMoE / 16 B/cycle | 256 | 6,454,185 | 6,503,768 | 0.99239× |

OLMoE不预取：16→64 lanes加速1.9611×；64→256 lanes仅再快约0.29%。
payload计算利用率（Router+Expert，排除startup）为99.39%→48.73%→12.22%。
Uniform预取自身16→64 lanes仍加速1.8684×，但64 lanes下慢于同档位不预取baseline。
因此“MAC扩宽有益”和“预测有净收益”必须分开判断。

合成模型标称4 B/cycle/channel下，16→256 lanes不预取仅加速1.0120×；
供数不足时扩宽几乎无用。标称64 B/cycle/channel下，16→64 lanes不预取加速3.6855×，
并出现1.02372×预取收益；继续到128 lanes预取跌至0.80224×，256 lanes为0.90200×。
速度比不是随MAC规模单调增加的，不能以更多DSP或更高predictor recall保证收益。

这是固定Uniform策略、固定OCM及控制参数下的有限扫描，不是最优allocator搜索。
不推断Universal最佳lane数，也不从三seed小样本推出统计置信区间。

## 规则对照

16 lanes、标称16 B/cycle/channel时：

| 负载 | legacy预取speedup | slot/pair+tile调度 | 再加burst/FIFO | 全部资源规则 |
|---|---:|---:|---:|---:|
| 合成 | 1.00789× | 1.00789× | 1.00789× | 1.01766× |
| OLMoE | 1.01359× | 1.00195× | 1.00195× | 1.00399× |

该特定档位burst/FIFO未改变makespan，不表示它们在其他参数下无影响。
全部资源规则会同时改变baseline和预测版本，不能只比较speedup来判断绝对吞吐。
逐点cycle、bytes、等待及峰值见`comparison.csv`和`rule_ablation.csv`。

## 验证与产物

最终完整运行：80个配置点、240个seed点、1008条逐样本结果。全部完成，无失败样本跳过。
完整pytest：**421 passed**。
另外检查产物唯一性、OCM容量、speculative=useful+wasted、无预测speculative=0、
单Engine及lane映射、descriptor≤64、FIFO预留及实际occupancy≤16、来源及RTL SHA256一致。
FIFO两个峰值本次均可达到16；相同峰值不代表它们在每个cycle都相同。
`memory_wait_sum_cycles`是Expert等待汇总，可能重叠，不能作为makespan互斥时间分区。

复现：

```bash
python examples/rtl_rules_pe_sweep.py --workers 4
pytest -q -o addopts=''
```

结果目录`results/rtl_rules_pe_sweep/`：

- `summary.json`：范围、限制、配置、Python/RTL来源哈希。
- `*_inputs.json`：每seed完整truth/candidate输入。
- `rule_declarations.json`：四组实际执行规则。
- `per_sample.csv` / `per_seed.csv` / `comparison.csv`：原始及配对汇总。
- `rule_ablation.csv`：16 lanes档位规则对照。
- `mac_scale.png` / `mac_scale.pdf`：总cycle、计算利用率和预取收益曲线。

## 对齐边界及下一步

本轮是**架构规则对照**，不是完整RTL联合仿真：Uniform allocator与当前RTL
HardwareGreedy-v1不同，宽MAC供数/归约/排空及大模型控制代价尚未验证。
slot容量按GU实际大小参数化，不是原RTL默认slot配置的直接运行结果。
DSP真实映射、OCM宽读口、DDR板级带宽、完整模型数值正确性均未在本轮新增验收。

先保持RTL不动。将来实施可配置DSP阵列时，应先选择16/32/64 lanes并验证OCM供数；
不能从本轮假想256-lane结果直接宣布可部署。预取收益不应作为扩宽MAC唯一目标，
应同时关注总延迟、供数代价和DSP/BRAM实际资源。
