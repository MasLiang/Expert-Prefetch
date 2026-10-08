# 合成配置下的预测预取收益区域

## 目的与结论边界

本轮不限定现实模型，使用主 `UnifiedSequenceSimulator` 扫描计算、带宽、
Expert尺寸、容量、预取深度和预测质量，寻找相对**无预测tile streaming**的净收益。
未修改模拟器核心、RTL或调度算法。所有硬件/模型均为合成参数。

首轮1188个配置点，补充594个，共1782个点、8910条样本结果。
其中1458个点为预测Uniform，162个无预测、162个true-only Oracle。
预测点中195个达到1.01×、87个达到1.05×、49个达到1.10×。
这是配置行计数，不是独立硬件数或随机抽样概率；请求深度超过Expert的点会
截断，存在等效执行，不能把这些计数用于统计显著性声明。

**主要发现：不是一个简单的compute-bound阈值。** 比例、预取覆盖量、
实际空闲带宽、OCM admission和不可抢占阻塞共同决定效果。同一比例也会因为
启动开销和非线性时间不同而产生不同结果；不声称一个参数即可保证收益。

## 配置

- H=128，F tile=64；Expert含1/2/4/16/64个F groups，SwiGLU、INT8。
- E=16、Top-1；每点5个独立单token序列，每序列5个连续MoE层；第一层冷启动。
- PE=1/4/16，每PE为32 ops/cycle；全部PE分配当前GEMM，Expert仍顺序执行。
- 4内存通道；每通道扫描1/2/4/8/16/32/64 bytes/cycle，非完整笛卡尔积，具体见两份YAML。
- OCM为0.25/1/4个Expert等价容量；小Expert补充矩阵仅扫描1/4，避免连当前tile工作集都装不下。
- 每次预测两个候选，质量0.5/0.9/1.0；Uniform请求GU深度1/4/16，超过Expert深度时截断。
- 200 MHz；DMA setup=8 cycles、计算startup=8 cycles、非线性64 elements/cycle。
- 前一层Router完成后20 cycles得到预测；不等待前一层全部权重加载。
- actual-byte OCM；整矩阵tile DMA不可抢占；不启用小burst仲裁。

`total_aie_units`是既有API字段，本轮映射为抽象PE，不指真实AIE或U200资源。
主机worker只并行运行独立实验，不增加模拟硬件并行度。

## 激励与质量语义

所有配置共享保存的truth。质量降低时，命中决策集合嵌套缩小；命中时真Expert
位于rank-1，miss时两个候选都错误。因此本轮质量同时改变candidate recall与
rank-1精度，不是保持排名不变的confidence calibration实验。
每个质量有20个预测决策，分别10/18/20次命中；score为固定排名值，非校准概率。
Uniform仍对两个候选加载，因此质量1也不意味着没有错误候选流量。

Oracle仅输出真实Expert、请求所有GU tiles，使用同一因果触发时间和容量限制。
它是诊断参考，不是最优调度，也不保证是任何策略的严格延迟下界。

## 归一化坐标

令C为所有PE总ops/cycle，B为所有通道总bytes/cycle。对于单token INT8 SwiGLU：

\[
\rho=\frac{t_{compute,payload}}{t_{memory,payload}}
=\frac{6HF/C}{3HF/B}=\frac{2B}{C}
\]

rho<1表示理想payload估计偏memory-bound，rho>1表示偏compute-bound。
这不是实际利用率：它忽略startup、Router、非线性、通道放置和有限OCM。
与之共同观察的指标包括OCM/Expert bytes、预取GU占全部权重比例
`2*effective_depth/(3*groups)`、实际useful/wasted bytes、首计算启动等待和
关键路径future-speculation阻塞。

## 固定策略曲线

统一固定PE=1、quality=0.9、OCM=4个Expert、GU depth=1，只改变带宽和Expert大小。
以下是真实采样点，不是拟合预测或理论上界：

| F groups / Expert | rho=0.5 | rho=2 | rho=8 |
|---|---:|---:|---:|
| 1 | 1.10672× | 1.20408× | 1.05670× |
| 2 | 1.05580× | 1.11052× | 1.02985× |
| 4 | 1.02855× | 1.05480× | 1.01533× |

可以观察到一个收益较高的中间区域，而非带宽越大收益越大。
memory-bound端竞争可能压过收益；带宽很充裕时baseline等待已经较少。
固定浅prefix在大Expert中只覆盖很小比例，收益随groups增加明显下降。
1 group是刻意加入的退化小Expert：一个GU tile已覆盖全部GU，不代表Mixtral。

不能把rho=2写成通用充分条件，也不能把表中采样点之间视为已验证区间。
尤其4 groups的rho=1处为0.84810×，存在明显非单调谷值。
在该点，baseline总cycles=184500，预测为217544；首启动等待之和反而从25800
降至7224，future-speculation关键路径阻塞为61720，OCM无未来预取驱逐。
这再次说明“提前启动”不等于“提前完成”；错误DMA与真实DMA的时序会改变关键路径。
阻塞计数不能直接从总时间扣除作加法分解，因为部分阻塞与其他工作重叠。

同点true-only all-GU Oracle为1.04376×，说明至少存在可利用的调度空间，
但这个对照同时改变候选集合和深度，不能当成仅改变准确率的严格消融。

## PE与容量

合成2 groups、quality=0.9、GU depth=1、OCM=4：

- PE=1、每通道8 bytes/cycle：1.11052×。
- PE=4、每通道32 bytes/cycle：1.11126×。

两点rho均为2，payload比例一致，性能接近；并非所有等rho点都完全相同。
在第一个点把OCM从4个Expert改成1个Expert时，实际speculative bytes降为0，
speedup=1：名义预测窗口仍在，但当前层工作集/admission不允许预取实际进入。

因此可用于后续设计的**候选条件**是：

1. 当前层存在可利用的传输窗口，而不是只存在很长的时间窗口。
2. OCM在当前工作集之外允许候选tiles真正进入并保留至消费。
3. 有用prefetch覆盖baseline关键路径等待的足够大部分。
4. 错误DMA阻塞、额外流量和容量竞争小于消除的等待。

严格的净收益判据始终是`baseline makespan - predictive makespan > 0`。
达到1.05×需要净消除baseline时间约4.76%，达到1.10×约9.09%。
这些是时间账本条件，不是可单凭rho推导出的调度保证。

## 图与产物

- `results/prefetch_phase/`和`results/prefetch_phase_refinement/`：输入JSON、
  per_sample/comparison/thresholds CSV、完整配置与源码SHA256、固定深度曲线和相图PNG/PDF。
- `results/prefetch_phase_analysis/benefit_curve.png/pdf`：上述固定depth=1的合并曲线。
- `fixed_policy_curve.csv`：曲线原始点；`summary.json`：完整行数和阈值计数。

相图明确标记为测试深度中的事后最优，不能用它的包络声称一个固定策略的部署收益。
折线仅连接已测点辅助阅读，不证明中间配置的行为。

## 验证与复现

全套392 tests通过。检查了两轮完成状态、所有记录源码SHA256、无重复配置、
OCM约束、speculative=useful+wasted、无预测流量为0、Oracle wasted为0。
主模拟器语义保持不变。

```bash
python examples/prefetch_phase_sweep.py --workers 4
python examples/prefetch_phase_sweep.py \
  --config configs/experiments/prefetch_phase_refinement.yaml \
  --output results/prefetch_phase_refinement --workers 4
python examples/analyze_prefetch_phase.py \
  --inputs results/prefetch_phase results/prefetch_phase_refinement \
  --output results/prefetch_phase_analysis
pytest -q -o addopts=''
```

这些是固定seed、5样本/点的机制探索，不是统计稳定性或现实硬件验证。
尚未扫描Top-K/Expert并行、多层lookahead、不同tile尺寸、burst仲裁及不同seed。
后续若要给出论文中的普适边界，需要对收益转折附近加密扫描、增加seed，
并做保持其他变量不变的单因素消融。
