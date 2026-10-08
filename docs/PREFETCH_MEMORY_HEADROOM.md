# Stream-only 内存利用率与预取空间：小规模方向实验

## 结论

2026-10-07 使用当前 `UnifiedSequenceSimulator` 完成 2,340 次模拟、156 个配置点，
输出 72 个准确率/利用率相图采样点。没有修改模拟器、Scheduler、RTL 或 XDC。

**“高内存利用率限制收益”得到支持；“利用率越低，收益必然越高”没有得到支持。**
这轮完美预测的收益集中于中间区域。内存几乎一直忙时没空间；内存极空闲时
baseline 本身也几乎没有可消除的等待。相同准确率下的调度时序还能产生非单调谷值。

## 实验定义

| 参数 | 值 |
|---|---|
| 模型 | 合成 INT8 SwiGLU，H=128，F=256，E=16，Top-1 |
| 序列 | 单 token，5 个 MoE 层；第一层冷启动，无跨 token cache |
| F groups | 4，F tile=64 |
| Engine | 单逻辑顺序 GEMM，32 ops/cycle，Router/GU/DOWN 共用 |
| 时钟 | 200 MHz |
| OCM | 固定 393,216 bytes，即 384 KiB、4 个 Expert |
| DMA | 4 通道、striped tile，每通道 1 in-flight，startup=8 cycles |
| DMA 仲裁 | demand 优先，已启动整矩阵 fragment 不可抢占；无 burst 仲裁 |
| GEMM / 非线性 | GEMM startup=8 cycles；非线性64 elements/cycle、startup=0 |
| 控制 | 默认研究规则，不加入 RTL ingress/FIFO/退休延迟 |
| 预测触发 | 前一层 Router done +20 cycles，一层 lookahead，无全权重屏障 |
| 唯一硬件扫描轴 | rho=0.2/0.5/0.75/0.9/1/1.25/1.5/2/3/4/8/16 |
| 速率换算 | 每通道 byte/cycle=4*rho，即0.8–64；不是板卡峰值 |
| 准确率 | 0/0.25/0.5/0.75/0.9/1，固定单候选 |
| 重复 | 3 个原 seed，每 seed 5 个原保存样本，共15序列/配置 |

激励复用 `results/prefetch_dense/inputs.json` 的 seed 20261005/20261006/20261007。
每 seed 的 20 个预测决策，分别命中0/5/10/15/18/20次；降低准确率时命中集合嵌套缩小。
命中输出真实 Expert，miss 输出原保存 wrong_ids[0]；confidence固定1，不做calibration。
固定**一个候选**隔离准确率，不是此前两候选Uniform实验；不期望其1.125×等于先前1.1107×。

三个固定策略使用完全相同的 streaming 计算：

- OFF：无预测、无预取；GU tile resident即允许计算，不等待完整 Expert。
- LIGHT：只请求候选 Expert 的 GU group 0，16 KiB；不预取DOWN。
- PREFETCH：请求候选 Expert 的全部GU+DOWN logical tiles，96 KiB总目标。
  这是**预取完整权重目标，计算仍streaming**，不是“全部驻留后再开始计算”的whole-ready模式。

perfect指accuracy=1、无错误候选；仍受因果时序/容量/DMA影响。
它不是零stall ideal，也不是最优调度延迟下界；perfect LIGHT也可能比OFF慢。

## 横轴：实测利用率，不用rho代替

主图横轴为平均 channel occupancy：

\[
U_{occupied}=\frac{\sum_c busy\_cycles_c}{N_{channels}T_{stream-only}}
\]

包含DMA setup；四通道同时工作100 cycles算400 channel-cycles，不是只算100。
分母包括全部5层推理时间、Router和第一层冷启动，不只取steady-state窗口。
baseline没有推理结束后的DMA tail，已在每次运行中断言，否则需裁剪interval。

同时保存更严格的payload带宽利用率：

\[
U_{payload}=\frac{W_{baseline}}{N_{channels}BW_{per-channel}T_{stream-only}}
\]

统计均采用合并样本的channel-cycles/合并时间，而不是各sample利用率的算术平均。
预测策略自己的占用指标用各自资源drain窗口，但主图/相图横轴始终来自配对OFF baseline。
平均利用率不告诉我们空档位于何处、是否在预测就绪之后、是否足以容纳不可抢占DMA。

## 完美预测曲线：实际结果

速度比为配对OFF总cycles / 预取总cycles；以下是测试点，不插值成通用边界。

| rho | baseline channel占用率 | payload利用率 | perfect LIGHT | perfect PREFETCH |
|---:|---:|---:|---:|---:|
| 16 | 6.98% | 5.98% | 1.00905× | 1.00905× |
| 8 | 12.83% | 11.84% | 1.01707× | 1.01707× |
| 4 | 24.20% | 23.23% | 1.03300× | 1.03300× |
| 3 | 31.59% | 30.58% | 1.04363× | 1.04363× |
| 2 | 45.66% | 44.73% | 1.06448× | 1.05490× |
| 1.5 | 59.16% | 58.19% | 1.06076× | 1.07312× |
| 1.25 | 69.45% | 68.50% | 1.01059× | 1.06082× |
| 1 | 84.12% | 83.25% | **0.94683×** | 1.04376× |
| 0.9 | 86.11% | 85.29% | **1.12504×** | 1.05194× |
| 0.75 | 89.22% | 88.49% | 1.09438× | 1.03924× |
| 0.5 | 94.95% | 94.46% | 1.04210× | 1.01666× |
| 0.2 | 97.91% | 97.71% | 1.01698× | 1.01698× |

图：`results/prefetch_memory_headroom_20261007/perfect_prefetch_vs_utilization.png/pdf`。
阴影为3 seed的min/max，不是置信区间；部分点三seed完全一致是同质负载/固定hit数的结果。

### 为什么高利用率端空间小

在不降低必需传输量W、不提高带宽B、不提前到计时窗口外传输的假设下：

\[
T_{prefetch}\ge W/B,\qquad S\le\frac{T_{baseline}}{W/B}=1/U_{payload}
\]

它是必要的带宽上限，不是可达到的性能保证，也不能把含setup的occupancy无条件代入。
97.71% payload利用率点，这个宽松上限约1.0234×，实测约1.0170×，符合空间受限的判断。
本轮全部perfect点额外传输bytes=0，因此差异不是错误流量造成。

### 为什么低利用率端也未出现大收益

本轮通过**提高带宽**降低利用率，而非增加可供预取的其他工作窗口。
传输本来就快时，消除少量memory stall对compute主导的总时间影响很小。
平均空闲带宽大，不代表baseline有大量可消除的关键路径等待。
其他降低利用率的方法（层间计算、通道放置等）尚未扫描，不能声称所有低U点都低收益。

### 为什么有局部负谷值

rho=1、perfect LIGHT：15序列总首启动等待减少61,920 cycles，但总执行增加31,080 cycles，
future-speculation blocking记录123,720 cycles。没有错误预测，也没有额外传输bytes。
因此“预测正确、首tile提前”仍不足以保证更早完成。
这些计数可重叠，不是严格互斥的时间分解；这轮没有逐请求反事实证明全部差值来源。
整fragment非抢占规则下的时序竞争是需进一步定位的因素，不应直接外推到burst-safe RTL。

## 准确率×利用率相图

图：`results/prefetch_memory_headroom_20261007/accuracy_utilization_phase.png/pdf`。
用散点保留非均匀实测横坐标，不用等宽heatmap冒充均匀utilization采样。

**分类不是先写好的期望表，也不是新增部署策略**，而是三个固定策略的事后有限比较：

1. LIGHT/PREFETCH只有当3个seed各自速度比都>=1.01才成为候选。
2. 若都不通过，标记OFF；不表示所有可能策略绝对无收益，只表示未过本轮门槛。
3. 合并总周期最快者为参考；差距在0.5%以内优先实际speculative bytes更少者，等字节选LIGHT。

代表实际采样点：

| prediction accuracy | U=6.98% | U=45.66% | U=59.16% | U=86.11% | U=97.91% |
|---:|---|---|---|---|---|
| 100% | OFF | LIGHT | PREFETCH | LIGHT | LIGHT |
| 90% | OFF | LIGHT | PREFETCH | LIGHT | OFF |
| 75% | OFF | LIGHT | LIGHT | LIGHT | OFF |
| 50% | OFF | LIGHT | LIGHT | LIGHT | OFF |
| 25% | OFF | LIGHT | OFF | LIGHT | OFF |
| 0% | OFF | OFF | OFF | OFF | OFF |

高利用率97.91%点要求perfect才过1%门槛；90%已OFF。
但84.12%点即使90%也OFF，86.11%点25%可LIGHT，说明average utilization不是充分分类器。
不能据此绘制“某个利用率阈值右侧一律OFF”的连续政策边界。

## 更准确的方向判断

需要同时满足三个条件：

1. 有剩余内存服务空间（高U是强限制）。
2. baseline仍有值得消除的memory critical-path等待（低U本身不保证）。
3. 因果可用空档、OCM准入和DMA调度允许预取不伤害当前层。

因此可以把utilization作为**必要资源信号之一**，但不能单独用它预测收益或开关预取。
下一步可在相同实测U附近改变窗口位置、burst粒度及预算，检验差异是否消失。
本轮只改变带宽，尚未证明跨模型、跨Top-K、跨内存后端的普适关系。

## 工程与验证

新增：

- 配置：`configs/experiments/prefetch_memory_headroom.yaml`。
- 驱动：`examples/prefetch_memory_headroom.py`。
- 单元测试：`tests/test_prefetch_memory_headroom.py`，6个测试。

输出目录 `results/prefetch_memory_headroom_20261007/`：

- `inputs.json` / `summary.json`：保存trace、完整配置、来源SHA-256和定义。
- `per_sample.csv`：2,340行，逐次周期/流量/利用率/阻塞。
- `comparison.csv`：156行，配对汇总与seed范围。
- `perfect_prefetch.csv`：24行，两条固定perfect曲线。
- `phase.csv`：72行，真实分类和各策略speedup。
- 两组PNG/PDF图。

检查包括当前源码与激励hash、唯一性、baseline无尾部DMA、利用率不超1、
OCM上限、speculative=useful+wasted、perfect wasted=0和匹配样本分母。
全量回归：**584 passed, 14 warnings**；警告为既有依赖弃用提示。

```bash
PYTHONPATH=src python examples/prefetch_memory_headroom.py \
  --workers 8 --output results/prefetch_memory_headroom_new
pytest -q -o addopts=''
```

拒绝覆盖已有输出；实验worker数不是硬件并行度。没有调用Vivado或声称完成RTL验证。
