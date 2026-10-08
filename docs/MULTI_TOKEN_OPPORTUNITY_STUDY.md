# Multi-token 初步机会验证

2026-10-08：正式矩阵 **COMPLETE，108个配对cell，1080次真实模拟**。
其中216次是独立单token参考模拟，864次是 grouped baseline/预测模拟。
全量回归 **599 passed，14条已有依赖弃用 warning**，含附件8-token的4/2/1/1权重守恒用例。

## 主要结果

下表均为两条固定seed trace的总cycle比值，而不是独立seed speedup的算术平均。
`rho=0.5`，每通道2byte/cycle；C/D为已测试GU深度与OFF的事后包络。

| Batch | 路由 | Reuse R | A/B-opt共享收益 | B-opt/C正确预测增量 | B-opt/D合成预测增量 |
|---:|---|---:|---:|---:|---:|
| 8 | dispersed | 1.00 | 1.00053× | 1.00943× | 1.00139× |
| 8 | balanced | 2.00 | 1.84581× | 1.00000× | 1.00000× |
| 8 | hot | 2.67 | 1.68412× | 1.09849× | 1.07088× |
| 16 | balanced | 4.00 | 1.92548× | 1.01105× | 1.00853× |
| 16 | hot | 4.00 | 1.82042× | 1.05582× | 1.04450× |
| 32 | hot | 8.00 | 1.96766× | 1.00162× | 1.00162× |
| 64 | balanced | 16.00 | 1.97494× | 1.00081× | 1.00081× |

结论：multi-token确实引入大幅权重流量减少与吞吐收益，但 **sharing收益不是prediction贡献**。
B=8均衡共享：独立316824cycle，Grouped171645cycle，正确预测仍171645cycle。
B=8热点：Grouped188124cycle，正确预测171257cycle，合成预测均值175672cycle。
热点有更大的剩余空间，不能仅以平均reuse或总必要bytes判断增量收益。
B=16 balanced/hot都有R=4，但Grouped耗时不同，正确预测增量也不同。

较高带宽的B=8热点点：rho=1时共享1.05976×、正确预测1.00325×；rho=2时
共享1.04770×、正确预测1.00646×。低带宽下的1.68×不能泛化到所有硬件。
B=64 balanced/hot计算payload利用率约99.6%，此时剩余空间重新收缩。

dispersed必要bytes不减少，部分rho=1/2的 grouped反而略慢于独立执行；
单Engine上改变Router阶段聚合、请求队列与串行Expert等待节奏，并不自动改善makespan。
这个结果不能归因为复用失败，因为这些trace本来就没有复用。

开启预取的原始点中，Oracle有46/216点比B-opt慢，合成预测有57/216点更慢。
它们完整保留于per_run.csv；图中的包络包含OFF，不是“预取不会变慢”的证明。
本轮所有配对包络C不慢于D，但这不是任意策略/容量下的普遍保证。
实测逐token预测命中率范围0.5–1.0，特别是小batch只有少数判断，不能据此称为
固定80%准确率的真实预测器实验。

初步判断：值得继续研究 **中等复用+路由偏斜+有限计算/访存重叠** 的区域，
而不是单纯增加batch。首先修复有限OCM下的活性问题，再对比共享baseline的DMA/
compute优化或pack拆分，确认1.098×空间是否仍然存在。不应先将其写成新贡献。

![共享收益](../results/multi_token_opportunity_20261008/sharing_gain.png)

![正确预测的增量机会](../results/multi_token_opportunity_20261008/prediction_headroom.png)

![合成预测的增量机会](../results/multi_token_opportunity_20261008/synthetic_prediction_gain.png)

图中不同带宽子图的纵轴范围不同；请按数值判断增量大小。

## 问题与范围

依据用户的分析，先分离两个问题：

1. 独立冷 token 执行改为按 Expert 共享权重，能获得多少收益？
2. 已具备 Grouped Tile Streaming 的强 baseline 上，预测还有多少额外收益？

本实验只使用现有主模型 `UnifiedSequenceSimulator`，不修改 RTL、核心模型或
调度算法。不评估拆分 pack、多 Engine 负载均衡、跨层 token 自由流动或真实神经预测器。
每个 token 代表独立请求的一个 decode step，不是同一请求未来的自回归 token。
每层全 batch 完成之后再进入下一层。这里只包含连续三个 MoE layer，不含 Attention。

## 固定参数（synthetic placeholder）

| 项目 | 设置 |
|---|---|
| 模型 | H=128，F=256，E=64，Top-1，INT8 SwiGLU |
| Expert 权重 | 3HF=98304 byte，即96KiB |
| 层数 | 3，layer barrier |
| 计算 | 单共享 Engine，32 operation/cycle，startup=8 cycle |
| 存储 | 四通道 striped，每通道单 in-flight，DMA startup=8 cycle |
| 每通道带宽 | 2/4/8 byte/cycle，对应单 token rho=0.5/1/2 |
| 时钟 | 200MHz，仅用于参数单位换算，不代表实现频率结论 |
| OCM | 固定2MiB，不随 batch/路由增加 |
| 非线性 | 64 element/cycle |
| Batch | 1、4、8、16、32、64 |
| 路由 | dispersed、balanced、hot |
| 激励种子 | 20261008、20261009 |

dispersed：每个 token 不同 Expert，U=B，不存在批内权重复用。
balanced：最多4个 Expert，token 数尽量均匀。
hot：ceil(0.75B) token 进入一个热点 Expert，其余在最多3个 Expert 间轮转。
每层独立重命名、打乱 token 顺序；种子生成的 truth/prediction 保存于 inputs.json。

## 对照与公平性

- A：Independent Cold Streaming。单 Engine 串行处理 B 个冷 token，每 token 均使用 tile
  streaming，不保留跨 token 权重缓存。几何完全一致，A=batch×单 token 实测序列耗时。
  单 token 参考值分别真实运行，不把乘出来的 batch 当作一次新的模拟。
- B：Grouped Streaming，现有 expert_locked/ready-first 逻辑、native ID demand 顺序；
  允许在16/64的 F-tile 中选最低 latency。
- B-opt-tested：两种 F-tile×两种 demand-ID顺序中的最短 latency。
  hot_first 通过逐层对 truth/prediction 同时重命名ID实现；striped placement 不依赖
  Expert ID，因此不改变物理通道。计算仍使用主模型，不是自由优化 compute order。
  该 baseline 是有限候选中的事后包络，不是全局最优调度。
- C：使用 B-opt-tested 相同 tile/order，输出逐 token 正确 Expert，测试 GU prefix depth
  1/4。上一层 Router 完成+20cycle 开始预测，无上一层全部权重就绪的硬屏障。
- D：与 C 相同，但每 token 有0.8概率预测正确，否则在其他63个 Expert 中随机选择。
  每 token 输出一个候选；按 sum 聚合为 batch Expert ranking，同一 tile不重复加载。
  0.8是生成概率，不保证每条 trace 的实测准确率正好0.8。

**C/D只预取 GU prefix，不是完整 Expert 预取，也不是零stall理想硬件。**
候选覆盖真实 Expert 的比例与逐 token 命中率分别记录，不能混用。
对照没有将8/16等多个 Expert 计算并行化；所有 Expert 的 n_e 个 token 一次打包计算。

报告的 C/D envelope 取 {depth1、depth4、OFF} 最小 latency。它只能说明已测试
配置的机会，不是可部署自适应算法，也不是严格 prediction upper bound。
原始 per_run.csv 保留所有开启策略，包括比 B-opt 慢的点，避免 OFF 掩盖负收益。

## 度量口径

\[
G_{sharing}=T_A/T_{B\text{-opt-tested}},\quad
G_{prediction}=T_{B\text{-opt-tested}}/T_C
\]

\[
R=BK/U,\quad W_A=LBK W_E,\quad W_B=\sum_l U_l W_E
\]

A/B 同时包含 packing 减少 GEMM startup 和 Router/control 的重复启动开销，
因此不能将全部 latency 收益都归因于字节减少。记录 required_compute_ops 检查
payload operation 不因 grouping 减少。字节减少因子独立于 latency speedup。

payload-only 下界为计算 payload 时间与必要权重传输时间之最大值；忽略 startup、
非线性依赖、Router barrier和控制，只提供非常乐观的同硬件速度上限。
compute_payload_utilization 含 Router 和 Expert。内存利用率采用最终资源退休时间
作为分母，防止错误 speculation 的 DMA 尾部越过推理 makespan 导致指标大于1。
tokens/cycle 是 batch throughput；cycles/batch_size 是摊销开销，不是某个请求的响应延迟。

## 发现的活性边界

初轮固定384KiB，在 dispersed B=32/64、F_tile=64、rho=0.5、seed=20261008时，
expert_locked 无法继续执行。其余 Expert 的首 GU/DOWN占用资源，当前 Expert 后续 tile
无法 admission。这是现有模型的调度/容量活性限制，不是物理配置不可实现。
同参数 F_tile=16 可执行，不能把整个 batch size 判为不支持。

本轮调整固定容量为2MiB以先隔离共享机会；没有修复或隐藏该限制。
小 OCM、更多 Expert、不同 tile 的资源保证及 pack splitting 仍需后续单独验证。
失败目录已明确标记，不能作为完成的性能 CSV 使用。

## 复现与产物

```bash
python examples/multi_token_opportunity.py --output results/multi_token_opportunity_reproduce
pytest -o addopts='' -q
```

默认产物位于 `results/multi_token_opportunity_20261008/`：
inputs.json、per_run.csv、comparison.csv、summary.json，以及共享、Oracle增量、合成预测
增量的三组 PNG/PDF。summary保存完整参数、核心代码/config哈希；运行期间 source改变
则禁止标记 COMPLETE。目录存在时拒绝覆盖。

本实验不对附件中 Expert Streaming 2026 的相关工作比较做独立文献核查，
也不据此声称新的论文贡献已经成立。
