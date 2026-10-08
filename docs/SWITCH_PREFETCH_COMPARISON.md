# 未剪枝 Switch 预取对照实验

本报告记录历史 `wait_for_previous_layer_weights=true` 结果。
当前默认实验已关闭该 barrier，输出到 `results/switch_prefetch_flexible/`，
新旧差异见 `SWITCH_FLEXIBLE_PREFETCH.md`；历史结果目录未被覆盖。

## 范围与复现

历史实验参数保存在 `results/switch_prefetch/summary.json` 的 `settings` 中；
当前实验运行 `python examples/switch_prefetch_comparison.py`，测试 `pytest -q`。
结果位于 `results/switch_prefetch/`：`comparison.csv`（48 个聚合点）、
`per_sample.csv`（2400 次模拟）、两份共享激励 JSON、`summary.json` 及两组 PNG/PDF。
没有修改主模拟器或调度算法。Switch 工作量采用已接入的 `relu_ffn`。

## 工作负载与预测假设

- Switch-base-8/16：H=768、F=3072、Top-1，decoder 六个 MoE 层。
- 两矩阵 ReLU，未剪枝，无 bias；INT8 权重/激活是部署假设，单 Expert 4.5 MiB。
- 每模型 50 个独立、冷启动的单 token 六层序列，不是跨 token 保持 cache 的完整生成。
- 五个后续 MoE 层有预测，共 250 个目标。211 命中，双候选覆盖率恰好 84.4%。
- 命中候选排名 1/2 分别有 106/105 个；这是合成假设，不是 CEPR 实测排名分布。
- 仅参考 FLAME 的覆盖率，不实现 CEPR。预测用真实路由构造，是受控合成 trace。
- 预测由上一 MoE 层 Router 完成触发，latency=20 cycles；实际预取等待上一层权重
  加载完成。提前一个 MoE 层不是提前一个所有类型的 Transformer block。
- 所有策略共用同一批 truth、candidate ranking、score 和硬件配置。
- 不包含 encoder、attention、KV cache、其他层耗时或 N:M pruning。

## 硬件与策略

配置文件为 `configs/experiments/switch_prefetch.yaml`。硬件是 U200-like 可配置抽象，
不是板级实测或 FLAME 复现：200 MHz，4 通道，每通道配置带宽 3.2 GB/s，效率 1，
DMA setup=8 cycles，burst=16 KiB，每通道一个 in-flight；单逻辑计算引擎，
32/512 configured ops/cycle，task startup=8 cycles。历史字段名 AIE 在这里表示抽象
PL engine，U200 没有 AIE。ReLU 吞吐配置为 16 elements/cycle，非实测值。

Expert OCM 预算为 8/16/24 MiB，按实际 byte 计容量，不使用 RTL 固定 slot，
且不声称所有物理片上 SRAM 都分给 Expert。F tile=128，共 24 个 tile groups，
每张矩阵的一个 tile 为 96 KiB。

| 策略 | 预取目标 |
|---|---|
| no_prediction | demand tile streaming，可与计算重叠；不是串行加载弱 baseline |
| full_expert_prefetch | 两个候选的全部 Up/Down tiles |
| uniform_prefix | 两个候选各 2 个 Up tiles |
| ranked_prefix | 第一候选 2 个、第二候选 1 个 Up tiles |

完整目标仍受 OCM、DMA 和时间窗口限制，不是 whole-Expert 原子 admission。
8 MiB 本身小于两个完整候选所需的 9 MiB；16/24 MiB 容量上允许两候选权重，
但不保证它们按时加载或不干扰 demand。CSV 的
`full_experts_transferred_before_router` 统计完整 Up/Down tile 集合的传输完成，
不是同时驻留证明，也包含错误候选。

## 结果

speedup 是 **无预测总 cycles / 当前策略总 cycles**。小于 1 表示变慢。
聚合使用 50 组 cycles 求和再取比值，不是简单平均每样本 speedup。

| 模型 | ops/cycle | OCM | 完整 Expert | Uniform tile | Ranked tile |
|---|---:|---:|---:|---:|---:|
| Switch-base-8 | 32 | 16 MiB | 0.983170× | 1.003665× | 1.003665× |
| Switch-base-8 | 512 | 16 MiB | 0.994824× | 0.991177× | 0.992989× |
| Switch-base-16 | 32 | 16 MiB | 0.983528× | 1.003660× | 1.003660× |
| Switch-base-16 | 512 | 16 MiB | 0.995091× | 0.991443× | 0.993255× |

全部矩阵范围：完整 Expert 0.934511–0.995091×；Uniform tile
0.991177–1.003665×；Ranked tile 0.992989–1.003665×。

16 MiB、32 ops/cycle 时，tile 策略相对完整目标约快 2.05–2.08%。
Switch-base-8 的 AIE starvation 总 cycles 从无预测 463200 降至 tile 的 137416，
但计算本身占比很高，所以总延迟只下降约 0.365%。
512 ops/cycle 时，有限的实际预取窗口和竞争使这些策略没有正收益；
完整候选在目标 Router 前完成的计数为零（16 MiB 配置）。

这些结果表明本配置下高 candidate coverage 不足以保证速度收益。
错误候选流量、容量与时间窗口、DMA 排序和既有 demand overlap 都会影响结果。
这里只支持“本轮配置下”的结论，不能推断所有 Switch/CEPR 场景都无收益，
也没有把完整 Expert 的劣势直接归因于某一个未经隔离的因素。

## 验证与边界

364 个测试通过，新增测试检查精确覆盖率、候选唯一性、命中排名、
四种策略确定性、容量约束和无预测的 byte conservation。
两模型均观察到 0.844 覆盖率，全部点完成六层且 OCM 峰值不超预算。
后续若扫描预测提前时间、带宽或改为跨 token 复用，应单独保存配置与结果，
不将本轮合成结论描述为 FLAME 的实测结果。
