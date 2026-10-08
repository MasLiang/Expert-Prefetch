# Switch 资源驱动预取：取消上一层全权重 barrier

## 改动

主 UnifiedSequenceSimulator 原本就允许 ready prediction 立即尝试 OCM admission，
本次关闭 Switch 实验中的 `wait_for_previous_layer_weights`，去掉旧真实模型 driver
里强制开启 barrier 的硬编码。保守模式仍可通过配置显式开启，用于历史对照。
旧 pilot 配置显式保留 true，避免把其历史实验参数默默改写。

预测就绪后即可请求未来权重，并不保证立即开始 DMA：demand 优先、非抢占 DMA、
OCM 容量、真实 Router 依赖和计算依赖均保留。没有把 Router 推迟到预取完成，
没有使用事后最优选择，也没有修改 Expert 调度算法。

## 复现与产物

```bash
python examples/switch_prefetch_comparison.py --reference results/switch_prefetch
pytest -q
```

默认输出 `results/switch_prefetch_flexible/`，不覆盖旧结果。
此处记录该轮历史输出；添加弱 baseline 后，当前脚本默认输出改为
`results/switch_prefetch_baselines/`，见 `SWITCH_BASELINE_COMPARISON.md`。
共 2400 次模拟、48 个聚合点，与历史实验共享 84.4% 双候选覆盖率、排名、
truth、硬件和 tile 配置，仅关闭 barrier。

- `per_sample.csv`：2400 行，逐单 token 六层结果。
- `comparison.csv`：48 行，各策略相对无预取/完整目标的 speedup。
- `barrier_comparison.csv`：48 行，与旧 barrier 点逐项比较。
- `per_prediction.csv`：9000 行，预测 ready、请求释放、Router 和预取完成量。
- 双模型输入 JSON、summary（配置和来源 hash）、两组 PNG/PDF。

所有 12 个无预取配置点的总 cycles 与历史结果完全一致。
365 个测试通过；新增测试证明未来 DMA 能在上一层权重未全部取完时启动，
并保持 truth、OCM 容量和确定性。未修改 RTL。

## 窗口变化

Switch-base-8，16 MiB，第 0 个样本第一条完整目标预测：

| 配置 | 旧释放 cycle | 新释放 cycle | Router 完成 cycle | 旧窗口 | 新窗口 |
|---|---:|---:|---:|---:|---:|
| 32 ops/cycle | 286656 | 412 | 297816 | 11160 | 297404 |
| 512 ops/cycle | 74144 | 52 | 74568 | 424 | 74516 |

新释放时刻等于 predictor ready；各 DMA 的启动仍由通道资源决定。

## 性能结果

以下 speedup 均为无预测总 cycles / 当前策略总 cycles，聚合 50 个样本。

| 模型 | ops/cycle | OCM | 完整 Expert | Uniform tile | Ranked tile |
|---|---:|---:|---:|---:|---:|
| Switch-base-8 | 32 | 16 MiB | 1.003665× | 1.003665× | 1.003665× |
| Switch-base-8 | 512 | 16 MiB | 0.994824× | 0.968870× | 0.985252× |
| Switch-base-16 | 32 | 16 MiB | 1.003660× | 1.003660× | 1.003660× |
| Switch-base-16 | 512 | 16 MiB | 0.995091× | 0.968961× | 0.985432× |

16 MiB、32 ops/cycle 时，完整目标相对旧 barrier 快约 2.05–2.08%，
两个候选的全部权重均在各目标 Router 前传输完成：500 个候选 Expert，
其中包含错误候选；不是 500 个正确 Expert，也不是同时驻留的证明。
同配置 tile 策略周期未变，说明原来的短窗口已足以覆盖这一小段 prefix 的有效收益。

512 ops/cycle 配置下，取消 barrier 没有使完整目标变快，两个 tile 策略反而变慢。
窗口变大不意味着资源竞争必然变小，也不能保证 speculative scheduler 最坏回退到
无预取。没有用取 min 的方式掩盖负收益。

## 仍需注意

本轮仍保留原 DMA 粒度：`burst_arbitration=false`，每次一个 96 KiB 矩阵 tile
不可抢占。`burst_bytes=16 KiB` 不代表启用了 16 KiB 边界仲裁。
该选择用于隔离 barrier 的影响，不是最终硬件服务模型。
后续可单独对 burst 仲裁、预取预算和启动 tile 优先级做受控实验，不能将取消
barrier 与这些优化同时混在一起后归因给 barrier。

容量 8/16/24 MiB 是 Expert OCM 预算；200 MHz、配置算力/带宽和 ReLU 吞吐均
不是板级测量。模型未剪枝，CEPR-inspired synthetic predictor 不等于真实 CEPR。
没有 encoder/attention/KV cache 和跨 token 持久 cache；本报告不声称完整 LLM 性能。
