# Switch 非流水完整 Expert 预取严格对照

## 定义

双方均使用主 `UnifiedSequenceSimulator` 的 `demand_loading=full_expert_load`：
当前真实 Expert 的所有 Up/Down tile 权重全部 resident，才允许其第一次 GEMM；
开始计算后不再 DMA 本 Expert 的权重，不做本 Expert 内部权重加载/计算重叠。
每个 Expert 的计算仍采用相同 F-axis groups、startup 和 ReLU 时长。

- `demand_full_expert`：无 prediction，Router 后按需加载整个当前 Expert。
- `nonpipeline_full_expert_prefetch`：相同执行规则，增加跨层完整候选目标预取。

“非流水”不禁止未来层权重 DMA 与当前层计算重叠，否则不能研究跨层预取。
它也不表示去掉 F tiling 或替换 GEMM 性能模型。完整目标不是 all-or-nothing
Expert admission，也不保证全部候选在 Router 前加载完成；容量不足时仍可能只
完成部分预取，Router 后补齐真实 Expert，计算门槛始终是本 Expert 全权重 ready。

## 配置与复现

```bash
python examples/switch_nonpipeline_prefetch.py
pytest -q
```

配置沿用 `configs/experiments/switch_prefetch.yaml`：Switch-base-8/16，H=768、
F=3072、Top-1，未剪枝两矩阵 ReLU，INT8 部署假设，六个 decoder MoE 层。
每模型 50 个独立冷启动 token 序列，第一层没有预测，后五层 250 个目标中
211 个命中双候选，覆盖率 84.4%；命中排名人为均分，不实现真实 CEPR。

上一 MoE 层 Router 完成后 20 cycles 预测 ready，不等待上一层全部权重完成。
OCM 按实际 byte 分配，预算 8/16/24 MiB，单逻辑计算引擎 32/512 ops/cycle；
200 MHz，4 通道各配置 3.2 GB/s，DMA setup=8 cycles，非线性吞吐16 elements/cycle。
这些均是配置假设，不是 U200 实测或 FLAME 复现。保持本轮不启用 burst 仲裁：
一个矩阵 tile 请求 96 KiB 不可抢占；不改其他调度算法。
没有 encoder、attention、KV cache 或跨 token cache 复用。

## 结果

速度比为 50 个样本的非流水、不预取总 cycles / 非流水、预取总 cycles。
下表中的延迟降低百分比与 speedup 不同，均直接由总 cycles 计算。

| 模型 | OCM MiB | ops/cycle | Speedup | 延迟降低 |
|---|---:|---:|---:|---:|
| Switch-base-8 | 8 | 32 | 1.057688× | 5.454% |
| Switch-base-8 | 16 | 32 | 1.163985× | 14.088% |
| Switch-base-8 | 24 | 32 | 1.163985× | 14.088% |
| Switch-base-8 | 8/16/24 | 512 | 1.071699× | 6.690% |
| Switch-base-16 | 8 | 32 | 1.057625× | 5.449% |
| Switch-base-16 | 16 | 32 | 1.163787× | 14.074% |
| Switch-base-16 | 24 | 32 | 1.163787× | 14.074% |
| Switch-base-16 | 8/16/24 | 512 | 1.071926× | 6.710% |

Switch-base-8，16 MiB：

| ops/cycle | 非流水、不预取 cycles | 非流水、完整目标预取 cycles |
|---|---:|---:|
| 32 | 110997600 | 95359968 |
| 512 | 27945600 | 26075968 |

32 ops/cycle、16/24 MiB 时，每模型 500 个候选 Expert 全权重传输在目标 Router
前完成（包括错误候选；不是 500 个真实命中，也不证明任意时刻同时驻留）。
其中 211 个真实命中免去了对应的完整 demand 加载等待。
Switch-base-8 的减少 cycles 正好为 `211 * 74112 = 15637632`，
74112 是该配置一个完整 Expert 的 DMA payload 加 setup 时间。

8 MiB 时空间不足以放下两个 4.5 MiB 候选，同时还要容纳当前层权重，收益下降。
512 ops/cycle 时所有容量点完整候选在 Router 前完成的计数为零；
约 1.072× 来自提前完成部分传输及随后补齐，不可声称“完整 Expert 在 Router
前已经 ready”。更快计算减少了可用预取窗口。

## 与前几轮的区别

前几轮的 1.2485× 使用了流水预取作为分子对应方案，不能作为此严格 case 的答案。
本轮非流水完整目标预取的收益约 1.164×（16 MiB、32 ops/cycle），
说明完整 readiness 的 baseline 有更多可隐藏的权重等待。
但这不是 FLAME 的 1.48× 精确复现；硬件服务比例、预测排名/时间与运行方式不同。

## 产物与验证

`results/switch_nonpipeline_prefetch/`：

- `comparison.csv`：24 个配置点（12 对）。
- `per_sample.csv`：1200 次模拟（600 对样本）。
- `per_prediction.csv`：3000 个预测目标时序记录。
- 双模型输入 JSON、summary/config/source hashes、speedup PNG/PDF。

600 个非预测 baseline 样本逐项与上一轮的 `demand_full_expert` 完全相同，
12 个聚合 baseline 点不变。没有覆盖旧目录或修改 RTL。
371 个测试通过。新定向测试验证全权重 ready 才计算、未来 DMA 与前层计算
真实重叠、确定性、容量约束和受控 perfect-hit 场景的收益。
