# 主模拟器：独立配置 tile allocator 与 DMA 顺序

## 范围

本次只扩展 `UnifiedSequenceSimulator`，不修改 RTL，也不通过周期补偿对齐。
默认 `PrefetchConfig()` 保持历史行为：无 adapter 时继续使用 prediction spec
的旧策略；传入 scheduler 对象时，byte 模式仍采用原来的 Up-first 顺序。
历史实验产物没有重跑，不应把它们视为新配置的实验结果。

## 配置

```python
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.prefetch_config import PrefetchConfig

config = AdvancedExecutionConfig(
    f_tile_size=128,
    speculative_byte_budget=9 * 1024 * 1024,
    prefetch=PrefetchConfig(
        allocator="hardware_greedy_v1",
        dma_order="pair_first",
        confidence_encoding="probability",
    ),
)
```

显式 allocator 可选 `full_expert`、`uniform`、`ranked`、`hardware_greedy_v1`。
Uniform 按候选轮转、逐轮增加深度；Ranked 使用浮点 confidence 和
`ranked_depth_decay`，不是 RTL 的整数策略。HardwareGreedy-v1 直接复用
冻结的 Q12 整数 golden model，通过完整整数交叉乘积比较收益与代价。
旧 M9 `HardwareAwareGreedyScheduler` 仍保留，但不等同于 HardwareGreedy-v1。

`probability` 要求 score 在 [0,1]，转换为 `max(1, round(score*4095))`。
`q12` 要求 score 是 [1,4095] 的整数值，可避免已有 RTL confidence 再次量化。
转换后分配结果一致不代表真实 predictor 准确率相同。

### DMA 顺序

| 请求模型 | Up-first | Pair-first |
| --- | --- | --- |
| byte OCM、独立请求 | 支持：计划中所有 GU，然后所有 DOWN | 支持：每个逻辑 tile 的 GU，随后 DOWN |
| RTL finite-slot、原子 pair admission | 明确拒绝 | 支持，包括配置的 descriptor/FIFO/ingress 与重试 |

有限模式中同时保留 GU/DOWN slot 和请求 credit，不能声称这种模式支持独立
Up-first admission。需要它时必须另行设计硬件协议，不能静默移除约束。
`dma_order="legacy"` 在 byte adapter 模式解析为 Up-first，原子模式解析为
Pair-first。显式 DMA 顺序要求显式 allocator 或 scheduler 对象。
同时提供配置 allocator 和 scheduler 对象会报错，避免两套策略互相覆盖。

分配顺序是 **speculative 请求顺序**，不是强制物理完成顺序；Demand 和
promotion 优先级、多通道并发、不可中断 burst 与 burst 边界让路仍由原来的
事件驱动内存资源模型处理。

### 资源与 channel 的含义

- byte 模式预算为当前空闲 OCM 与 speculative byte budget 的较小值。
- 原子模式先生成受 OCM 总容量/配置 payload budget 限制的计划；实际 admission
  再检查空闲 slot、bank、保留 demand 空间、descriptor 和 FIFO credit。
  启用 `retry_future_prefetch` 后保持 allocator 原始次序重试，不重新排成 rank-major。
- `planned_speculative_bytes` 是计划 payload；`initial_allocated_speculative_bytes`
  在原子模式只累计 admission 成功的 payload。slot 占用按实际 slot 大小统计，
  不能把计划 bytes 当作已经 resident 或已经完成的 bytes。
- v1 的逻辑 channel：`round_robin_expert` 用物理 channel 数，Expert ID 取模；
  `single_channel` 和 `striped_tile` 用一个聚合逻辑 channel。四个 DDR bank
  不意味着一个跨四 bank 的逻辑 tile 有四个独立 allocation channel。
- 可配置 `channel_credit_bytes`，长度须与逻辑 channel 数一致；仅 v1 使用它。
  这里是本次计划的 payload credit，不是 DDR 带宽。带宽在实际 DMA 事件中约束。
- 不猜测未来 Router 时间，不保证所有计划在 Router 前完成，也不保证预取不变慢。

## 运行与验证

```bash
python examples/prefetch_allocator_order.py
pytest -q
```

示例运行六种研究组合和三种有限 Pair-first 组合。输出
`results/prefetch_allocator_order/comparison.csv` 及每个配置的完整 JSON，包含
模型、硬件、配置、预测记录、请求和事件 trace。每个资源模式使用自身无预取
baseline，不混用资源约束。示例模型和硬件都是 **synthetic placeholder**，
不是 U200 实测性能，也不是 CEPR/FLAME 真实预测 trace。

新增测试覆盖六种组合的物理发出顺序、有限 credit 下的重试顺序、byte 守恒、
真实路由不变、确定性、整数 golden、多 channel credit 和非法配置。
现有宽候选实验的默认 adapter 顺序不变。RTL 与主模型使用相同 v1 配置时，
本次实现提供对齐所需的策略接口，但不是重新完成一次 RTL 联合仿真验收。
