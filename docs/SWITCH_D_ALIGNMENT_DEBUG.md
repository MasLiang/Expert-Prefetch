# D case 请求级对齐修复

## 范围与结果

继续使用 H768/F3072/E8、Top-1、六层、INT8 两矩阵 ReLU、16-lane MAC、
16 MiB fixed-slot OCM、四通道 AXI 行为模型。没有引入真实 DDR 器件模型。
使用主 `UnifiedSequenceSimulator`，没有恢复 M10AlignmentSimulator。

修复后 D：主模型 **2,006,406 cycles**，RTL **2,006,224 cycles**，
差 **182 cycles（+0.00907%）**；原差异为 52,138 cycles（+2.59881%）。
没有添加总周期补偿项。

## 主因：reserved credit 被错误当作已入 FIFO 的 token

`MultiChannelMemory.complete()` 在 speculative burst 完成时，以
`bool(demand_tokens or promoted_tokens)` 判断是否有更高优先级工作。
但 tokens 列表还包含已经保留 FIFO credit、实际尚未完成 ingress 的请求。
因此模型可能在物理 FIFO 尚为空时，提前将 active speculative descriptor
放回队尾，然后开始下一个 speculative descriptor。

RTL `burst_dma_channel` 的 `priority_queued` 只查看实际 FIFO count。
在完整尺寸日志中，旧模型第二个首次 DMA 是未来层 Expert 1/tile 0 的 DOWN；
RTL 第二个首次 DMA 是当前层 Expert 0/tile 0 的 UP。这个早期顺序分歧随后
改变了 DMA/OCM 演化与 Router truth 到达时的取消窗口。

修复为只在更高优先级 token 的 `enqueue_ready_cycle <= current_cycle`
时才触发 burst 边界让出。DMA 请求的原始 demand/speculative 分类不变；
不是降低 demand 优先级，只是区分 credit 预留和物理入队可见性。

## 同时补齐的两个可配置资源规则

1. `prefetch_pair_interval_cycles=6`：HardwareGreedy-v1 在每个成功 pair
   handshake 后重新选择 winner。RTL 日志的连续 speculative pair 接受间隔
   为六拍，主模型此前只有三拍 ingress 成本。增加独立 allocator wakeup 事件，
   不用每周期轮询。默认零保持其他硬件配置的原行为。
2. `demand_ingress_blocks_prefetch=true`：当前 demand admission 游标完成前，
   未来预取不能绕过其 bank 背压占有 ingress。默认 false；只有声明该规则的
   RTL 执行配置启用。**不是等待上一层权重全部 DMA 完成的屏障**。
   定向测试证明：当前层所有 pair admission 完成后，即使最后一个权重仍在
   DMA 中，未来预取也可以恢复。

两项要求 atomic pair 与 future retry，否则抛出清晰配置错误，避免被忽略。
单独补齐这两项后，当前完整尺寸 D 总周期仍为 2,058,362；加入 FIFO 可见性修复
后才降至 2,006,406。故它们不是原 2.60% 偏差的主因，也没有作为周期拟合工具。

## 四组复跑与 byte 守恒

| Case | 修复后主模型 cycles | RTL cycles | 差值 |
|---|---:|---:|---:|
| A | 3,013,531 | 3,013,249 | +282 |
| B | 1,821,517 | 1,821,517 | 0 |
| C | 1,998,389 | 1,998,342 | +47 |
| D | 2,006,406 | 2,006,224 | +182 |

A/B/C 没有性能回归。D 的以下指标与 RTL 一致：

| 指标 | 两侧共同结果 |
|---|---:|
| demand bytes | 18,677,760 |
| speculative bytes | 23,691,264 |
| useful speculative bytes | 9,633,792 |
| wasted speculative bytes | 14,057,472 |
| canceled speculative bytes | 22,511,616 |
| OCM peak slots | 112 |

此前多出的 1,179,648 wasted bytes 已消除，必需权重仍为
18,677,760+9,633,792=28,311,552 bytes。

## 验证方法与边界

按照 `vivado-simulate-rtl` 的流程，重新编译/展开默认关闭的 handshake-only
观测 testbench，以 XSim 2023.2 运行完整尺寸 D，保留终止 PASS、golden 输出、
protocol/data-path checker 和 watchdog。观测包括 pair admission、descriptor
claim、DMA completion、Router promotion/cancel，不驱动 DUT，不抓取全设计波形。
30 个 DUT RTL 源文件的 SHA256 与此前四 case 结果完全一致；没有修改 RTL/XDC。

请求级比较工具独立检查首次 DMA 顺序、compute 顺序、五类 byte 统计和
峰值 slot，不能用总周期相同代替上述门限。DMA 恢复后的重复 claim 单独去重，
不当作重复首发。

完整六层 D 新编译 XSim 仿真已通过，golden/protocol 检查通过，RTL 周期仍为
2,006,224，仿真 wall time 531.7 秒。诊断状态为 `REQUEST_ORDER_AND_BYTES_PASS`：
431 个首次 DMA 的全序列与 288 个计算阶段全序列一致，八个门限均通过。
编译/展开没有错误，有一条 testbench 未连接 `router_weight_reload_ready` 输出的
警告；该输出不参与本次夹具控制，仍单独保留警告，不把功能 PASS 宣称为完整 lint 清零。
Python 完整回归为 **543 passed, 14 warnings**，14 条为现有依赖弃用警告。

剩余 182 cycles 的每层净差为 **6、38、35、31、36、36**，合计 182；
层间交接成本一致。它们是本夹具下剩余的局部事件时间偏差，尚未逐 burst
证明全部来自哪一个控制边界，因此不宣称 cycle-exact、形式等价或板级验证。
不应继续使用旧的“额外 1.125 MiB 无用 DMA”解释当前结果。

## 文件与复现

修改主模型：

- `src/edge_moe_sim/hardware/multi_channel_memory.py`：实际 FIFO 可见性判定。
- `src/edge_moe_sim/execution_rules.py`：可配置 allocator cadence/ingress ownership。
- `src/edge_moe_sim/events.py`、`unified_sequence_simulator.py`：资源 wakeup 与仲裁。
- `examples/switch_full_axi_main_comparison.py`：共同夹具启用上述 RTL 执行规则。

验证工具：

- `hardware/m10_u200/tb/tb_switch_full_size.sv`：默认关闭的请求级观测。
- `hardware/m10_u200/scripts/run_switch_full_size.py` 和
  `run_switch_full_size_snapshots.py`：`--alignment-debug` 开关。
- `hardware/m10_u200/scripts/diagnose_switch_full_axi.py`：请求级对照与失败门限。
- `tests/test_prefetch_ingress_arbitration.py`、`test_switch_request_diagnostics.py`：
  最小可见性复现、cadence、资源独占、进度与诊断负向测试。

```bash
PYTHONPATH=src python examples/switch_full_axi_main_comparison.py \
  --output results/m10_switch_axi_main_fixed_20261007

python hardware/m10_u200/scripts/run_switch_full_size.py \
  --cases D_stream_tile_prefetch --jobs 1 --alignment-debug \
  --output results/m10_switch_D_debug_replay

python hardware/m10_u200/scripts/diagnose_switch_full_axi.py \
  --main results/m10_switch_axi_main_fixed_20261007/D_stream_tile_prefetch/main_result.json \
  --rtl-case results/m10_switch_D_debug_20261007/D_stream_tile_prefetch \
  --output results/m10_switch_D_debug_20261007/diagnostics

pytest -q -o addopts=''
```

复跑到新目录时应相应修改 `--rtl-case`。主模型与 RTL 的所有原始输入/结果均保留，
修复前结果在 `results/m10_switch_axi_main_comparison`，修复后在
`results/m10_switch_axi_main_fixed_20261007`。

请求级日志和比较结果位于 `results/m10_switch_D_debug_20261007/`，
其中 `diagnostics/request_comparison.json` 明确标记 `cycle_exact=false`。
