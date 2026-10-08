# GU-only 统一后的剩余对齐检查

## 结论与原则

当前 RTL 已删除旧启动参数，GU payload 就绪即可开始，DOWN 在自己的计算阶段等待。
主 `UnifiedSequenceSimulator` 在 `startup_gu_tiles=1` 时也采用 GU-only 启动。
这里只统一了一个依赖条件，不能据此宣称整体性能已对齐。旧 M10 adapter 仍是历史 pair-ready 模型，
其两个比较入口已明确拒绝当前 RTL；之后的正式实验应使用主模型，而不是改旧 adapter 来拟合 RTL。

对齐目标是同一工作负载、同一资源约束和同一策略，而不是强迫 Simulator 复现所有寄存器边沿。
可消除的控制等待优先优化 RTL；物理资源与依赖约束必须如实表示；不可消除的有效服务时间使用
配置与独立微基准校准，不从整层延迟差值反推常数。

## 仍有实质差异

| 项目 | 当前 RTL | 主模型 | 建议 |
|---|---|---|---|
| OCM 分配和回收 | GU/DOWN pair 一起预留；GU_DONE / DOWN_DONE 独立释放 | 每 stage 独立 admit；GU/DOWN 各自在 AIE 操作完成时释放 | 回收规则已进一步统一；atomic pair admission 仍是独立差异，不能据此宣称整体对齐 |
| OCM 容量计费 | 固定 4096-byte slot，四 bank 各 32 slots；还有 bank 与 descriptor credit | 按 payload byte admission | 不能只设置相同总字节就称等价；应比较实际 slot/bank/FIFO/descriptor 可用预算。硬件几何约束不是控制等待 |
| 就绪候选选择 | 注册 ready 搜索，多个候选同时可启动时按 Router 顺序选 | 启动候选按 token_count 降序、Expert ID 升序；已有 active Expert 会继续推进 | 需用多个候选同时就绪且 Router 顺序与 ID 顺序相反的激励验证；主模型已有就绪选择，不必迁就 RTL Router 顺序 |
| DMA 请求顺序 | true tile demander、pair reservation 和 FIFO 决定发出顺序 | 先当前 startup stage，GU 启动后主动请求下一 GU tile | 需用慢带宽和有限 OCM 检查发出序列；可优先优化 RTL 为下一 GU 的 lookahead/紧迫度调度，不能忽略 DOWN deadline |
| memory 通道语义 | 四物理 bank 聚合 stripe，同一 descriptor 锁定聚合 DMA | 可配置多个可独立执行请求的 memory channels | 四 bank 不等于四个可并行 descriptor；初期用匹配聚合服务率和单请求并发的配置，不能套用独立四 channel 假设 |
| 实际计算服务 | INT8、16 lanes、tagged II=1 chunk 流水，独立 Router Top-K/softmax 控制 | 按 operation count 与配置有效吞吐率计时 | 使用新流水微基准重新校准服务配置。主模型不必增加每级 MAC 的内部 FSM |
| 计算引擎占用 | tile controller 整 tile busy，等待非线性/DOWN 时不接新 tile | AIE pool 和 nonlinear 分开预留；配置允许多个 active Expert | 首轮明确 max_parallel_experts=1、pool=1；后续若要利用等待窗口运行别的 Expert，RTL 必须有多个 partial context，不能只放开 busy |
| 多层和提前预测 | 当前 U200 top 的 USE_SEQUENCE_CONTROL=0，清理/结果 context 仍以当前事务为主 | sequence、source milestone、source/target layer 和 lead time 都已建模 | 当前单层八点不能证明多层对齐；后续接入真实 layer context、activation 依赖和预测 target 标识，不把主模型退化为单层 |
| predictor 与 allocator | 外部事务、Q12 confidence、有限宽度整数分配 | 外部 trace/候选聚合与研究策略，可使用不同分配规则 | 固定相同候选、rank、量化后的 score、tile budget 和策略，再比较 allocation/issue 序列；不能只比较 recall |
| 计时窗口 | 启动 idle、controller 内 WAIT_DOWN、stage 执行与 combine 是不同窗口 | starvation 包含 stage 等待，输出边界定义不同 | 使用统一功能事件对齐；目前启动 memory_wait 不含运行中 WAIT_DOWN，不能与模型全程 starvation 直接相减 |

代码依据：`unified_sequence_simulator.py` 的 `_schedule`、`_progress`、`_aie_done`、
`_refill_demands`、`_issue`；RTL 的 `tile_execution_scheduler.sv`、`tile_compute_controller.sv`、
`true_tile_demander.sv`、`ocm_slot_ownership.sv`、`int8_swiglu_tile_backend.sv` 与 `u200_moe_top.sv`。

## 优先考虑的 RTL 降周期工作

以下观察来自优化前冻结的 `results/m10_gu_only_start/`。第一步的控制器 ready-stage
交接已完成，见 [控制交接优化](M101_COMPUTE_HANDOFF_OPTIMIZATION.md)：八点减少 63–64 cycles。
后续的注册搜索 lookahead 也已完成，见 [第二轮优化](M101_SCHEDULER_LOOKAHEAD_OPTIMIZATION.md)：
在第一轮基础上再减少 30 cycles。随后已完成 GU 阶段独立释放，见
[独立回收](M101_GU_STAGE_RELEASE.md)：小容量复用验证通过，默认八点 cycles 不变。
稳定请求级仍保留；DMA 紧迫度/lookahead 仍是后续工作。
MAC chunk 流水现已完成数值与功能验证，见
[第四轮优化](M101_MAC_CHUNK_PIPELINE.md)。本文件下述串行 MAC 数值属于历史背景；
有效服务率应按新流水重新独立校准，DMA 紧迫度/lookahead 仍未改动。

1. **压缩逐 tile 的调度/交接空档。** 当前 demand_fast 的 32 tiles 共计 ready→accept 63、
   accept→GU 64、DOWN_DONE→retire 32 cycles，另有 previous_retire→window 33 cycles。
   这些是事件观测分项，不是可直接相加的收益预测。可提前准备下一请求，在合法退休边沿接力；
   必须保留 stale-search 重验、背压稳定性、flush 与 generation 保护，且重新验证 200 MHz 时序。
2. **GU_DONE 后释放 GU slot（已完成）。** GU 数据已经被消费，释放可给新 GU/DOWN 请求腾空间，尤其有利于
   小 OCM。需要将现有 pair retirement 分拆，防止最后的 pair release 误释放已复用的新 generation。
   这不是删一条等待判断即可完成的优化。
3. **下一 tile 请求 lookahead。** GU 计算期间准备后续 GU，并按 DOWN 的实际需求时间调节 DMA 优先级。
   先在同一 byte budget 下验证是否减少关键路径等待，避免提前加载后续 GU 导致当前 DOWN 饥饿。
4. **降低 MAC initiation interval（已实现功能流水）。** 原 FSM 的 READ→MAC→PAIR→QUAD→OCT→TOTAL→ACC→STORE
   多级运算没有重叠不同 chunk。可让同步 RAM 读、乘法和 reduction 流水重叠，配合 row/chunk/tag
   及反馈累加设计；不能仅删除状态或把整棵加法树塞进一个周期。现已实现 tagged II=1 chunk 流水，
   微基准与完整 XSim 通过；综合估计/布线后 STA 必须区分，不能将仿真当时序闭合。

先做第 1 项，其次第 2/3 项；第 4 项更可能影响主要 compute 时间，但验证与时序风险也更大。
不把这份建议当作上述优化已完成的声明。

## 本轮完成与验证

完成：移除 `GU_EARLY_START` 的三个 RTL 层级接口和启动分支；调整 TB 启动等待观测；
阻止旧 adapter 静默运行；保留历史结果用于比较，没有改主模型的算法或服务时间。

行为仿真为 XSim 2023.2。显式 non-project 源列表是 U200 top 与自检 TB，未更改既有 Vivado 项目。
检查到已有 MCP 会话的 current_project PART 是 `xc7vx485tffg1157-1`，并非 U200；
行为仿真不能据此证明 U200 综合或布局布线通过，本轮没有使用该项目作目标器件验证。

237 项 Python 测试、53 个 RTL 回归运行点、8 个完整配置 XSim 自检通过；lint 成功但有既有警告。
八点周期仍与上一轮 GU-only 完全一致：demand/wrong fast=13909，gaps=13916，hit=13810。
GU-only 清理没有进一步改善性能，它只是删除了不再需要的旧模式。
尚未验证新的综合网表、200 MHz STA 或主模型多层联合对齐。

```bash
PYTHONPATH=src pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test lint
make -C hardware/m10_u200 gu-early-test
```

结果位于 `results/m10_gu_only_start/`，冻结旧规则数据仍在 `results/m10_boundary_alignment/`。
