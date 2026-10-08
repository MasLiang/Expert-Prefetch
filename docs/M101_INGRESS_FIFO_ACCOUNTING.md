# M10.1：层间资源退休与 FIFO 统计拆分

## 修改范围

本轮补充 `UnifiedSequenceSimulator` 的接口事件模型与比较脚本。设计 RTL/XDC 未修改；testbench 仅增加 `MAIN_RETIRE` 被动观测。原有逐 descriptor 配置和历史结果保留。

独立配置为 `configs/hardware/u200_ingress_alignment.yaml`。这是小型共同激励配置，不是 U200 实测性能：H=16、F=32、E=6、K=2、INT8、F_TILE=16、512-byte slot、四路 DDR 接口。

## 层间交接

过去只有固定层间 retirement gap，无法解释 ingress 恰好跨过交接边界时多出的等待。现在增加共享 ingress pair 资源及完成事件：接受 GU/DOWN logical pair 后，经过配置的 3 cycles 才可接受下一对。

层间 retirement 检查在源层完成后启动，等待：

- ingress pair 资源实际空闲；
- 源层 resolver 条目与 route/demand-release 控制活动退休。

资源就绪后才支付剩余交接尾部时间。原有层间 background/segment 依赖仍分别检查。不会要求所有未来层 speculative DMA 完成，也不会要求整个 descriptor table 清空，否则会错误地阻断跨层预取。

新增配置为 `ingress_pair_cycles=3`、`layer_retirement_probe_cycles=2`、`layer_retirement_wait_for_ingress=true`；既有 `layer_retirement_cycles=15` 分解为检查前阶段与剩余尾部，不是额外叠加 15 cycles。

这是接口事件级模型，不声称每个 RTL 内部寄存器状态均逐周期一致。logical reservation 与物理接受阶段可以存在相位差；验收重点是资源依赖、附加等待和最终 layer start。

### RTL 直接证据

`late_445` 的 DRAIN 在绝对 cycle 472 已满足 ingress idle / retire ready；`late_446` 在 cycle 472 两者仍为 0，cycle 473 才变为 1。allocator、demand 和 compute 此时均不忙，resolver ready 为 1。因而后者多等一个 DRAIN cycle，下一层分别在绝对 cycle 486 / 487 开始。

原模型的 `late_446` 总周期为 1322，RTL 为 1323；新模型等待实际 ingress 退休后为 1323。上述绝对周期包含 testbench 前导时间，不直接等于主模型相对周期。

## FIFO 指标

新增三组 entry 单位指标：

| 指标 | 含义 |
| --- | --- |
| `fifo_reserved_credit_peaks` | 已保留的队列容量，包含尚未物理入队及 speculative return 所需容量 |
| `fifo_occupancy_peaks` | 已实际入队且尚未完成物理 dequeue 的 token 数 |
| `fifo_reserved_not_enqueued_peaks` | 保留但未成为实际 FIFO occupancy 的容量 |

逐时刻满足 `reserved_credit = occupancy + reserved_not_enqueued`。三者各自的峰值不一定发生在同一时刻，不能用峰值相加验证守恒。

旧 `fifo_peaks` 保留为历史 logical-token-reference peak，不再用于与 RTL FIFO occupancy 比较。比较脚本明确使用 `fifo_occupancy_peaks`；descriptor reservation/live peak 仍单独比较，不把不同口径强行合并。

`fifo_push_blocks_dequeue=true` 表示 RTL arbiter 在队列 push 时不完成 lookup/pop。被选中的 token 在物理 pop 可见前仍计入 occupancy，并保持 descriptor 引用，防止过早回收。这个握手阶段与现有 DMA selection/setup 抽象重叠，不再额外增加一份 DMA 服务时间。

每个用例输出 `main_fifo_timeline.csv`（逐时刻 credit/occupancy）、`rtl_retirement.csv`（实际交接控制信号）、`resource_comparison.json`。完整运行保留原始日志及 source/config manifest。

## 复现与验收边界

```bash
make -C hardware/m10_u200 ingress-boundary
pytest -q
```

结果目录为 `results/m10_ingress_boundary/`。边界矩阵覆盖 30 组端到端用例，另有 3 组就绪边界与 2 组 commit 观测。本轮未重新综合/PnR，不能作为时序闭合或板级 DDR 带宽证据。

最终计数与未对齐项以该目录的 `boundary_summary.json`、`comparison.csv` 为准。DDR 空拍服务近似和 descriptor 峰值口径仍须独立审查；不能仅凭总周期或 FIFO 峰值相同宣称完整资源等价。

本轮实际验证结果：334 个 Python 测试通过；30 组端到端、3 组就绪边界、2 组 commit 观测功能检查通过。30/30 执行规则和 FIFO occupancy 峰值一致，760 个计算 stage duration 一致，29/30 总周期一致。五组固定计划节省周期在主模型与 RTL 中均为 64、68、76、68、76。

剩余差异没有隐藏：`ddr_gaps` 主模型为 1338 cycles，RTL 为 1330 cycles，仍有 8-cycle 服务近似差；大 OCM 用例 descriptor 峰值主模型 6、RTL 5。因此完整资源等价与完整周期对齐仍标记为 false。

## Vivado DDR 仿真支持

本机 Vivado 2023.2 catalog 已确认 `xilinx.com:ip:axi_vip:1.1` 与 `xilinx.com:ip:ddr4:2.2`。安装目录存在 `data/ip/xilinx/ddr4_v2_2/data/dlib/ultrascale/ddr4_sdram/tb/ddr4_sdram_model_wrapper.sv`、`ddr4_model/ddr4_model.sv` 及 `sim_tb_top_axi.sv`。

- **AXI Slave Simple Memory VIP**：响应 AXI 读写并提供简单内存，适合协议、数据、burst、ID 和背压验证；不能把它当作真实 DRAM bank/row/refresh 延迟模型。[AMD PG267](https://docs.amd.com/r/en-US/pg267-axi-vip/AXI-Slave-Simple-Memory-VIP)
- **DDR4 controller IP + DDR4 SDRAM model**：使用 IP example design/testbench，可验证配置对应的控制器与存储器数字时序。控制器效率依赖请求模式和 DRAM 时序。[AMD PG150 Example Design](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Example-Design)、[Efficiency Workloads](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Efficiency-Workloads)

当前共同激励 testbench 尚未接入 DDR4 IP/model。接入前须匹配 U200 的实际存储器器件、rank、数据宽度、时钟、地址映射和 shell/互连；稳态性能统计应排除初始化/校准阶段。数字仿真也不能替代板级信号完整性与带宽测量。建议保留快速回归，再增加少量代表性 DDR4 控制器级用例。
