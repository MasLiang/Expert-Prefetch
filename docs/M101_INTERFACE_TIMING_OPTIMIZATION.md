# M10.1：接口时间模型补充与四 DDR 通路优化

## 本轮结论

本轮同时补充主模型和优化 RTL，没有把总周期残差加成固定补偿，也没有修改
HardwareGreedy 算法、GU/DOWN 数值计算或非抢占 burst 语义。
九组真实 XSim 2023.2 的功能、控制容量上界及七项执行规则比较通过；256 个
GU/DOWN stage 时长仍一致（40/24 cycle）。Python 回归为 313 passed。

**尚未宣称完整资源或逐周期对齐。** FIFO 峰值仍有差异，八层实验仍差 45 cycle。
所有结果继续保留 `alignment_passed=false`、`resource_equivalence=false`。
旧 `M10AlignmentSimulator` 仍默认失效，现有研究配置不会自动切换接口规则。

## 已确认的误差来源

上一轮只以每通道 16 byte/cycle 描述 DDR 服务，但共同激励的 memory responder 在
R handshake 时清除 rvalid，在下一拍才生成下一 beat（两个互斥 if/else 分支）。
因此即使 `DDR_GAPS=0`，也不是每拍一个 beat。新增接口日志确认连续 beat 间隔为
2 cycle。这是 **testbench 服务节拍**，不是 U200 DDR 实测吞吐。

其他差异包括：逻辑 pair 原子预留与物理 descriptor 分拍发射、最后一个 truth
比第一个晚一拍、候选到达后仍需 allocator 流水、层退休期间禁止新 future admission。
burst 的最后 OCM 写握手与 descriptor-ready 生效也不是同一个抽象动作。

## 主模型新增接口参数

实现仍在 `UnifiedSequenceSimulator`、`MultiChannelMemory` 和 `ExecutionRules` 中。
新增参数默认关闭/保持原服务率，示例值仅在
`configs/hardware/u200_interface_alignment.yaml` 中启用。

| 参数 | 本夹具值 | 语义 |
|---|---:|---|
| dma_request_latency_cycles | 3 | 原子资源预留到请求可供仲裁的接口延迟 |
| dma_request_interval_cycles | 1 | 共享物理发射端口的最小请求间隔 |
| dma_payload_cycle_multiplier | 2 | payload 服务周期倍率，表达 bench 两拍一 beat |
| dma_return_cycles | 2 | payload 完成到 ready 返回的接口尾部估计 |
| router_truth_emit_cycles | 1 | 完整 Top-2 truth 比首个 truth 晚一拍 |
| prefetch_allocator_cycles | 11 | 单候选 context/FEED/选优流水到初始计划可见的估计 |
| block_prefetch_during_layer_retirement | true | 层间清理窗口禁止新 future admission |

request 可见 cycle 为 `max(issue_cycle + latency, previous_enqueue_cycle + interval)`。
计时期间 descriptor/FIFO credits 已预留，故 FIFO 峰值仍是事件级占用包络，不是
每拍写入的精确峰值。发射可见与退休 deadline 都用事件唤醒，不能因没有 DMA 而死等。

payload 周期按原配置的解析 DMA payload 周期乘倍率；return 尾部不累计 payload。
`burst_history` 保存 payload 窗口末端，实际资源占用末端另存于 `burst_end_cycle`。
Router truth 扩展不改变真实 Expert 集合或 gating 权重。

allocator 初值来自 choice/LAUNCH/FEED/selection-launch、16 行五级选优树、winner/offer
寄存步骤，不读取 RTL 总周期拟合。本实现是初始计划延迟的事件级近似，未逐拍模拟
每个 tile 重新选优、所有 candidate 数量或流水状态之间的重叠。
退层期间计划保持等待，target 开始后停止该层尚未 admission 的预取；不是把
prediction 延迟后的整个窗口都当成可传输窗口。

DDR gap 组保留额外 0.5 带宽效率近似；它与两拍基础节拍不是同一参数。
真实硬件若支持每拍 beat，应使用倍率 1 和实际服务配置，不能照搬本夹具参数。

## RTL 优化

修改设计文件只有 `hardware/m10_u200/rtl/striped_ddr_read_adapter.sv`：

1. 空闲时收到 burst command，AR 直接提供当前地址和长度，取消先寄存再发 AR 的空拍。
   每个 bank 独立握手；未接受的 AR 转入 pending 寄存器，地址/长度保持稳定。
2. 最后一个 joined payload 与 OCM 写握手同拍产生 burst_complete，取消完成信号的
   额外寄存空拍。没有在最后写入之前宣告完成，也没有允许抢占正在传输的 burst。

新增组合路径可能影响时序；本轮不以功能加速代表 200 MHz timing closure。
没有修改 XDC、DDR controller/CDC、AXI 位宽或 OCM ownership 协议。

testbench 添加 AR 背压稳定性、首拍 AR 发起、末 payload 同拍完成断言，并保留
payload/slot/epoch/generation、RRESP、full/partial burst、RAM 数值检查。
完成观测从注册 pulse 改为 handshake 条件后，bench 在实际采样边沿之后检查计数；
不是删除守恒检查来得到 PASS。

## 前后对照

以下“旧 RTL”来自保存的上一轮相同激励日志，没有覆盖历史结果；新值来自本轮重建。
模型参数与 RTL 分别改变，不能将总周期缩小全部归功于 RTL 优化。

| 实验 | 旧 RTL | 新 RTL | 新主模型 | 新 RTL−主模型 |
|---|---:|---:|---:|---:|
| 无预取三层 | 1410 | 1398 | 1389 | 9 |
| 命中三层 | 1326 | 1322 | 1307 | 15 |
| 错误预测三层 | 1410 | 1398 | 1389 | 9 |
| 命中八层 | 3491 | 3487 | 3442 | 45 |
| 更大 OCM | 1326 | 1322 | 1307 | 15 |
| descriptor/FIFO 压力 | 1323 | 1318 | 1303 | 15 |
| predictor 延迟 64 | 1326 | 1322 | 1307 | 15 |
| predictor 延迟 448 | 1326 | 1360 | 1348 | 12 |
| DDR gaps | 1331 | 1330 | 1323 | 7 |

无预取层自身时间现在为模型 453、RTL 456；首个 GU 是 113 对 114。
命中后的层自身时间为模型 412、RTL 418；首个 GU 为 72 对 76。
truth-complete 在各层自身起点后 68 cycle，两侧一致。

三层命中收益：上一轮模型 52、RTL 84 cycle；本轮模型 82、RTL 76 cycle。
收益残差从 32 缩至 6 cycle，但八层和 FIFO 剩余差异说明这不是普遍精度保证。

### 晚预测反例

固定 predictor offset=448 时，两条候选仍分别在 source 起点后 450、453 cycle 到达。
更快的源层缩短了 allocator 可完成 admission 的时间窗口，第一个 target 预取错过
层退休边界。新 RTL 因此比旧 RTL 多 34 cycle；**优化局部接口不保证所有调度输入
端到端单调加速**。不能改 predictor offset 来掩盖这个例子。

最初 bench 不加区分地要求所有 FUTURE_MODE 都发生至少两次 early admission，这不适用于
晚事务。现在仅对 delay=0 保留这一明确 overlap 合同；所有延迟值仍无条件检查
真实数值、操作数量、因果依赖、drain、OCM 上限及 speculative 字节守恒。
模型补入 allocator 延迟及层间 admission 禁止后，这组字节分类也与 RTL 一致。

## 产物与复现

```bash
pytest -q -o addopts=''
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 alignment-interface
```

`results/m10_interface_alignment/` 保存九组比较、main trace、输入、源文件 hash、
XSim 编译/运行记录。新增 `pipeline_comparison.csv` 按每层自己的起点比较 truth、
first-GU 和 layer-done，避免混入上一层残差；`rtl_interface_timeline.csv` 保存各 bank
实际 AR/R handshake 和 joined payload 写入事件。日志还保存 descriptor READY 事件。

RTL 联合回归退出码 0，77 条 PASS（不是 77 个独立 bench 的声明）；Lint 未清零，
警告分类与前轮相同，没有新的 LATCH、MULTIDRIVEN、UNOPTFLAT 或 WIDTH 警告。
本轮未重跑 paper-scale DSE、未运行 post-route/板级验证。

另用现有 `scripts/synth_ooc.tcl` 执行独立小配置 U200 OOC 综合：MAX_CANDIDATES=4、
TILES_PER_EXPERT=4、descriptor=16、OCM 两 bank 每 bank 4 slots、DMA FIFO=4，
其余采用顶层默认参数（不是此次三/八层仿真或完整默认资源规模）。综合与报告生成成功，
保存新 DCP；synth_design 阶段为 0 Error、0 Critical Warning，仍有 273 Warning。
资源为 28304 LUT、23294 FF、2 RAMB36、64 RAMB18、64 DSP。
报告位于 `results/m10_interface_alignment/synthesis/`；不要与此前不同规模的默认
工程资源占用直接比较，也不能将 OOC 通过解释为板级带宽或 timing closure。

后续仍应扩大 credits/candidate/预测时刻的边界矩阵，明确每 tile allocator 的服务
间隔及 FIFO 精确入队时机，再决定是否增加模型粒度；不复制整个 RTL FSM，也不
用当前小模型的总周期回归拟合真实性能参数。
