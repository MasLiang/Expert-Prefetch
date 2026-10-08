# M10.1：有限控制资源、burst 仲裁与控制时间对齐

本报告记录上一轮未修改设计 RTL 的验证。后续已补充接口节拍并优化四 DDR adapter，
最新结果见 [接口时间与 RTL 优化](M101_INTERFACE_TIMING_OPTIMIZATION.md)，历史结果保留。

## 结论和验收边界

本轮在主模型 `UnifiedSequenceSimulator` 中补充可配置的架构约束，未新建性能
模拟器，未修改设计 RTL/XDC 或 Scheduler 算法。testbench 只增加观测和实验参数。
新增规则默认关闭，不自动改变 M7–M9 的研究配置。

9 组 Vivado XSim 2023.2 实验均通过功能检查、资源上界检查及七项执行规则比较：
descriptor admission 顺序、DMA descriptor 顺序、物理 burst 顺序和大小、tile 计算顺序、
speculative 字节分类、OCM allocation 峰值、predictor 到达时间。
256 个 GU/DOWN stage 的时长一致，分别为 40/24 cycle。

**这不是完整资源/逐周期对齐通过。** FIFO 峰值只在两组实验完全一致，总周期仍有
34–72 cycle 的差距。输出保留 `resource_equivalence=false` 和 `alignment_passed=false`，
不能以执行顺序一致代替性能误差验收。旧 `M10AlignmentSimulator` 继续默认拒绝使用。

## 配置及来源

配置位于 `configs/hardware/u200_control_alignment.yaml`。共同激励使用小型合成模型：
H=16、F=32、E=6、K=2、单 token、INT8、每个 Expert 两个 F-axis tiles。
GU/DOWN payload 分别为 512/256 byte，各占一个 512-byte slot；OCM 为两个 bank，
每 bank 2 或 4 slots。一个逻辑计算 Engine 代表 RTL 的共享 MAC，并不是 U200 有 AIE。

内存模型为四通道、每通道 16 byte/cycle、256-byte burst、每通道最多一个 active burst。
200 MHz 仅用于配置中的单位换算，**不是时序闭合或板级实测结论**。
DDR gap 实验中的效率 0.5 是事件级吞吐近似，不等于逐拍 AXI backpressure 模型。

新增 `ExecutionRules` 参数如下（cycle 参数均为整数）：

| 参数 | 本夹具值 | 建模意义 |
|---|---:|---|
| descriptor_entries | 8，压力组 4 | 有限 live descriptor 容量 |
| dma_fifo_depth | 4，压力组 2 | demand/promoted/speculative 三个独立 FIFO |
| descriptor_retirement_cycles | 2 | 完成/引用退休后 valid/free 传播的结构估计 |
| burst_arbitration | true | 在物理 burst 边界重新考虑高优先级请求 |
| dma_selection_cycles | 4 | lookup、claim、setup、launch 的结构初值 |
| dma_burst_gap_cycles | 1 | 同 descriptor 连续 burst 的间隔估计 |
| router_load_cycles | 0 | 本夹具初始权重在计时前已装入 |
| router_control_cycles | 53 | Top-K、定点 softmax 和 truth 输出的结构估计 |
| demand_dispatch_cycles | 5 | route_last 后 drain/done 固定部分 |
| route_scan_cycles_per_entry | 1 | resolver 扫描 table 的线性部分 |
| tile_dispatch_cycles | 4 | GU 启动控制初值 |
| nonlinear_handoff_cycles | 1 | 非线性引擎交接初值 |
| layer_combine_cycles | 4 | Expert 完成后的 combine 控制尾部 |
| layer_retirement_cycles | 15 | 当前夹具层间控制、host 重装和探测开销 |

Router 控制估计为 E×K 扫描 12、K 个 EXP 2、34×(K−1) 除法 34、
末次余数保存 2、BEGIN 1、K 个 EMIT 2，共 53 cycle。
resolve→新 demand 的估计为 5+descriptor_entries cycle。
层间 15 cycle 包含 chain 四个状态、host 重装 7 cycle、bench 负向探测 4 cycle；
最后一项属于测试协议，不能外推为真实应用的固定开销。
这些参数事先声明，不读取 RTL 总周期做回归拟合，也不保证状态之间不存在重叠。

## 描述符与 FIFO 生命周期

GU/DOWN pair admission 原子检查 OCM、descriptor 和所属 FIFO credits。
没有容量时保持计划等待，不创建半个 pair。Demand 优先于 promoted，promoted 优先于
speculative；原有投机 FIFO token 在 promotion 后保留为 stale 引用，不能立即回收。
活跃 speculative 请求保留必要的回队 credit，避免让出 burst 后无法重新入队。

descriptor 只有在 payload 完成或 cancel、truth 分类已知、所有 FIFO 引用退休后，
再经过配置的 retirement 延迟才可回收。DMA 先完成而 truth 未到达时，字节不提前
判定 useful/wasted。错误请求的退休延迟从 truth 到达时起算，不能使用旧 DMA 时间。
`CONTROL_CREDIT_READY` 事件唤醒因回收等待的 admission，防止无 DMA 活动时死等。

模型记录 live descriptor、FIFO 峰值、tombstone 退休数和 credit 背压次数。
这里的背压次数是事件级 admission 尝试计数，不是 RTL 逐拍 stalled-cycle 计数。
逻辑 request ID 单调递增，用占用和引用生命周期表达 table 容量；未逐位模拟物理
request ID 环回/generation 的 ABA 防护。因此容量不越界不代表全部控制资源时序等价。

## Burst、Router 和 predictor 的时间语义

已经开始的 burst 不抢占。Speculative descriptor 可以在 burst 完成后让出通道；
demand/promoted descriptor 保持连续服务，重新选择 spec 时重新支付 selection/setup。
`burst_history` 记录实际 payload 窗口，startup、暂停、gap 不累计传输字节。
stage resident 只在最后一个 burst 完成时产生。

Router truth-ready 与新 demand-release 是两个事件。已命中且 GU resident 的 tile
可以在 truth-ready 后启动，不必等待 resolver 扫完整个 table；miss 的新 demand
仍须等待扫描完成。GU 提前计算不要求 DOWN payload resident，但要求 DOWN ownership。

Router 首次权重装载发生在本夹具接受 layer 输入事务之前，故不计入 layer 时间。
模型提供独立 `router_load_cycles`，测试验证其延迟，但本次没有把 Router host load
放入 Expert DDR 争用队列。不能声称已验证 Router/Expert DDR 带宽共享。

外部 predictor 仍是事务输入，不建模预测算法计算。当前 bench 在 source layer 0
发出未来层事务，到达偏移为 `2 + 3*(target_layer-1) + predictor_delay_cycles`。
三拍步长来自 begin/candidate/teardown 协议；观测值与模型输入逐项比较。
delay=448 对 source layer 已较晚，但对未来 target 仍可能足够早，因此这组没有收益
下降不意味着 predictor 的到达时间不重要。

## 实验结果

| 实验 | 主模型 cycle | RTL cycle | RTL−主模型 |
|---|---:|---:|---:|
| 无预取，3 层 | 1341 | 1410 | 69 |
| 命中，3 层 | 1289 | 1326 | 37 |
| 错误预测，3 层 | 1341 | 1410 | 69 |
| 命中，8 层 | 3419 | 3491 | 72 |
| 命中，更大 OCM | 1289 | 1326 | 37 |
| descriptor/FIFO 压力 | 1285 | 1323 | 38 |
| predictor 延迟 64 | 1289 | 1326 | 37 |
| predictor 延迟 448 | 1289 | 1326 | 37 |
| DDR 响应间隔 | 1297 | 1331 | 34 |

命中三层相对无预取：主模型节省 52 cycle，RTL 节省 84 cycle，收益仍相差 32 cycle。
虽然总周期误差已缩小，不能因此声称收益幅度已经准确。
例如命中组主模型 demand FIFO 峰值为 4、RTL 为 3；更大 OCM 和压力组全部 credit
峰值相等。剩余差异来自 admission/dequeue 的接口流水、AXI address/response gap、
控制阶段是否重叠等；这是后续检查方向，尚未将每一拍差异唯一归因。

另运行 108 点**仅主模型**敏感性扫描：2 种 OCM、2 种带宽效率、3 种 predictor 延迟、
3 种控制开销倍数、3 种预取条件。在 36 个命中条件中均有正收益，控制倍数不使收益
变号，但同一条件下节省周期最多变化 90 cycle；错误预测有 3 个负收益点。
这说明本小夹具的命中收益方向稳定，收益幅度仍敏感，不是 paper-scale 验证。

## 验证和复现

本轮 `pytest`：306 passed。RTL 功能、Lint 和 Python/RTL golden 联合命令退出码为 0，
日志有 77 条 PASS（不是 77 个独立 testbench 的声明）。Lint 未清零：仍有文件命名、
空端口连接、unused signal/parameter、BLKSEQ、SYNCASYNCNET、GENUNNAMED 等警告；
Icarus 对断言中的 `$error` 有综合提示。未重新综合、PnR 或板级测试。

最终九组 XSim 加上原五组规则回归均重建并运行；原配置的主模型周期保持
1026/1010/1026/2680/1010，证明新增约束默认关闭时保留基线。

```bash
pytest -q -o addopts=''
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 alignment-control
make -C hardware/m10_u200 control-sensitivity
make -C hardware/m10_u200 alignment-main \
  ALIGNMENT_ARGS='--output /home/jiawei/workbench/expertPrefetch/results/m10_rules_regression_current'
```

产物为 `results/m10_control_alignment/` 的 summary、每组 resource/order/stage 比较、
main trace、真实 XSim 编译/运行日志及源文件 hash manifest；敏感性 CSV 和 manifest
位于 `results/m10_control_sensitivity/`。历史结果目录没有覆盖。
配置的 `strict_execution_rules=false` 允许诊断矩阵记录规则差异而不中止，**不把差异
当作通过**；功能错误和容量越界仍立即失败。当前九组实际七项规则均通过。

后续应在更紧 credits、更多 candidate、真正接近 target truth 的晚预测窗口下继续
验证差异是否改变策略选择，再决定是否增加接口级控制约束；不建议复制完整 RTL FSM，
也不应直接把残差拟合成一个固定常数。
