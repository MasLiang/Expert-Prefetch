# 完整 Switch-base MoE 尺寸 RTL 性能对照

## 范围与实验合同

本实验运行当前 `u200_moe_top` 的真实 Router、16-lane int8 MAC、OCM payload
RAM、四 bank striped DMA、ReLU、partial accumulator/combine 和六层激活依赖。
不是用主模拟器公式替换 RTL 计算，也不是只有 allocator 的 byte-credit 验证。

模型几何：H=768、F=3072、E=8、Top-1，24 个 F_TILE=128 的逻辑 tile。
每个 Up/DOWN fragment=98,304 byte；每个 Expert=4,718,592 byte（4.5MiB）。
六层无预取应实际传输 28,311,552 byte（27MiB），不能缩短矩阵或跳过零权重。
E=16 也提供命令行配置，但没有运行则不列为已验收。

这是完整 **Switch 尺寸的六层 MoE 子图**，不是完整 Transformer 的 attention、
norm/residual，也不使用训练 checkpoint。测试权重提供可独立验证的路由与
矩阵结果；它们是确定性稀疏值，但当前 RTL 不跳零，实际读所有 dense payload
并执行所有 MAC。输入 token=1，batch=1。

## 同配置的四种策略

| case | 预取 | Expert 启动条件 |
| --- | --- | --- |
| A_whole_demand | 无 | 整个真实 Expert 的 Up/DOWN 全部 resident |
| B_stream_demand | 无 | Up tile ready 即执行，随后等待对应 DOWN |
| C_whole_full_prefetch | 前一层期间预测两个候选，每个完整 Expert 全深度分配 | 整个真实 Expert resident |
| D_stream_tile_prefetch | 前一层期间预测全部八个候选，用整数 HardwareGreedy-v1 分配前缀 | Up tile ready 即执行 |

C 与 D 使用相同的两个 Expert payload 预算（9MiB），相同 OCM、bandwidth
responder、MAC 和 Router。C 的两个候选完整 tile 计划与 D 的八候选前缀计划
不同，这正是 full-Expert 与 tile-allocation 的区别。credit 只是计划上限，
不保证每个计划在 target Router 前完成；真正 admission、优先级、commit/cancel
和有限资源背压由 RTL 决定。C 的 whole-resident checker 在每次 GU 启动时执行。

所有资源约束都实际保留：

- OCM=4 banks × 32 slots × 128KiB=16MiB。
- slot 为 128KiB，fragment payload 为 96KiB，不能混淆容量与有效 bytes。
- descriptor entries=128，DMA FIFO depth=64。
- logical DMA channel=1，每个 fragment stripe 到四个 physical bank。
- max burst=256 byte，单共享 Router/Expert 16-lane MAC，core clock=200MHz。
- Expert stride=8MiB，满足 RTL future interface 的 stride 至少为完整两矩阵
  Expert bytes 的检查。Up base=0x01000000，DOWN base=0x10000000，layer
  stride=0x40000000；8/16 个 Expert 的矩阵地址区间不重叠。
- Demand 优先，已开始的 burst 不可中断；错误的未开始预测可取消。
- 不启用“上一层所有真权重完成后才能预取”的人为屏障；下一层预测在当前层
  接受 transaction 后提交，与 Router/Expert 执行并行，OCM 和实际队列约束不变。

## DDR 后端与预测质量：不要误读

本轮完整尺寸四策略 **使用四 bank behavioral AXI read responder**，不是
MIG/RDIMM 全时序模型，也不是 DDR AXI VIP。全部真实 payload 仍经正常 AXI
handshake 写入真实 OCM RAM，再由 MAC 读取，没有 force/deposit/backdoor。

responder 的 startup/beat gap 由命令行设置，首轮均为 0。当前实现 RVALID
在握手后清除再重新置位，因此即使 gap=0，也不是每 channel 每 cycle 一个
beat；还包括真实 RTL AR、burst completion、四 bank 对齐和 OCM 写入开销。
不能将这些配置称为 U200 实测带宽或 DDR 物理峰值。

预测首轮 pattern=0，即每次 Top-1 预测均正确：是 **全命中条件下的性能**，
不是 CEPR 的 84.4% accuracy。其 confidence 使用固定 Q12 3104/352/尾部，
候选排序与策略无关：C 取前两个，D 用完整列表。
pattern=1 会在奇数目标层把真实 Expert 移到排序尾部，是明确的错误预测压力
模式，不能视为校准的 CEPR trace 或实际 predictor 质量分布。

## 数值检查与测量

测试 Router 对每层唯一真 Expert 输出最大 logit；实际 RTL 必须返回该 Expert
和 Top-1 Q15 gate=32768。独立矩阵 fixture 的参考为每行
`y = tile_count × x × (1 + expert_id%2)`，层间激活为
`saturate_int8(y >> ceil(log2(tile_count)))`。full 24 tiles 的 shift=5。
每一层的全部 768 个输出必须等于参考，最后输出经过 drain 后仍需保留。

无条件检查：6 个 Router、144 个 GU、144 个 DOWN、144 次 tile-complete、
6 个真实路由、OCM capacity、无 protocol/data error、物理 bytes 与 telemetry
守恒。`compute_done` 计逻辑 tile，不计单个 GEMM，两个 GEMM 有独立计数器。
源文件运行前后 hash 必须不变。PASS 必须包含 `$finish` 和精确尺寸 marker。

周期区间：第一层 transaction begin handshake 到第六层结果，不含第一层
Router weight 的 host 初始化，也不含最终结果后的清理；包含中间五次 Router
weight reload/层间控制。四种 case 的边界相同。

输出中的 `wait_gu`/`wait_down` 只是对应 RTL 信号为高的周期，**不是完整 AIE
memory stall 分解**，尤其不包含 whole-resident barrier 的全部初始等待。
`expert` 是 compute context busy（可能包含 nonlinear/down 等待），不是纯
MAC busy，不可直接用它减 makespan 或与等待计数相加来算利用率。
`FULL_LAYER_RESULT` 提供 first GU cycle，timeline 提供实际开始事件用于进一步分解。

## 运行方式与产物

```bash
python hardware/m10_u200/scripts/run_switch_full_size.py --jobs 4 \
  --output results/m10_switch_full_size_new_run
```

可改变 bank responder latency/geometry；每组变化必须重新跑同配置的四个 case：

```bash
python hardware/m10_u200/scripts/run_switch_full_size.py \
  --experts 16 --layers 6 --startup 16 --beat-gap 4 \
  --output results/m10_switch_full_size_e16_latency16
```

验收输出为 `results/m10_switch_full_size_verified/summary.json`、`comparison.csv`，每 case
有完整 simulation/console log 与 event timeline.csv，build 有编译/elaboration
日志。CSV 同时提供相对 A 和相对 B 的速度比，不能省略分母。
性能模式 debug=off，不记录全量波形、不启用 coverage；所有显式自检均保留。
HDL watchdog=100ms，进程每阶段 wall-time timeout=1800s。
运行目录必须是新的输出目录，不能与正在运行的任务共用；复制 snapshot 时
每个 case 使用独立 runtime 目录，不共享 XSim runtime 临时文件。

初版完整尺寸运行使用 4MiB Expert stride，虽足够一个独立矩阵，却不满足
future interface 的完整 Expert stride 合同。已明确停止并标为
`STOPPED_INVALID_PREFETCH_STRIDE`；其中完成的 streaming baseline 不被混入
最终新版本四 case 比较。修正到 8MiB 后四 case 重新使用相同源码与布局。
另外加入 transaction handshake 的 1024-cycle 超时和 time-zero 地址/stride
合法性检查，避免再次到长仿真结束才发现激励未被接收。没有修改 DUT。

开发预检查 `--mini` 明确标记 `full_switch_dimensions=false`，与完整尺寸结果
分目录，不能用于最终速度比。沙箱内 XSim 启动失败时可经审批在沙箱外运行
同样源码/参数/检查，不能把 startup exception 当成设计 FAIL 或削弱检查。

完整尺寸结果以 summary 的终止 `status=PASS`、各 case 功能 PASS 和全层 marker
为准；运行中或异常终止的产物不构成完成验收。

## 本次实际结果（四组全部 PASS）

测量为六层序列的总周期；毫秒按 200MHz 换算，不代表已完成该尺寸的物理时序收敛。

| Case | cycles | 模拟时间 ms | 相对 A 加速比 | 相对 B 加速比 |
|---|---:|---:|---:|---:|
| A：整 Expert 按需加载后计算 | 3,013,249 | 15.066245 | 1.0000× | 0.6045× |
| B：无预取，tile streaming | 1,821,517 | 9.107585 | 1.6543× | 1.0000× |
| C：完整 Expert 预取，整 Expert 就绪后计算 | 1,998,342 | 9.991710 | 1.5079× | 0.9115× |
| D：八候选 tile 预取，tile streaming | 2,006,224 | 10.031120 | 1.5020× | 0.9079× |

本配置中 B 最快；不能把 C/D 相对弱 baseline A 的收益写成对 streaming baseline 的收益。
C 包含第一层冷启动；这些数值是短六层序列，不是稳态无限层吞吐。
D 使用当前冻结 HardwareGreedy-v1，并非 Uniform/Ranked allocator 的 RTL 测量。
预测 Top-1 全命中仍不意味着全候选正确：C 的第二候选、D 的其他七候选仍可能消耗无用传输。

### 负收益的可核对分解

`timeline_partition.json` 精确分割同一 makespan，D 相对 B 的差为：

| 观测事件间隔 | 差值 cycles |
|---|---:|
| 各层开始 → 首个 GU 开始 | −17,810 |
| GU 开始 → 对应 DOWN 开始，逐 tile 求和 | +133,696 |
| DOWN 开始 → 下一 tile GU 开始，逐 tile 求和 | +68,821 |
| 最后 DOWN → 层结果 | 0 |
| 层间间隔 | 0 |
| 总周期 | +184,707 |

因此“更早开始第一个 tile”确实发生，但不保证整个层更快。D 的 DOWN 等待信号累计
133,692 cycles，B 为 0；D 的首 GU 优势被后续阶段间隔增加抵消。
这里的事件间隔包含计算、控制与等待，不能全部标作 memory stall，也不能再与等待计数相加。

B 实际读取 27MiB；D 读取 40.40625MiB，其中 wasted speculative 为 13.40625MiB。
D 的 useful speculative 9.1875MiB 加 demand 17.8125MiB 正好为所需 27MiB；
canceled speculative 21.46875MiB 没有成为实际 payload 传输。
C 实际读取 37.78125MiB，其中 wasted speculative 10.78125MiB。
峰值占用 B/A 为 48 slots，C/D 为 112 slots，均不超过 128 slots。
额外无用传输与更高占用支持资源竞争解释；当前日志不单独证明每个延迟周期究竟来自
DMA 排队、slot 释放还是控制。因此不能用这些结果声称预取普遍无效。

### 验收与复现边界

四组均通过真实 Router、六层完整输出、自检计数、无协议错误、OCM 与传输 byte 守恒。
每层 768 个输出参考值依次为 384、576、432、624、456、672；最终 drain 后再次检查。
完整回归为 **505 tests passed**，14 条既有依赖弃用警告。时间分解另有因果顺序和守恒测试。
没有修改 DUT RTL/XDC，summary 保存运行前后校验通过的源文件 hash。

这些是完整 H/F/E 尺寸 MoE 子图的功能/性能仿真，不是训练权重模型精度验证、完整 Transformer、
真实 DDR 时序测试或该 16MiB OCM 配置的综合/PnR 验收；特别是不能把当前 RAM 实现直接
等同于 U200 上已完成的 BRAM/URAM 资源映射。E=16 或非零 DDR latency 示例尚未运行。

时间分解复现：

```bash
python hardware/m10_u200/scripts/analyze_switch_full_size_timelines.py \
  results/m10_switch_full_size_verified
pytest -q
```
