# M10.1 RTL 与性能模拟器：激励适配与第一次对齐

## 一条命令复现

在项目根目录运行：

```bash
make -C hardware/m10_u200 alignment
```

或者指定输入、输出目录：

```bash
PYTHONPATH=src python hardware/m10_u200/scripts/compare_u200_alignment.py \
  --config configs/hardware/u200_rtl_alignment.yaml \
  --trace hardware/m10_u200/traces/alignment_cases.jsonl \
  --output results/m101_u200_alignment
```

需要 Icarus Verilog、Python 项目依赖和 PyYAML。默认模式在语义错误时失败，性能偏差
单独记录；加 `--strict-timing` 后，总周期、等待时间或 speedup 任一超容差均非零退出。
本次完整性能容差尚未通过，因此 strict 命令预期失败，不应把默认退出码当完整性能 PASS。

## 同一输入与硬件约束

输入 JSONL 记录预测 Expert、Q12 confidence、预算、DDR 停顿模式、真实 Expert 和
Q15 gates。当前第一版仅支持明确验证的固定数值 fixture，超出范围会拒绝运行，
不静默套用不一致的模型。

- 模型：单 token、单 layer，H=16、F=32、6 Experts、Top-2，F tile=16。
- 输入 activation 全为 int8 16；Router W3/W4 的第一列为 1，其余 Expert 第一列为 -1；
  其他列为 0，因此 logits 为 -16/-16/-16/16/16/-16，路由为 3、4，gate 各 16384。
- 每个 Expert 有两个 F tiles，Gate、Up、Down 都是 16×16 单位矩阵。
  GU shift=0，Router shift=0；SiLU_Q4(16)=12，最终每个输出元素是 24。
- 一个共享 MAC，int8 乘法、int32 累加；4 operation/cycle 的有效 payload rate。
- OCM：4 个 512-byte slots，GU/DOWN 原子占两个 slots；留两个 slots 给 demand。
  DOWN 实际数据只有 256 bytes，仍占一个完整 512-byte slot。
- 8 descriptors，DMA FIFO depth=4，四 physical DDR banks，每 bank 一个 outstanding；
  所有 stage 按 16-byte stripe 使用四 bank。每 burst 256 bytes。
- tile 必须 GU 和 DOWN 同时 ready 才启动；两 slots 一直保留到 tile 完成。
  不同 Expert 的 ready-first 遵从 Router entry 顺序；同 Expert 保持 prefix 顺序。
- HardwareGreedy-v1，单 aggregate allocator credit；预算为 0 或 768 bytes。
- TB clock 周期 5 ns。fast 模式没有额外 modulo 停顿，但响应产生/接受仍分拍，
  不是 AXI 每拍返回；gaps 模式保留每 bank 不同的周期性 AR/R 停顿。

manifest 和每个 stimulus.json 都保存显式 activation、Router/Expert 矩阵以及数值参考，
方便核查。SV bench 以同一固定矩阵规则生成 payload，driver 将预测/预算/DDR mode
翻译为 elaboration 参数；当前尚不支持任意外部矩阵文件。

## 为什么增加独立适配层

历史 Advanced/Unified 模型不完全等价于当前 RTL：部分模式仅 GU ready 即可启动，
GU 后可释放其容量，或支持多个 Expert 并行。仅修改 clock、带宽 YAML 不能改变这些依赖。

`src/edge_moe_sim/rtl_alignment.py` 增加 `M10AlignmentSimulator`，复用已有
EventQueue、AIEPool、MultiChannelMemory、CapacityOnChipMemory 和 HardwareGreedy-v1。
AIEPool 在这里代表单条 PL MAC 资源，并非宣称 U200 有 AIE。
适配层实现原子 pair admission、固定 slot charging、保留 demand slots、整 stage
不可抢占、commit/cancel、both-ready tile 启动、prefix 和共享 MAC 约束。
historical UnifiedSequenceSimulator、Scheduler 及 M7–M9.3 结果没有修改。

服务时间来自独立的解析配置，不读取 RTL 完成时间再回放，也不拟合 layer 结果。
四 bank 用同步 stripe stage 的聚合服务建模；每个物理 burst 收 setup。
compute 按工作量/有效 rate 建模；Router GEMV 后另计 Top-K、定点除法和事务开销。
这不是 cycle-accurate emulator，也不是直接用历史 simulator 跑出相同实验的声明。

## 输出与验收口径

`results/m101_u200_alignment/` 包含：

- `input_manifest.json`：配置、共用激励、数值 fixture、26 个 RTL 源码的 hash。
- `comparison.csv`、`comparison.json`、`REPORT.md`：逐 case 的精确检查及性能偏差。
- 每个 case：`stimulus.json`、RTL 编译/运行日志、`rtl_events.csv`、`rtl_timeline.csv`、
  `rtl_summary.json`、`simulator_timeline.csv` 和 `simulator_summary.json`。

RTL 只增加 TB 观测，不改设计逻辑。时间以 layer begin handshake 为 cycle 0，
在 posedge 前可见的 valid/ready 采样。NL done 是保持信号，只记录首次上升，不能
把每一拍高电平都误记成一次 completion。

精确验收包括实际 speculative admission、tile compute 顺序、所有 byte 分类、
真路由/Q15 gates、数值输出、OCM 上限、单 MAC 与全-bank DMA 区间互斥，
并核对 DMA trace 与 telemetry 的实际 bytes。
数值输出的对照对象是独立 Python fixture reference，性能 simulator 不执行 GEMV 数值。

`weight_wait_cycles` 是 truth ready 后还有真实 prefix tile 未完成，但没有 ready pair
可以执行的空闲时间，排除执行中、选择寄存级和 combine。RTL 原来的
`controller_demand_stall_cycles` 定义不同，作为额外指标保留，不强行比较两个口径。
完整性能验收分别检查总周期误差、等待绝对误差、speedup 绝对误差。

## 首次结果

8 个 case 覆盖零预算、第一/第二 Expert 命中、全错预测 × fast/gaps：

| 模式 | case | RTL cycles | Simulator cycles | RTL 权重等待 | Simulator 权重等待 | RTL speedup | Simulator speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| fast | 零预算 | 1804 | 1798 | 59 | 42 | 1.00000 | 1.00000 |
| fast | 命中 3 或 4 | 1748 | 1756 | 0 | 0 | 1.03204 | 1.02392 |
| fast | 全错 | 1804 | 1798 | 59 | 42 | 1.00000 | 1.00000 |
| gaps | 零预算 | 1814 | 1807 | 69 | 51 | 1.00000 | 1.00000 |
| gaps | 命中 3 或 4 | 1748 | 1756 | 0 | 0 | 1.03776 | 1.02904 |
| gaps | 全错 | 1814 | 1807 | 69 | 51 | 1.00000 | 1.00000 |

全部语义检查通过；Router MAC done 一致，truth ready 相差 1 cycle。
总周期最大相对误差为 0.46%，但本矩阵受计算时间主导，不能据此断言性能完全对齐。
权重等待低估 17/18 cycles，speedup 低估约 0.0081/0.0087，超出本次配置的
4-cycle 和 0.005 容差，**完整性能验收不通过**。

时间线给出了具体排查方向：RTL truth 最后 entry 为 cycle 103，零预算 case
首个 demand admission 为 cycle 121，首个 DMA 为 cycle 124；event adapter 在
cycle 104 即 admission/启动，未完整表示 resolver、pair admission、queue/control 的
延迟。另一方面 event adapter 的统一 dispatch/combine 估计使后续计算间隙偏大，
两个方向的误差部分抵消了总周期。不能只调整带宽或拟合总周期掩盖这个问题。

下一步应以独立控制/短传输微基准确定 admission/dispatch 服务开销，再用未参与校准
的激励验证；还需扩充候选数、tile depth、带宽受限矩阵和更小 OCM。
本次已完成的是适配、比较脚本和第一次真实运行，不是 M9.3 论文结论复现或上板性能验证。

## 回归验证

```bash
python hardware/m10_u200/scripts/validate_design.py
```

41 个 RTL 运行点、207 个 Python 测试通过，原有 5 allocation traces 和 6 dense
numeric golden 继续通过。新增 11 个 Python 测试验证确定性、slot/byte conservation、
both-ready dependency、MAC/DMA 互斥、第二 Expert ready-first 和不支持配置的拒绝。
