# M10 RTL / Simulator 等待指标拆分

后续口径收紧见 [功能边界契约v1](M101_FUNCTIONAL_BOUNDARY_CONTRACT.md)：
本页保留上一轮状态计数数据，95/256不是完全同口径，不应直接解释为控制路径相差161cycles。
最新比较按实际请求握手及GU_START独立边界计算。

## 结论

已实现只读计数、逐cycle可审计时间线和严格守恒检查，并实际重跑全部8个共享激励。
223个Python测试、52个既有RTL运行点和8个XSim完整默认配置自检均通过。
设计RTL、XDC、硬件配置、输入激励和所有服务时间参数均未改动。
8点的总周期、旧等待和speedup与上一轮完全一致；26份设计RTL源码hash一致。

命中预取用例的真正权重缺失等待为0；原15–17cycles全部来自控制快照等待。
Demand/wrong的旧等待误差主要也是控制成分，但仍有5/7cycles的真实就绪时间差。
原4-cycle等待容差没有放宽，完整性能验收仍未通过。

## 指标定义

新测量窗口要求Router truth已完整接收、尚有未完成真实Expert、execution未running、
未处于两级combine。窗口包含issue_pending和请求握手所在采样周期。
它是Expert tile发射窗口，不是整个layer idle，也不是MAC利用率的补集。
Nonlinear期间execution仍running，故不计入本窗口。

在窗口内检查**所有真实Expert的下一prefix tile**，不能因Router序第一个Expert未就绪
而忽略另一个已经就绪的真实Expert，也不能把不同Expert/tile的GU和DOWN拼成一对。

| 指标 | 定义 |
|---|---|
| `memory_wait_cycles` | 没有任何可执行prefix的GU+DOWN物理payload完整驻留 |
| `ready_control_wait_cycles` | 至少一对完整驻留，但尚未进入running |
| `expert_issue_idle_cycles` | 上述互斥两项之和 |
| `weight_wait_cycles` | 保留原gu_found快照口径，不改变历史含义/验收 |
| `legacy_memory_wait_cycles` | 旧计数窗口内，物理权重确实尚未就绪的部分 |
| `legacy_control_wait_cycles` | 旧计数窗口内，物理权重已就绪的部分 |
| `issue_pending_wait_cycles` | 新窗口中请求寄存/握手等待子集 |
| `snapshot_refresh_wait_cycles` | 权重已就绪但gu_found快照未反映的控制子集 |

后两项是诊断子集，不能作为额外等待再次相加。检查以下恒等式：

```text
expert_issue_idle_cycles = memory_wait_cycles + ready_control_wait_cycles
weight_wait_cycles = legacy_memory_wait_cycles + legacy_control_wait_cycles
```

“内存等待”表示实际权重可用性缺失，仍可能包含demand尚未发出时的resolver/admission延迟，
不能把它全部解释为DDR物理传输延迟。未running时已有完整payload但promotion未结束，
属于ready控制等待。内存等待进一步按resolver/排队/DDR划分不在本次范围。

## RTL的独立就绪判断

testbench绕过注册的gu_match/gu_found匹配树，扫描execution保存的有效descriptor，
核对真实Expert及其当前next_tile。物理OCM必须满足payload_complete、非FREE、
没有cancel/quarantine，且epoch和generation与descriptor ownership相符。
已知truth后，命中的speculative resident也算物理权重就绪，不等待promotion才算驻留。
所有内部引用均为RTL模式下的只读观察，不写回、不force、不修改功能数据通路。

按posedge、NBA写回之前采样；一次观测代表`[cycle, cycle+1)`。
事件文件记录WAIT_MEMORY/WAIT_CONTROL；脚本合并连续同类样本为EXPERT_ISSUE半开区间，
拒绝重复cycle、计数不守恒、负计数、子集超父集以及时间线/计数不一致。
网表模式不读取内部对象，不输出本次新增的等待分项。

## Simulator口径

仅在M10AlignmentSimulator增加观测，不改变任何排程或耗时计算。
原weight_wait本来依据直接OCM residency，因而与新memory_wait相等。
ready到GU开始之间的原tile_dispatch_cycles记为ready控制等待（本配置每tile8cycles）。
当前32个tile合计256cycles；RTL观察为95cycles，二者并不相等。
这个161-cycle控制差异表明解析dispatch近似不同，不能因总周期接近就声称控制流水已对齐。
仍保留每tile3-cycle retirement/combine近似与RTL仅Expert尾tilecombine的历史模型差异。
未按RTL实测值重新拟合这些参数，也未改历史UnifiedSequenceSimulator或Scheduler算法。

## 实际结果（整数cycle）

| 用例 | 旧RTL等待 | RTL内存等待 | Sim内存等待 | 内存差RTL−Sim | 旧等待中控制部分 | RTL新控制等待 |
|---|---:|---:|---:|---:|---:|---:|
| demand_fast | 132 | 115 | 110 | 5 | 17 | 95 |
| hit_fast | 16 | 0 | 0 | 0 | 16 | 95 |
| second_hit_fast | 15 | 0 | 0 | 0 | 15 | 95 |
| wrong_fast | 132 | 115 | 110 | 5 | 17 | 95 |
| demand_gaps | 144 | 126 | 119 | 7 | 18 | 95 |
| hit_gaps | 17 | 0 | 0 | 0 | 17 | 95 |
| second_hit_gaps | 16 | 0 | 0 | 0 | 16 | 95 |
| wrong_gaps | 144 | 126 | 119 | 7 | 18 | 95 |

例如demand_fast原差22 = 17控制 + 5内存；demand_gaps原差25 = 18控制 + 7内存。
新窗口比旧窗口广，包含全部ready-to-issue控制阶段，所以95不能直接加到旧weight_wait上。
所有点issue_pending子集32cycles；快照刷新子集等于本表旧窗口控制部分。

语义8/8通过，总周期和speedup仍通过原容差。新内存等待的4-cycle容差仅命中点通过，
整体`memory_wait_tolerance_passed=false`，旧综合门槛`timing_passed=false`。

## 复现与产物

从仓库根目录运行：

```bash
PYTHONPATH=src python hardware/m10_u200/scripts/compare_u200_default_alignment.py \
  --output results/m10_wait_split_alignment
python -m pytest -q
make -C hardware/m10_u200 rtl-test
```

`--strict-timing`会因未满足原性能容差而返回失败；默认仍区分功能通过和性能未通过。
新结果目录与旧`results/m10_default_alignment`独立，保留上一轮网表和性能证据。

- `comparison.json/csv`：旧等待、新两项、差值、守恒后的语义与原性能门槛。
- 每case的`rtl_summary.json`、`simulator_summary.json`：全部原指标及新增分项。
- `events.csv`和`rtl_timeline.csv/simulator_timeline.csv`：可审计等待区间及原资源区间。
- `input_manifest.json`：共享配置、数值fixture和源码hash。
- `rtl_build`与每case运行日志：XSim编译、展开、TEST_PASS、正常finish及命令/资源记录。
- `pytest.log/rtl_regression.log`：本轮回归证据。

验证后端为Vivado/XSim 2023.2，目标U200 xcu200-fsgd2104-2-e，200MHz时钟激励。
Python测试仍运行于当前环境Python3.9.19，不冒充Python3.11环境验证。
本次未重跑网表、综合或PnR；不宣称时序闭合、SDF仿真或实板带宽验证。
共享激励仍是单候选、单token/layer、Top-2、identity数值fixture，不扩大为任意工作负载证明。
