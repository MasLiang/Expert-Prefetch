# M10 完整默认配置：RTL / Simulator / 综合网表对齐

## 1. 结论与范围

本轮已扩展并实际运行共享激励验证：8个XSim RTL/M10 event adapter用例，
以及2个最终综合网表代表点。语义8/8通过，网表2/2通过。
总周期和speedup均通过原容差；等待时间未通过，不能称为完全性能对齐。

后续已完成等待分项并重跑8点RTL，见 [等待指标拆分](M101_WAIT_METRIC_SPLIT.md)。
本文表格及网表数据保留为拆分前原始记录；新的直接OCM就绪观察不再使用gu_found快照。

没有修改任何设计RTL、XDC、历史UnifiedSequenceSimulator或M7–M9.3政策算法。
只修改验证平台、M10独立适配层、配置、入口脚本、测试与中文文档。
这是共同输入的离线交叉验证，不是RTL与Python按时钟在线锁步。

## 2. 完整默认硬件与模型

| 项目 | 配置 |
|---|---|
| 模型 | 单token、单layer，E64、Top-2、H16、F256 |
| Tile | F_TILE16，每Expert16个logical tiles |
| GU / DOWN payload | 512 / 256 bytes |
| OCM | 4 banks×32 slots×4096 bytes，512KiB物理槽容量 |
| Descriptor table / FIFO | 64 entries / depth16 |
| DMA | 单非抢占stage事务，所有stage均stripe到4个128-bit DDR接口 |
| 计算 | 单共享PL INT8 Router/Expert MAC，独立非线性阶段 |
| 时钟目标 | 200MHz，原5ns XDC，不代表已闭合 |
| Predictor | 外部单候选transaction，Q12 confidence=3000 |
| Allocator | HardwareGreedy-v1，单aggregate channel credit |

地址为GU base=65536、DOWN base=131072、Expert stride=16384；
DOWN physical_tile_bytes=256。OCM按完整slot收费，不能按payload紧密装填。
银行hash采用epoch/layer/expert/tile/matrix XOR，逐bank容量约束显式检查。
AIEPool在adapter中只是复用的单计算资源抽象；U200没有被描述为具有AIE。

H16/F_TILE16是验证计算规模，不是生产MoE模型、FLAME性能复现或U200实测带宽。
板级DDR PHY、shell、CDC、host/XRT与PnR未接入。

## 3. 共享输入与数值参考

共用 `configs/hardware/u200_rtl_alignment_default.yaml` 与
`hardware/m10_u200/traces/alignment_cases.jsonl`。
Python产生manifest/每case stimulus，把同一候选、预算和DDR模式编码为SV plusargs。
配置同时驱动event adapter，RTL容量由源码默认参数核对，不缩小硬件。

Activation为int8 16；Router权重对E3/E4第一列为1，其余Expert为-1。
真实Router执行GEMV、Top-K和定点softmax，输出E3/E4、Q15 gates各16384。
各tile的G/U/D为16×16 identity，独立整数fixture oracle得到16个输出均为192。
Simulator是性能模型，不执行数值GEMV；数值比较对象明确为独立Python oracle。

未支持任意矩阵JSON文件、多候选或任意Top-K：超出本次fixture范围会明确报错，
不能用支持单候选的结果宣称所有allocation政策都已对齐。

## 4. 已运行矩阵与结果

| 激励 | RTL cycles | Sim cycles | 周期误差 | RTL观测等待 | Sim等待 | RTL speedup | Sim speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| demand_fast | 13923 | 14010 | +0.62% | 132 | 110 | 1.00000 | 1.00000 |
| hit_fast | 13810 | 13900 | +0.65% | 16 | 0 | 1.00818 | 1.00791 |
| second_hit_fast | 13810 | 13900 | +0.65% | 15 | 0 | 1.00818 | 1.00791 |
| wrong_fast | 13923 | 14010 | +0.62% | 132 | 110 | 1.00000 | 1.00000 |
| demand_gaps | 13934 | 14019 | +0.61% | 144 | 119 | 1.00000 | 1.00000 |
| hit_gaps | 13810 | 13900 | +0.65% | 17 | 0 | 1.00898 | 1.00856 |
| second_hit_gaps | 13810 | 13900 | +0.65% | 16 | 0 | 1.00898 | 1.00856 |
| wrong_gaps | 13934 | 14019 | +0.61% | 144 | 119 | 1.00000 | 1.00000 |

基线是同一DDR模式的零预算用例。
命中时RTL收益约0.82%（fast）或0.90%（gaps）；错误预测在本矩阵中不改变总周期，
但实际浪费768 bytes。长Router窗口足以把错误prefix加载完，不代表错误预测永远无害。

精确检查：truth/Q15 gates、actual speculative allocation、tile compute顺序、
五类字节、数值、OCM总量与银行容量、FIFO/table边界、MAC与DDR互斥、trace/telemetry字节守恒。
除满足容量约束外，本轮全部8case的OCM总峰值和逐bank峰值也恰好一致：
总峰值58或60 slots，每bank峰值14–16 slots。RTL demand FIFO峰值16、descriptor峰值17。
零预算为24576 demand bytes；命中为768 useful spec+23808 demand；
错误为768 wasted spec+24576 demand，取消字节为0。

## 5. 最终综合网表代表点

复用冻结的最终U200默认DCP导出的funcsim网表，不重新综合或改变参数。
testbench通过NETLIST宏适配packed struct字段端口，不修改生成网表。

- demand_fast：64 descriptors、32 compute completions、24576 bytes。
- wrong_gaps：66 descriptors、32 compute completions、25344 bytes，其中768 wasted spec。

两个点的全部共同端口事件（含cycle、descriptor身份、slot、bytes）和PORT_METRICS
与XSim RTL完全一致，自检均TEST_PASS、正常finish、无Fatal/Error。
网表不探测被优化的内部信号；完整GU/DOWN/nonlinear时间线来自RTL观测。
这不等于任意输入形式等价或SDF timing simulation。

网表compile约3.58秒、elaborate约102.40秒，展开峰值RSS约5.46GiB；
两个运行分别约114.27/114.36秒，运行峰值RSS约1.17GiB。
未记录全量波形。GNU time的RSS是阶段最大子进程值，不是并发进程树总RSS。
每阶段20分钟超时、每进程24GiB地址空间上限。

## 6. 时序偏差与指标口径

原总周期相对容差10%、等待绝对容差4cycles、speedup绝对容差0.005完全不变。
总周期误差+0.61%～+0.65%，speedup差约0.00027～0.00042；
等待误差15～25cycles，故 `timing_passed=false`。

Router控制采用K*E+42，resolver demand释放采用64entries扫描+4drain，
来自结构性解释，不读取RTL完成时间再回放。
DMA每burst解析计setup，bank-specific modulo停顿仍以聚合有效rate近似。

已知差异：

- adapter沿用每tile3-cycle retirement/combine近似；RTL只在Expert尾tile执行combine。
  因而不能用接近的总周期判断每阶段控制时延已正确建模。
- RTL的现有weight_wait观测使用registered gu_found/readiness scoreboard；
  它排除running/issue_pending/combine，但仍可能包含快照更新等待。
  它不是独立扫描live residency后的“纯payload缺失”证明。
- pair admission、token retirement、selection refresh与bank停顿相位未逐周期复现。

本轮把偏差如实保留，没有用改变RTL、调整服务率拟合总周期或放宽容差取得PASS。
若论文需要纯memory-stall分解，下一步应补独立live-residency观测，再区分memory/control idle；
不要直接把现有15–25cycles误差全部解释成带宽误差。

## 7. 复现

准备Vivado2023.2/XSim在PATH及Python项目依赖。在正常终端从仓库根运行：

```bash
# 8点RTL/Simulator矩阵
make -C hardware/m10_u200 alignment-default

# 同时跑两个最终网表代表点
PYTHONPATH=src python hardware/m10_u200/scripts/compare_u200_default_alignment.py \
  --config configs/hardware/u200_rtl_alignment_default.yaml \
  --trace hardware/m10_u200/traces/alignment_cases.jsonl \
  --netlist hardware/m10_u200/reports/post_synth_20261004/u200_moe_top_funcsim.v \
  --output results/m10_default_alignment
```

网表旁的 `netlist_manifest.json` 保存完整几何、netlist/DCP SHA256与全部26个RTL hash；
入口拒绝旧网表、缺失几何、不同源码或不同硬件配置。本次来源核验实际通过。
当前受限agent沙箱会阻止XSim启动；本轮获准在沙箱外运行，未屏蔽异常。
历史小配置仍用 `make ... alignment`，不能把完整YAML传给旧固定小TB入口。

默认退出码要求语义与指定网表PASS，性能偏差单独记录；
追加 `--strict-timing` 时本轮等待容差会导致失败，不应把默认exit0称为全性能PASS。

## 8. 文件与输出

主要修改：

- `src/edge_moe_sim/rtl_alignment.py`：独立M10 adapter的bank容量与demand释放事件。
- `configs/hardware/u200_rtl_alignment_default.yaml`：完整默认几何与原容差。
- `hardware/m10_u200/tb/tb_u200_post_synth.sv`：共享runtime激励、端口事件、RTL阶段/资源观察。
- `hardware/m10_u200/scripts/compare_u200_default_alignment.py`：XSim矩阵、网表代表点与自动比较。
- `hardware/m10_u200/scripts/compare_u200_alignment.py`：复用fixture验证/数值生成；保护旧小配置入口。
- `tests/test_u200_default_alignment.py`：14个新用例，包含stale provenance拒绝。
- Makefile、README、本中文文档及冻结网表来源metadata。

产物 `results/m10_default_alignment/`：
input_manifest.json、comparison.csv/json、REPORT.md；
每case stimulus、events.csv、RTL/Simulator timeline.csv与summary.json；
compile/elaborate/runtime日志、实际命令与资源测量；
netlist代表点端口events/metrics comparison、pytest及RTL/golden/lint回归日志。
没有清理或覆盖以前的小配置实验报告。

## 9. 回归与排除项

Python：221 tests passed；新增适配检查全部通过。
既有RTL：52个运行点通过；5个allocation golden与6个dense数值golden通过；
lint命令成功但保留原有advisory，不称为零warning。
运行环境的现有Python是3.9.19，本轮没有安装/更换解释器；
项目声明的Python3.11+环境未在本轮单独验证。XSim实际为2023.2。

全部冻结设计RTL和XDC SHA256保持不变。
新矩阵不含flush/error、多epoch压力、多候选及其他allocator政策；
它们的既有RTL回归通过，不代表它们已完成Simulator/综合网表对齐。
未跑PnR/SDF/CDC/上板，200MHz尚未闭合（WNS=-0.973ns）。
