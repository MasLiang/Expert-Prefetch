# M10.1 真实 tile 数据通路 RTL

## 可选 Expert 类型

顶层 `SWITCH_RELU=1'b1` 选择 Switch 两矩阵 ReLU；默认 `1'b0` 保持 SwiGLU。
这是综合时开关，第一矩阵大小和计算行数均随模式调整，跨层预取使用同一比例。
详见 `../../docs/M101_SWITCH_RELU_RTL.md`。
`make -C hardware/m10_u200 switch-test` 运行数值/多层回归，
`make -C hardware/m10_u200 switch-four-cases` 运行主模型与 XSim 的四 case 对照。

## 四 case 性能对照

`make -C hardware/m10_u200 four-cases`（从仓库根目录运行，需 Vivado 2023.2
工具可用）运行主 UnifiedSequenceSimulator 与真实数值 RTL 的 24 点 XSim 对照。
结果在 `results/m10_four_cases_promotion_fixed/`。新增 `WAIT_FULL_EXPERT=1`
只用于整 Expert 启动基线；默认仍为 GU tile 就绪启动。方法和限制见
`../../docs/M101_PROMOTION_FOUR_CASES.md`。此实验不是 FLAME 或 U200 板级性能复现。

U200 新功能顶层是 `rtl/u200_moe_top.sv`：真实共享 Router/Expert MAC、四路物理
DDR AXI、外部 predictor。完整接口与量化规则见
`../../docs/M101_U200_ROUTER_DDR_INTERFACE.md`。
`rtl/tile_prefetch_predictive_axi_top.sv` 保留外部 truth / 单 AXI 默认兼容模式。

## 接口顺序

1. 在 prediction begin handshake 提供 epoch、sequence、layer、candidate count、
   预算、地址几何、activation 和 requant shift，配置在该拍锁存。
2. 按 rank 顺序提交 candidate；last 必须与 count 一致。Expert ID 在
   `[0,NUM_EXPERTS)`，confidence 是非零 unsigned Q12。
3. 新顶层先由 host 顺序加载一层 Router 权重，再提交 prediction begin。
   Router 执行 int8 GEMV、Top-K、定点 softmax，内部产生 Q15 truth；predictor 不改变 truth。
   旧兼容模式仍由外部提交 begin 和 Top-K entries。NoPrefetch 使用合法 transaction 与零预算。
4. 子系统自行 commit/cancel、生成缺失 demand tile、读取权重、计算和 combine。
   result_valid 为一个 cycle，result_y 在下个 layer begin 前保持。
5. AXI AR/R 必须遵守 valid-ready。transaction_error 表示失败；host soft-flush 并等待
   旧 DMA/ownership 退休后重试。失败结果不会 valid。

同一时间处理一个 layer transaction；所有接口处于一个 clk domain。
USE_SEQUENCE_CONTROL=1 接入六层 permission controller，默认由外部 host 控制 sequence。
新顶层已实现 Router GEMV，predictor 仍是外部 transaction producer。

## 权重布局与单位

默认 H=16、F_TILE=16。GU slot 存放 output-major Gate、Up，各为 F_TILE 行、H 列。
DOWN slot 是 H 行、F_TILE 列。每行连续 int8，beat 最低 byte 是 lane 0。
默认 GU=512 byte、DOWN=256 byte、logical tile=768 byte。

physical_tile_bytes 指 DOWN bytes，必须为 H*F_TILE，真实模式 GU_BYTE_RATIO=2。
GU tile 地址为 `gu_base + expert*expert_stride + t*2*physical_tile_bytes`；
DOWN 为 `down_base + expert*expert_stride + t*physical_tile_bytes`。host 提供覆盖完整
Expert 且不交叠的 stride；新四 bank 模式地址/stride/tile bytes 对齐 64 byte，
host 按 16-byte stripe 散布到四 bank；旧模式对齐 16 byte。

OCM payload 是同步读 byte-lane RAM，不复位 RAM 内容；ownership 与 payload-complete
metadata 控制可读性，默认预留两个 slot 给 demand。Q4 SiLU LUT 是量化近似；完整
数值规则见 `../../docs/M101_IMPLEMENTATION_STATUS.md`。

## 验证

首次 RTL/性能 simulator 对齐实验：

```bash
make -C hardware/m10_u200 alignment
```

说明与已发现的性能偏差见 `../../docs/M101_RTL_SIMULATOR_ALIGNMENT.md`。
8 个小模型激励的语义检查通过，但权重等待/speedup 容差未通过，不宣称完全性能对齐。
当前 Python 回归为 243 个测试；RTL 回归包含 59 个运行点。

### 唯一执行规则：GU 权重就绪即可启动

GU payload 已提交且驻留、同一 tile 的
GU/DOWN descriptor 与 slot ownership 已建立后，就可以启动 GU；不再等待 DOWN DMA 完成。
DOWN 权重可在 GU/非线性期间加载，但 DOWN GEMM 必须等自己的 payload 就绪。
GU_DONE 后独立释放 GU slot；DOWN_DONE 后释放 DOWN slot，不放宽容量限制。
释放与有效位退休均校验 epoch/generation，旧 tile 的最终退休不能清除复用的新 owner。
旧 `GU_EARLY_START` 参数与 pair-ready 启动分支已删除，所有自动执行入口采用相同规则。

```bash
make -C hardware/m10_u200 gu-early-test
```

该命令运行八个完整配置 XSim 自检，并与冻结的旧 RTL 结果比较，不运行或修改
主 Simulator。按需/错误预测减少 14–18 cycles，命中用例不变；数值和字节语义全部通过。
该轮冻结结果见 `../../results/m10_gu_only_start/REPORT.md`；当前重复运行上述命令默认输出到
`../../results/m10_rtl_current/`，不覆盖优化基线。详细规则见
[GU 提前启动说明](../../docs/M101_GU_EARLY_START.md)。
剩余差异及 RTL 优化优先级见
[后续对齐检查](../../docs/M101_REMAINING_ALIGNMENT.md)。

### 第一轮控制优化：跳过已就绪阶段的等待状态

请求接受时 GU 已 committed/resident，控制器在该边沿捕获 descriptor/generation 并产生
注册 GU 启动脉冲；非线性完成时 DOWN 已就绪，同样直接产生注册 DOWN 启动脉冲。
不就绪时仍走 WAIT_GU/WAIT_DOWN；clear 仍优先。第一轮当时仍采用 pair 退休；
当前阶段释放规则见下文。

```bash
make -C hardware/m10_u200 handoff-test
```

该命令以冻结 `results/m10_gu_only_start/` 为优化前基线，运行八点完整配置 XSim，
要求每点减少周期并保持 GU/DOWN 执行时长、输出与字节语义。实测减少 63–64 cycles/layer，
该轮冻结结果在 `../../results/m10_compute_handoff/`，重复运行输出到
`../../results/m10_compute_handoff_current/`，避免覆盖后续优化的比较基线。说明见
[控制交接优化](../../docs/M101_COMPUTE_HANDOFF_OPTIMIZATION.md)。
该轮未改主 Simulator，未重新综合或宣称 200 MHz 闭合；下一节记录后续 scheduler 优化。

### 第二轮控制优化：下一 prefix 注册搜索

scheduler 在当前 Expert tile 执行期间搜索其下一 prefix，只缓存搜索结果，不提前发射。
退休后仍检查当前 prefix、有效 slot、GU ready 和捕获冲突；DMA 完成、capture、route 或
clear 导致的 stale 搜索仍需要刷新。稳定请求寄存级和单 engine 互斥不变。

```bash
make -C hardware/m10_u200 scheduler-test
```

以冻结的 `results/m10_compute_handoff/` 比较八个完整配置 XSim 用例，每层再减少 30 cycles。
冻结结果在 `../../results/m10_scheduler_lookahead/`，重复运行输出到
`../../results/m10_scheduler_lookahead_current/`，说明见
[注册搜索 lookahead](../../docs/M101_SCHEDULER_LOOKAHEAD_OPTIMIZATION.md)。
### 第三轮资源优化：GU 阶段独立释放

GU_DONE 后回收 GU 的 OCM metadata、payload binding 和 scheduler 有效位；
DOWN 仍保留到 DOWN_DONE。请求表继续等待 FIFO/DMA/completion 引用退休，
不随 GU compute 完成直接回收。四 slot 压力测试覆盖旧 GU slot 被下一 DOWN 复用、
迟到旧 generation 释放及旧 tile 最终退休不破坏新 payload。

```bash
make -C hardware/m10_u200 stage-release-test
```

八点完整配置 XSim 均通过，每点 32 次 GU 与 32 次 DOWN 独立释放。
相对第二轮默认配置总 cycles 不变；容量未卡住关键路径时不应声称一定提速。
结果在 `../../results/m10_gu_stage_release/`，详细说明见
[GU 独立回收](../../docs/M101_GU_STAGE_RELEASE.md)。
### 第四轮计算优化：共享 MAC 的 chunk 流水

Router/GU/DOWN 共用同一 16-lane INT8 MAC，每周期发一个 16-byte chunk。
RAM 读取、乘法、四级归约、累加和写回相互重叠；row/first/last 标签确保跨 row
累加与最后写回正确，stage-done 只在末次写回后产生。clear 清空全部有效标签。
本轮同时将 backend 重构成每个 always_ff 仅更新一个字段。

```bash
make -C hardware/m10_u200 mac-pipeline-test
make -C hardware/m10_u200 synth-default
```

完整默认八点 XSim 全部通过，GU/DOWN 每 tile 从 256/128 降为 40/24 cycles。
整层从 13717–13822 降到 2987–3083 cycles；数值、truth、allocation 和字节分类守恒。
冻结证据在 `../../results/m10_mac_pipeline/`，重跑输出到单独的
`../../results/m10_mac_pipeline_current/`。详见
[MAC chunk 流水说明](../../docs/M101_MAC_CHUNK_PIPELINE.md)。
此前 `gu-early-test` / `handoff-test` / `scheduler-test` / `stage-release-test` 的
默认检查是“只改控制、计算时长不变”，不适用于当前计算流水；它们会明确失败，
不应删除时长断言来强行通过。当前使用 `mac-pipeline-test` 检查新的解析服务公式。
旧配置里的 `ops_per_cycle=4` 是历史适配层参数，不代表当前流水有效服务率。
完整默认 U200 OOC 综合通过：87187 LUT、53585 FF、130 RAMB36、148 DSP。
5 ns 综合估计 WNS 为 −0.866 ns，200 MHz 尚未闭合；最差路径为 resolver→descriptor/OCM
控制链。1219 个输入和 2106 个输出缺少板级 IO delay，不能据此宣称板级 timing PASS。
DMA 紧迫度/lookahead 策略、真实多 layer context、布线后 200 MHz 时序仍待验证/实现。

当前共同激励入口：`make -C hardware/m10_u200 alignment-main`。
它实际运行 `UnifiedSequenceSimulator` 与多层 XSim，而非旧适配层。
当前四项执行规则和验收见 [执行规则统一](../../docs/M101_EXECUTION_RULE_ALIGNMENT.md)，
首次诊断见 [历史共同激励说明](../../docs/M101_UNIFIED_RTL_ALIGNMENT.md)。
当前结果是诊断完成，**不是完整资源/周期对齐通过**。
后续有限 descriptor/FIFO、burst 仲裁与控制时间验证使用
`make -C hardware/m10_u200 alignment-control`，敏感性扫描使用
`make -C hardware/m10_u200 control-sensitivity`。
九组结果、参数来源与验收边界见 [控制资源对齐报告](../../docs/M101_CONTROL_RESOURCE_ALIGNMENT.md)。
最新 RTL 取消四 DDR AR 发起和末 payload 完成的额外空拍；共同激励入口为
`make -C hardware/m10_u200 alignment-interface`。
九组结果与晚预测边界反例见 [接口时间与 RTL 优化](../../docs/M101_INTERFACE_TIMING_OPTIMIZATION.md)。
新增 `make -C hardware/m10_u200 boundary-validation`：30 组端到端边界矩阵、3 组
就绪边界和2组commit观测。定位与不可忽略的六个截止点差异见
[边界等待验证](../../docs/M101_BOUNDARY_WAIT_VALIDATION.md)。
最新主模型补齐入口为 `make -C hardware/m10_u200 descriptor-boundary`，使用独立的
`u200_descriptor_alignment.yaml`，结果保存到 `results/m10_descriptor_boundary/`。
逐 descriptor commit/cancel、payload 可见、combine 和 allocator 容量修正见
[逐 descriptor 对齐报告](../../docs/M101_DESCRIPTOR_VISIBILITY_ALIGNMENT.md)。
历史结果不覆盖；本轮没有修改设计 RTL/XDC。
进一步的层间 ingress/控制资源退休及 FIFO credit/occupancy 拆分使用
`make -C hardware/m10_u200 ingress-boundary`，结果在 `results/m10_ingress_boundary/`。
配置、指标语义及 DDR 仿真支持见
[层间退休与 FIFO 统计](../../docs/M101_INGRESS_FIFO_ACCOUNTING.md)。
新增可选四路 DDR4 controller + RDIMM model + 异步 AXI bridge 验证后端：
`make -C hardware/m10_u200 ddr4-validation`，结果在 `results/m10_ddr4_controller/`。
这是验证环境接入，交付 top 仍为四路 AXI，不代表已连接 U200 shell 或完成板级部署。
配置与统计边界见 [DDR4 控制器验证](../../docs/M101_DDR4_CONTROLLER_VALIDATION.md)。

下列 pair-ready 对齐结果属于历史版本。`M10AlignmentSimulator` 对当前 RTL 已失效，默认构造即拒绝；
只有历史回归可显式使用 `allow_historical=True`。旧 M10 适配层仍采用 pair-ready，
`alignment` 和 `alignment-default` 会明确拒绝运行，防止误称旧适配层与当前 RTL 已经对齐。
旧 DCP/网表也不能作为新 RTL 的验证证据；本轮未重新综合或验证 200 MHz 时序。

历史完整默认配置的共享激励对齐（XSim 2023.2，pair-ready）：

```bash
make -C hardware/m10_u200 alignment-default
```

已实际运行8个完整配置RTL/Simulator用例及2个最终综合网表代表点：
语义检查全部通过，网表/RTL端口事件与指标完全一致；总周期与speedup通过原容差，
但等待时间相差15–25 cycles，完整性能容差仍未通过。
配置为E64、16 tiles/expert、128个4096-byte OCM slots，不是上述历史小配置。
说明、可复现网表命令及范围限制见
[完整默认配置对齐文档](../../docs/M101_DEFAULT_RTL_SIMULATOR_ALIGNMENT.md)。

等待指标已拆分：`memory_wait_cycles` 直接检查下一prefix的GU/DOWN物理payload就绪；
`ready_control_wait_cycles` 统计权重已就绪但尚未发射的控制等待。
保留旧 `weight_wait_cycles` 和旧容差，并输出旧窗口内的内存/控制分项。
新窗口包含issue_pending，排除running和combine，两项之和是expert发射空闲，不是整个layer idle。
复现及实际分项结果见 [等待指标拆分](../../docs/M101_WAIT_METRIC_SPLIT.md)：

```bash
PYTHONPATH=src python hardware/m10_u200/scripts/compare_u200_default_alignment.py \
  --output results/m10_wait_split_alignment
```

8点语义通过；命中用例内存等待为0，demand/wrong为115/126 cycles，
模型为110/119 cycles，仍超出原4-cycle容差。本次未重跑网表，不覆盖上一轮网表证据。

后续已统一请求接受/计算启动事件边界，见
[功能边界契约v1](../../docs/M101_FUNCTIONAL_BOUNDARY_CONTRACT.md)。
同一分析器输出每tile、每stage timing CSV；旧95/256控制等待不再直接比较。
统一ready→GU为RTL127–129/模型256cycles，模型accept→GU仍标记collapsed/unmodeled。
复现时将输出目录指定为`results/m10_boundary_alignment`，不覆盖历史结果。

```bash
make -C hardware/m10_u200 rtl-test golden lint
make -C hardware/m10_u200 synth-ooc
python hardware/m10_u200/scripts/validate_design.py
```

OOC 综合固定为 4 candidates、4 tiles/Expert、8 个 512-byte slot、16 descriptor entries。
历史缩小配置结果：WNS=+0.371 ns，TNS=0；26,271 LUT、21,708 FF、64 DSP。
这些数字不是完整默认配置的时序结论。默认配置的审计及本轮 RTL 优化结果见
[优化报告](reports/optimization_20261004/REPORT.md)。
RTL 52 个测试运行点和 Python 207 个测试通过；allocation/numeric golden 继续通过。
5ns（200 MHz）是检查目标，不是已测时钟。报告包含 utilization、timing、DRC、
前 20 条 critical paths 和 checkpoint。
数值 MAC 使用乘法/平衡归约/累加流水级；Q15 combine 分级完成，done 只在最终写回后
产生。增加流水级改变 cycle latency，不能把频率目标提升等同于端到端性能提升。
四路 DDR core-clock 接口已经接入，但 board DDR controller/XPFM、CDC、host/XRT 与
PnR 尚未接入；报告不构成部署验收。200 MHz 下四路 128-bit 的理想接口上限是
12.8 GB/s，不声明已经最大化 U200 DDR 带宽。
验证脚本同时运行 Python 回归，生成 reports/design_validation.json 与每项检查的完整日志。

## 2026-10-04 RTL 优化与接口说明

新增顺序多层数据通路：`u200_moe_top.LAYER_COUNT=1..8`，真实层输出经可配置
INT32→INT8 requant 后作为下一层 activation；每层重新加载 Router 权重，旧层资源
排空后自动清理再推进。`make -C hardware/m10_u200 multilayer-test` 运行专门 XSim 回归。
接口协议、测试与限制见 [顺序多层说明](../../docs/M101_MULTILAYER_CHAIN.md)。
后续已接入提前多个层的外部候选保存与跨层预取：完整候选立即参与资源仲裁，
不等待前一层真实权重完成；`layer_weights_done` 仅用于监测。
受共享OCM/credit约束，并为 demand 保留两个 descriptor；切层按epoch清理并保留未来层驻留。详见
[跨层预取接口与验证](../../docs/M101_CROSS_LAYER_PREFETCH.md)。
仍不包含Attention/residual、主Simulator多层对齐或新的200 MHz时序签核。

- Execution scheduler 使用平衡 slot 索引树和 readiness 快照，不再为每个 Expert
  扫描并选择完整 descriptor。快照刷新会增加控制等待周期；稳定 ready 集合的
  Router-order 优先级、最高匹配 slot 规则和 tile prefix 顺序不变。
- OCM 空槽和 occupancy 使用组合平衡树，不增加 admission 周期；所有权更新
  使用显式 next-state，包含完整 slot、epoch 和 generation 校验。
- Parser、descriptor manager、compute controller 和 int8 backend 新增同步 `clear`。
  独立例化必须显式连接；无需 abort 的实例接 `1'b0`。`rst_n` 仅用于全局复位，
  不再用 soft-flush 门控。flush 周期阻止新握手，并等待旧 DMA/quarantine 正常退休。
- FIFO、Gate/Up/Nonlinear payload 数组不异步清零，控制有效位仍复位。
- Speculative byte finalizer 对 `ENTRIES <= 8` 保留两拍统计发布延迟；大表增加
  分组快照级，发布延迟为三拍。表项在分类当拍退休，快照独立保存其贡献，避免重用丢数。
- Credit 在 winner selection 时预计算，admission 时一次扣费；不改变 allocation policy。

新增测试覆盖非二次幂容量、128-slot selection、ready-first 相邻完成竞争、同步
flush/restart、非法 slot/stale 标签，以及最大 byte 值、统计延迟和 generation 重用。
上述验证不等于综合后等价证明、任意矛盾命令穷举或板级时序签核。
