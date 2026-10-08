# 当前 RTL 与主模拟器的共同激励诊断

> 本文件记录首次诊断的历史配置和结果。四项执行规则已随后统一，当前入口与
> 验收范围请见 [执行规则统一](M101_EXECUTION_RULE_ALIGNMENT.md)。历史结果目录保留，
> 新结果写入 `results/m10_unified_rules_alignment/`，两轮不可混作同一配置。

## 旧模型失效

`M10AlignmentSimulator` 对当前 GU-ready、多层、跨层预取 RTL **已失效**。
它建模的是历史 pair-ready 执行规则，不得再用于当前性能对齐或论文结论。
默认构造会抛出明确异常；`allow_historical=True` 仅供旧回归复现，并不恢复当前兼容性。
原来的 `alignment` / `alignment-default` 与相关历史结果不是当前验收入口。

## 当前入口

```bash
make -C hardware/m10_u200 alignment-main
pytest -q
```

此命令实际重建并运行 Vivado 2023.2 XSim，以及主模型
`UnifiedSequenceSimulator`。不新增另一个性能模拟器，不修改主模型语义，
不把 RTL 观测周期、延迟或真实路由回灌为主模型输入。

共同配置是 `configs/hardware/u200_unified_alignment.yaml`。
固定验证夹具为 H=16、F=32、E=6、Top-2、INT8，每 Expert 两个 F=16 tile；
GU=512 byte、DOWN=256 byte。真实 Router 按预先定义的数值权重产生
第 l 层 Expert `(l%4+1,l%4+2)`，各 Q15 权重 16384。
RTL 还独立检查实际激活、非线性和最终数值输出；主模型只比较工作量及时序，
不具备数值运算正确性验证能力。

矩阵涵盖 3 层无预取、命中、全错，8 层命中，以及更大的 OCM。
四 DDR bank 每个 128 bit/cycle、关闭外加随机气泡；单共享 MAC。
主模型的 `total_aie_units=1` 在此仅表示一个逻辑计算资源，**不表示 U200 有 AIE**。
32 operation/cycle、8 startup cycle 是结构估计；DMA startup=0 是明确理想化。
200 MHz 只是 cycle 换算配置，不表示当前 RTL 已完成 200 MHz 时序收敛。
这些均不是 U200 板级实测、FLAME 复现或真实模型性能结论。

当前层预算为 0；未来层各有一个置信度 3000/4096 的候选，预算 1536 byte，
source 为 layer 0，并要求前一层真实权重取完后才放行。
主模型 predictor offset 使用事先声明的 `2*target_layer` cycle；RTL 经真实握手
接收事务，两者不是逐周期相等的 predictor parser。不得称为完全相同 ready time。
目前是有限固定夹具映射而非任意 JSONL 文件的 RTL transaction loader；
JSONL 是导出的主模型输入记录，RTL 由同配置生成的 generic 控制相同路由/候选夹具。

## 输出与验收解释

`results/m10_unified_alignment/` 包含源文件 hash、共同配置、输入 JSONL、
主模型完整结果/事件 trace、RTL 仿真日志、绝对周期阶段 CSV、分层 timing CSV、
跨层事件 CSV、比较 CSV 和 summary。

`rtl_functional_checks_passed` 表示数值、真实路由、每个 GU/DOWN tile 恰好执行一次、
RTL bench 的 OCM/依赖/retirement 断言通过；**不是性能对齐通过**。
`alignment_passed` 和 `resource_equivalence` 当前刻意为 false，因为以下差异仍存在：

- RTL 固定 512-byte slot，DOWN 也占一个 slot；主模型按实际 byte 占容量。
- RTL GU/DOWN 成对 admission 并保留两个 demand slot；主模型不是同一 admission 规则。
- RTL tile ready-first 可交错 Expert；主模型单并行 Expert 配置锁定整个 Expert。
- 未来请求的重试和 GU/DOWN 发出顺序不同。
- RTL parser、Router 权重装载、combine、层间 retirement 有实在控制延迟。
- 当前 DMA/非线性流水是解析估计，非逐 cycle 接口协议模型。

`delta_cycles = RTL 最后一层数值完成 - RTL 第一层开始 - 主模型 makespan`。
这不是单独的 memory stall 差值，含计算、控制、调度和层间空隙差异。
两侧时间原点统一为第一层开始；主模型末尾无用请求 drain 与推理完成分开。
`--require-alignment` 会明确返回失败，防止 CI 把诊断运行完成当作完整对齐。
下一步必须解决上述结构差异或显式量化误差，才能验收性能一致性。

## 本次实际结果

Vivado 2023.2 行为级 XSim 的五个 case 全部正常 `$finish`，每个恰好一个 PASS。
两侧真实 Expert 集合及所有 GU/DOWN tile key 均检查；没有把缺失阶段静默过滤。

| Case | 主模型 cycle | RTL cycle | RTL−主模型 |
|---|---:|---:|---:|
| 三层无预取 | 1026 | 1410 | 384 |
| 三层命中 | 1010 | 1326 | 316 |
| 三层全错 | 1026 | 1410 | 384 |
| 八层命中 | 2680 | 3491 | 811 |
| 三层命中、更大 OCM | 1010 | 1326 | 316 |

该小夹具中，命中预取节省 RTL 84 cycle、主模型 16 cycle。
方向一致不表示节省量一致，更不能外推真实模型/板级性能。
增大 OCM 未改变这组夹具结果；全错也没有额外推理延迟，
这是本次约束与提前窗口下的观测，不是全错预取无害的普遍保证。

逐 tile `stage_comparison.csv` 中本次全部 GU/DOWN 服务时长差为 0：
GU 40 cycle、DOWN 24 cycle；因此总时长差不能归咎于这两个 MAC 阶段的服务公式。
还需比较启动、DMA/OCM admission、非线性/调度间隔和层间控制。
验证：254 项 Python 测试通过，`rtl-test lint golden` 返回成功，
RTL/golden 日志共 77 个 PASS；lint 仍输出已有的未使用信号等 warning，
本次没有宣称 warning 清零，也没有重跑综合或实现。
