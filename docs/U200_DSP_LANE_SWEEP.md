# 固定 U200-like 内存参数的 DSP MAC 扩宽扫描

## 范围与硬件假设

本实验使用当前 `UnifiedSequenceSimulator`，五个模型为 Switch-base-8、Switch-base-16、
OLMoE-1B-7B、Mixtral-8×7B、Mixtral-8×22B。使用上一轮保存的同一批每模型前 20 个样本，
没有重新生成 routing 或 predictor。仍是单 token、六个 MoE 层、INT8 权重/激活的结构性能模拟，
不是运行实际 checkpoint，也没有新增 RTL、综合、PnR 或板级测量。
Mixtral-8×22B 的完整 INT8 routed 权重超过 U200 外存容量；本轮六层结构子集不构成完整模型单卡部署证明。

本次只扩宽单逻辑 GEMM Engine 的假设吞吐，不增加 Expert 并行数。Router 与 Expert 的吞吐一起变化，
预测仍在上一层 source Router done 后按原延迟释放；没有上一层全部权重加载完毕的额外放行屏障。

| 项目 | 配置 |
|---|---|
| MAC lanes | 16、32、48、64、96、128、192、256、384、512 |
| 假设 MAC 速率 | 每 lane 每拍一次 INT8 MAC，即两次 operation |
| operation/cycle | 32、64、96、128、192、256、384、512、768、1024 |
| DSP 映射假设 | 每 lane 一个 DSP，不使用 INT8 packing 加倍性能假设 |
| 内存 | 四通道，每通道名义 16 byte/cycle，聚合名义 12.8 GB/s |
| 时钟 | 200 MHz，不假设宽阵列可以提升频率 |
| OCM | 固定 16 MiB，按实际 byte 容量计 |
| 总规模 | 五模型、10 档位、270 个配置点、5,400 次模拟 |

Switch 保留六种 whole/streaming、full/Uniform/Ranked 对照策略；OLMoE 保留七种原策略，
Mixtral 保留四种 streaming 策略。Mixtral 的 whole-residency 超出 OCM，因此没有该 baseline。
Switch 的 burst 仲裁及 physical payload multiplier=2、OLMoE/Mixtral 的 matrix-fragment DMA 均保持不变。
不同 backend 不直接横比绝对延迟或将 nominal bandwidth 误当成相同有用权重吞吐。

## DSP 数量不等于可部署能力

U200 器件的 DSP 数量配置为 6,840。
小参数 Switch OOC 报告中的 60 个 DSP 只用作控制资源参考，不表示完整模型控制逻辑或 shell 的最终消耗。
实验另外设置总 DSP 的 80% 为研究预算，给其他逻辑及物理实现留余量；这是配置假设，不是硬件保证。

```text
estimated_mac_dsp = lanes × dsp_per_mac_lane
estimated_total_dsp = estimated_mac_dsp + controller_dsp_reference
study_dsp_budget = floor(device_dsp_count × dsp_budget_fraction)
```

512 lanes 的估计是 512+60=572 个 DSP，占器件 8.36%，所有扫描档位均在研究 DSP 预算内。
这只能说明按当前映射假设数量够，不意味着宽阵列的 BRAM/URAM、互联、SLR 跨越、时序或功耗已经通过验收。

本轮明确假设 OCM banking/读口、激活广播和归约能力随阵列扩宽。
**当前 RTL 128-bit 权重读口每拍只能提供 16 个 INT8 权重，不能自动支持这些更宽 lanes。**
48/96/192/384 lanes 的矩阵行尾、tile 尾部利用率、排空和物理流水代价也未按新宽阵列建模。
当 lanes 大于 DOWN 的 F-tile 宽度 128 时，还需要跨输出行并行或等效映射才能充分利用阵列，
不能假设把当前单行归约树直接加宽就能达到同样吞吐。
模拟器中的 operation rate 是假设的有效吞吐，不是根据已有宽阵列综合结果校准的值。

## 指标与饱和点

输出分别保留两个分母：

```text
speedup_vs_16_lanes_same_policy = sum(16-lane same-policy cycles) / sum(current cycles)
speedup_vs_stream_demand = sum(same-lane streaming demand cycles) / sum(current cycles)
```

前者说明扩大阵列是否有用，后者说明同样硬件下预取是否有额外收益。
速度比以配对样本总周期计算，不平均各样本速度比。

`knees.csv` 给出各模型/策略的最快已测 lanes，以及距最快点周期数 1% 内的最小已测 lanes。
它是离散测试定义，不假设曲线单调，不是连续空间的理论拐点，也不是自动选出的最优 RTL 参数。
还输出相邻档位的速度比，用于检查继续增 DSP 是否只有微小回报。

工作量使用项目公式，Router、GU、DOWN 都计入整数 operations：

```text
ideal_payload_compute_lower_bound_cycles = ceil(payload_ops / configured_ops_per_cycle)
payload_fraction_lower_bound = ideal_payload_compute_lower_bound_cycles / makespan
```

这是不含 startup、switch、控制和非线性的**算术工作量占比下界**，
不是实际 AIE busy utilization、DSP utilization 或精确 stall 分解。
`memory_wait_cycles` 同样是 aggregate Expert starvation，不可直接当全局 idle 从 makespan 中相减。

## 验证与复现

所有模型/档位共享保存的激励，每次模拟检查 OCM 上限与 speculative byte conservation。
与上一轮带宽扫描重叠的 16/256 lanes 点逐样本检查周期、memory wait、OCM peak、
demand/speculative/useful/wasted bytes；预期 840 个重叠样本，不能将旧结果混入本次模拟计数。
源文件 hashes 与完整原配置快照也保存于 summary。

```bash
PYTHONPATH=src python examples/u200_dsp_lane_sweep.py --workers 12
pytest -q
```

配置：`configs/experiments/u200_dsp_lane_sweep.yaml`。
默认目录：`results/u200_dsp_lane_sweep_20261007/`。
已有 summary 时拒绝覆盖；再次运行应通过 `--output` 指定新目录。
`workers` 是主机进程数，不是硬件 MAC 或 Expert 并行数。

主要输出：

- `summary.json`：RUNNING/FAILED/COMPLETE 状态、计数、配置、模型和来源；
- `*_inputs.json`：实际使用的保存激励子集，覆盖率沿用上一轮子集的实际值；
- `dsp_budget.csv`：各档位的 DSP 研究预算，不是综合报告；
- `per_sample.csv`、`per_prediction.csv`：原始结果及预测诊断；
- `comparison.csv`：周期、byte、两个分母的 speedup、相邻档位收益和算术占比下界；
- `knees.csv`、`REPORT.md`：离散饱和点及实际运行结果；
- 四组 PNG/PDF：延迟、扩宽收益、预取收益、算术占比下界。

## 2026-10-07 实际扫描结果

已实际完成 5,400 次模拟、270 个配置点、18,000 条预测诊断。
840 个重叠样本的七项指标全部与前次带宽实验一致；源 hashes 和四组 PNG/PDF 已检查。
全量回归为 `570 passed, 14 warnings`，警告均为既有依赖的弃用提示。

以下是 **streaming 无预取** 的结果，分母为同模型的 16 lanes，不是非流水 baseline。

| 模型 | 16 lanes 平均 cycles | 512 lanes 平均 cycles | 扩宽提速 | 距最快周期数 1% 内的最小已测 lanes |
|---|---:|---:|---:|---:|
| Switch-base-8 | 1,795,167 | 910,197 | 1.9723× | 64 |
| Switch-base-16 | 1,797,519 | 910,317 | 1.9746× | 64 |
| OLMoE-1B-7B | 19,277,520 | 4,751,280 | 4.0573× | 192 |
| Mixtral-8×7B | 132,263,520 | 33,064,464 | 4.0002× | 64 |
| Mixtral-8×22B | 226,695,264 | 56,666,832 | 4.0005× | 64 |

尽管 lanes 增加 32 倍，总延迟只改善约 2–4 倍。五个 baseline 的最快已测点均为 512 lanes，
但 64–192 lanes 已经在其 1% 周期邻域内，因此“最快点”和“资源回报较合理的点”不能混为一谈。
这些结论仅适用于本轮固定内存与调度配置，不是 U200 的普适最佳 MAC 数。

预取策略有各自的饱和点：例如 Mixtral-8×7B 的 Uniform 需要 96 lanes 才进入该策略自身最快点的 1% 邻域，
其 Ranked 和无预取则为 64 lanes。完整 Expert 和非流水对照也各自保存在 `knees.csv`，
不把 streaming baseline 的拐点套给所有策略。

**扩宽并未持续提高预取的额外收益。** 在 512 lanes 时，相对同档位 streaming 无预取：

| 模型 | Uniform speedup | Ranked speedup |
|---|---:|---:|
| Switch-base-8 | 1.000645× | 1.000645× |
| Switch-base-16 | 1.000684× | 1.000684× |
| OLMoE-1B-7B | 0.929940× | 0.985453× |
| Mixtral-8×7B | 0.989804× | 0.996598× |
| Mixtral-8×22B | 0.993871× | 0.997021× |

较快的计算减少可供隐藏 DMA 的计算窗口。实际净收益还受非抢占 DMA、错误投机与 OCM 竞争影响，
不能只凭 DSP 数更多推断 prefetch speedup 更大。

若下一步验证宽 RTL，建议将 **64 lanes** 作为 Switch/Mixtral 的第一候选；
OLMoE 可测试 192 lanes，若采用二次幂阵列及较简单的行映射，可先验证 256 lanes。
OLMoE 的 256→512 lanes 在本矩阵中仅减少约 0.55% 周期，不足以单凭本次模拟支持直接扩至 512。
按预算假设，64/192/256 lanes 加控制参考分别为 124/252/316 个 DSP，数量不是当前主要瓶颈；
OCM 供数、输出行并行、实际映射和时序必须单独验收。本轮没有实施这些 RTL 更改。

实际结果目录：`results/u200_dsp_lane_sweep_20261007/`。
