# Switch 弱 baseline 与部分 tile 对照

## 本次范围

只增加按需加载 baseline；不更改预测质量、提前层数、burst 仲裁或预取算法。
继续使用未剪枝两矩阵 ReLU Switch-base-8/16，双候选覆盖率 84.4%，
50 个独立单 token 六层序列，以及相同 8/16/24 MiB OCM 和 32/512 ops/cycle 配置。
硬件、INT8 和非线性吞吐均为配置假设，不是 FLAME 实测复现。

## 六种策略

| 输出 policy | 类别 | 行为 |
|---|---|---|
| demand_full_expert | 弱 baseline | Router 后按需加载当前完整 Expert 的全部 Up/Down tiles；所有权重 ready 后才开始计算；不提前加载下一 Expert |
| demand_tile_serial | 部分权重驻留 baseline | 每次只加载当前 Up tile，计算和 ReLU 完成后才加载当前 Down tile；计算结束才推进下一个 tile，无 DMA/计算重叠 |
| no_prediction | 强 baseline | 按需 tile streaming：当前 tile 计算时允许加载后续 tile，无预测 |
| full_expert_prefetch | 完整目标预取 | 上一 MoE 层预测就绪后尝试加载两候选的全部 Up/Down tiles，受 OCM 与 DMA 约束 |
| uniform_prefix | 部分 tile 预取 | 两候选分别预取 2 个 Up tiles |
| ranked_prefix | 部分 tile 预取 | 排名 1/2 的候选分别预取 2/1 个 Up tiles |

“部分权重驻留 baseline”是**不预测的 tile 按需加载**；部分 tile 预取 baseline 则是
uniform/ranked 两行。它们不可混为一谈，前者用于分离 tile granularity/overlap，
后者用于分离 predictor 与 tile budget allocation。

完整按需 baseline 仍按相同 24 个 F-axis groups 进行计算，使用相同每 tile startup，
不会偷偷改成一次大 GEMM 来减少控制成本。其 DMA 仍可利用相同多通道硬件并行，
但不与 Expert 计算重叠。OCM 必须容纳一个完整 Expert，否则明确报错，不降级成 tile。

## 接口与限制

主模型通过 `AdvancedExecutionConfig.demand_loading` 选择：
`tile_streaming`（兼容默认）、`tile_serial`、`full_expert_load`。
弱 baseline 不启用 prediction，要求顺序 Expert、startup_gu_tiles=1、byte OCM 和
无限 request credits；当前不支持 RTL fixed slots/有限描述符模式。非法组合报错。
后续严格非流水预取对照允许 `full_expert_load` 配合完整目标 prediction，
仍保留全权重 ready 的计算门槛；`tile_serial` 仍只支持无 prediction。
该新增 case 见 `SWITCH_NONPIPELINE_PREFETCH.md`，历史结果不变。
旧 AdvancedSimulator 不会静默忽略该字段，非默认值会明确提示使用主模型。

## 复现与输出

```bash
python examples/switch_prefetch_comparison.py --reference results/switch_prefetch_flexible
pytest -q
```

默认保存至 `results/switch_prefetch_baselines/`，不覆盖历史结果。
`comparison.csv` 提供四个不同分母：

- `speedup_vs_no_prediction`：相对强 demand tile streaming。
- `speedup_vs_full_expert`：相对完整 Expert **预取**，不是弱 demand baseline。
- `speedup_vs_demand_full_expert`：相对完整 Expert 按需加载后计算。
- `speedup_vs_demand_tile_serial`：相对 tile 按需串行。

每种速度比均为 50 个样本总 cycles 的比值；不能从弱 baseline 的大收益直接推断
predictor 收益。应先比较弱 baseline 与 no_prediction，再比较 no_prediction 与预取。

新增手算测试覆盖两个弱模式、多个 Expert、两层执行、DMA/compute 不重叠、
完整权重 ready 后才发射、byte conservation、容量拒绝和确定性。

## 运行结果

本轮完成 3600 次模拟、72 个聚合点。此前四种策略的 48 个配置点总 cycles
与 `switch_prefetch_flexible` 完全相同，确认没有同时改动其执行规则。

Switch-base-8、16 MiB，50 组六层序列：

| 策略 | 32 ops/cycle 总 cycles | 相对完整按需 baseline | 512 ops/cycle 总 cycles | 相对完整按需 baseline |
|---|---:|---:|---:|---:|
| 完整 Expert 按需加载 | 110997600 | 1.000000× | 27945600 | 1.000000× |
| tile 按需串行 | 110997600 | 1.000000× | 27945600 | 1.000000× |
| demand tile streaming | 89227200 | 1.243988× | 22360800 | 1.249759× |
| 完整目标预取 | 88901416 | 1.248547× | 22477136 | 1.243290× |
| Uniform 部分 tile 预取 | 88901416 | 1.248547× | 23079264 | 1.210853× |
| Ranked 部分 tile 预取 | 88901416 | 1.248547× | 22695504 | 1.231328× |

两个弱 baseline 周期相同不是模式未生效：本实验单 token Top-1，两个模式使用
相同 DMA tile 请求、setup 数量与计算工作量，且均不重叠，改变加载时刻不改变
这些时长的总和。空间行为却不同：完整加载需同时驻留 4.5 MiB；
tile 串行只需驻留当前一张矩阵的 96 KiB tile。

结论：相对弱 baseline 的约 1.24–1.25×，主要由已有 demand streaming 的重叠
取得，而不是 predictor。32 ops/cycle 时预取在其上进一步达到 1.003665×；
512 ops/cycle 时本轮预取策略对强 baseline 仍无正收益。
Switch-base-16 呈相同趋势，精确值见 CSV，不能将相对弱 baseline 的收益直接
称作预测收益，也不能据此声称复现 FLAME。

验证：370 个测试通过。新增测试也验证完整按需与 tile 串行峰值 OCM 不同。
生成两组 PNG/PDF（speedup 和等待 cycles），完整保留 CSV、激励、预测时序与 hash。
