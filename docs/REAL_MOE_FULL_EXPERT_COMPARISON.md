# 真实模型：完整 Expert 目标预取补充比较

## 口径

补跑主模型现有 `full_expert_prefetch`，所有输入沿用首轮 `real_moe_pilot`：三种模型、四个连续 MoE 层、INT8 假设、合成真实路由和候选、requested 8/16 MiB OCM、32/512 ops/cycle。

不改 RTL，不改主模拟器的任何调度算法或资源语义。补跑 24 个 full-target 点，另实际重新运行 12 个 no_prediction 基线，要求其周期与历史结果逐一一致。

**full_expert_prefetch 是“申请整个候选 Expert 的 GU/DOWN tiles”策略，不是保证整 Expert 预加载成功。** 现有代码在规划阶段受 speculative payload budget 截断，随后又受 slot、descriptor/FIFO credits、需求优先及可用时间限制。不能把这个标签等同于 all-or-nothing 的完整 Expert cache admission。

本次增加独立诊断：按照每个 target layer/Expert 的所有 GU 与 DOWN tile 唯一键，检查其 speculative transfer completion（含配置的 payload visibility 延迟）是否不晚于 target Router resolve。`experts_all_spec_transfers_completed_by_router` 是完整传输完成计数，不声称已证明所有 tile 同时驻留；若发生驱逐，传输完成也不等于 resident。

## 两个完整 Expert 能否放入 OCM

| 模型 | 两 Expert 实际 INT8 byte | 两 Expert 固定 slot 占用 | 加两个保留 demand slot 的最低容量 |
| --- | ---: | ---: | ---: |
| OLMoE | 12 MiB | 16 MiB | 17 MiB |
| Mixtral-8×7B | 336 MiB | 448 MiB | 450 MiB |
| Mixtral-8×22B | 576 MiB | 768 MiB | 771 MiB |

主模型支持按实际 byte 或固定 slot 计容量；**本轮研究配置选择了固定 slot**，不是模拟器所有实验都如此。GU/DOWN 分别占一个相同尺寸 slot，DOWN 的有效 payload 较小，因此有 padding。

按实际 byte，16 MiB 能装下两个 OLMoE Expert；按此次 fixed-slot + reserved demand 规则则不能。两个 Mixtral 模型在 8/16 MiB 下，无论实际 byte 还是 slot 都装不下两个完整 Expert。上表只是存储下限，不是充分条件，仍需足够的 descriptor/FIFO credits 与预取带宽窗口。

此外，本轮保持 `wait_for_previous_layer_weights=True`：收到预测不代表立即发 DMA。前一层权重必须读取完成才能释放下一层预取；在有限 buffer 的 tile streaming 场景里，这个释放可能很晚。不能通过名称“提前一层预测”推断存在完整一层长的 DMA 窗口。

## 输出和复现

```bash
python examples/real_moe_full_expert.py
pytest -o addopts= -q
```

输出目录 `results/real_moe_full_expert/`：

- `runs.csv`：24 个 full-target 点、容量诊断和相对三个基线的 speedup；
- `rerun_baselines.csv`：12 个新运行的 no_prediction 基线；
- `per_layer.csv`：96 行 full-target 层级时间；
- `summary.json`：当前代码/配置 hashes、历史 CSV/summary hashes 和验收结果。

相对 no_prediction 的 speedup 为 `no_prediction_cycles/full_target_cycles`。
`speedup_uniform_prefix_vs_full_expert` 为 `full_target_cycles/uniform_prefix_cycles`，因此大于 1 才表示 uniform 更快；ranked 同理。

脚本核对历史核心与配置 hashes 和相同 settings；实验 wrapper 新增诊断但未修改执行规则。历史前缀数据保留原始 provenance，本轮不重写其 summary，也不伪装为重新执行过所有前缀点。

## 本轮实际结果

36 次主模型执行全部完成：24 个 full-target 点，以及 12 个重新运行且与历史周期完全一致的无预测基线。输出为 24 行 full-target CSV、12 行基线 CSV、96 行层级 timing；当前补跑源码 hashes、容量边界检查通过，全库 351 个测试通过。

24 个点中，Router resolve 前完整传完所有 speculative GU/DOWN tiles 的 Expert 计数为 **0**。因此没有把任何点当作成功整 Expert 预加载案例。

full-target 相对 no_prediction 的 speedup 为 0.9981047×–1.0039152×。与同条件 uniform/ranked 比较，22 个点总周期完全相同；另外两个点是 OLMoE、16 MiB、512 ops/cycle 的 perfect/partial 条件，full-target 为 6,332,561 cycles，前缀为 6,332,564 cycles，仅差 3 cycles，不能据此声称显著策略优势。

结论是：**在当前容量、有限 credits 和释放窗口下，完整目标策略没有实现完整 Expert 预加载，而是退化成资源受限的部分预取。** 这些结果不能代表无限或足够容量下的完整 Expert cache，也不能作为公平的 all-or-nothing baseline 结论。

## 后续比较的边界

如果在这些限制下没有完整 Expert 传完，则实验衡量的是“以整 Expert 为目标的受限预取策略”，不能用来声称成功完整驻留 Expert 后的性能。

若后续要比较真正 all-or-nothing full Expert 策略，应明确独立的 admission 规则，并在不具备完整容量时记为不可行/不发预取；若扩大容量/credits 或改变预取释放时机，则必须同时重跑所有策略与无预测基线，不能混用这轮不同约束下的时间。
