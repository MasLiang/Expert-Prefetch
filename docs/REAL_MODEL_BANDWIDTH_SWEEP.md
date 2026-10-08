# 真实模型结构的带宽单因素扫描

## 要回答的问题

在算力、OCM、预测输入和调度算法不变时，提高权重外存带宽是否能：

- 缩短绝对延迟；
- 提高预取相对同带宽 streaming 无预取的额外收益；
- 找到从负收益转为正收益的区间，以及带宽继续提高后的收益上限。

这三个指标必须分开。HBM 的官方峰值不是实际有效带宽，HBM 容量也不等于本设计的 OCM 容量。

## 本次运行条件

五个模型仍是 Switch-base-8、Switch-base-16、OLMoE-1B-7B、Mixtral-8×7B、Mixtral-8×22B。
真实模型结构尺寸、INT8 假设、六个 MoE 层、单 token 的模拟范围均与上一轮一致。

从 `results/current_model_retest_20261007/` 的输入 JSON 读取每模型前 20 个保存样本。
没有根据新样本数重新生成随机 trace。所有 bandwidth/policy 点使用同一组 routing truth、候选与分数。
20 样本子集的实际 Top-2/candidate recall 单独写入 summary，不声称它仍精确等于原总体 84.4%。

| 维度 | 本次取值 |
|---|---|
| OCM | 16 MiB，按实际 byte 使用 |
| Switch 算力 | 32 ops/cycle |
| OLMoE/Mixtral 算力 | 32、512 ops/cycle，两条固定算力曲线 |
| DMA 通道 | 原来的 4 通道，映射不变 |
| 时钟 | 原来的 200 MHz |
| 每通道名义带宽 | 4、8、16、32、64、128、256、512、1024 byte/cycle |
| 对应聚合名义带宽 | 3.2、6.4、12.8、25.6、51.2、102.4、204.8、409.6、819.2 GB/s |
| DMA setup、burst、efficiency | 保持各模型原 profile，不随带宽改变 |
| 预测时序 | 原 source-router-done + predictor latency；无上一层全部权重屏障 |
| 总规模 | 7,560 次模拟、378 个配置点 |

Switch 沿用宽候选矩阵的六种策略：whole demand、stream demand、whole full prefetch、
stream full prefetch、stream Uniform、stream Ranked。候选列表覆盖全部 Expert，不等于排序准确率为 100%。
OCM、两 Expert 大小的预取 byte 预算和分配算法不变；physical payload multiplier=2 仍保留。

OLMoE 沿用原七种策略；Mixtral 沿用原四种 streaming 策略。
Mixtral 的完整 Expert 大于本 OCM，不能增加 whole-residency baseline；
`full_expert_prefetch` 在这里是 streaming 的完整 tile 集合目标，容量可能截断。
OLMoE/Mixtral 保持原 matrix-fragment 非抢占 DMA，不自动切换为 Switch 的 burst 仲裁模型。

## 如何理解 HBM 级扫描点

高带宽点只是外存服务速率的抽象敏感性参数。固定四通道、提高每通道 byte/cycle，
**不代表当前四通道 RTL、AXI 数据宽度或 PE 输入端口能达到该速率**。
本次没有建立 U280/V80 的真实 HBM pseudo-channel、互联拓扑、bank 冲突或布局布线模型。
因此只能回答“若可提供此有效服务能力，算法趋势如何”，不能宣称完成了 U280/V80 性能预测。

不同模型的原 backend 不同，不直接横比绝对延迟。也不能把 Switch 的 nominal bandwidth
忽略 payload multiplier 后直接视为有用逻辑权重吞吐。

## 指标与统计

同带宽预取 speedup 定义为：

```text
sum(streaming_demand_cycles) / sum(policy_cycles)
```

不是逐样本 speedup 的算术平均。整 Expert baseline 可行时另给一个独立分母。
绝对带宽收益使用同策略最低扫描带宽作为分母，不能与预取收益混淆。

`crossover.csv` 给出第一个正收益测试点、第一个超过 1% 收益的测试点、收益最高的测试点和最高带宽收益。
不做连续插值，不假定收益随带宽单调增加；小收益不自动等同于统计显著结果。

`memory_wait_cycles` 是各 Expert starvation 的聚合，不是全局 idle。
它可能与其他 Expert 计算重叠，不能直接从总 makespan 中减掉。

重叠的 Switch 8/16 byte/cycle 和 OLMoE/Mixtral 16 byte/cycle、16 MiB 点，会与上一轮同样本
逐指标比较：cycles、memory wait、OCM peak、demand/speculative/useful/wasted bytes 必须一致。
预期 1,080 个重叠样本。每次模拟还检查 OCM 上限与 speculative byte conservation。

## 运行与查看

```bash
PYTHONPATH=src python examples/real_model_bandwidth_sweep.py --workers 12
pytest -q
```

配置为 `configs/experiments/real_model_bandwidth_sweep.yaml`。
`workers` 是主机进程数，不是硬件 PE 数。可通过 `--samples` 改样本数，但不能超出保存的 trace。
结果目录已有 summary 时拒绝覆盖，请用 `--output` 选择新目录。

默认输出 `results/real_model_bandwidth_sweep_20261007/`：

- `summary.json`：RUNNING/FAILED/COMPLETE 状态、计数、模型、实际覆盖率、配置快照和源 hashes；
- `*_inputs.json`：使用的真实保存激励子集；
- `per_sample.csv`、`per_prediction.csv`：逐样本指标和预测诊断；
- `comparison.csv`：绝对周期/延迟、byte 指标与明确分母的 speedup；
- `crossover.csv`：离散阈值与最优点；
- `REPORT.md`：中文实际运行结果报告；
- 三组 PNG/PDF：绝对延迟、同带宽预取收益、aggregate starvation。

运行中每完成一个 20 样本配置就更新 `per_sample.csv` 与计数，不把部分结果标为 COMPLETE。
本轮未修改模拟器、Scheduler、RTL 或 XDC 的语义。

## 2026-10-07 实际运行结果

已完成 7,560 次模拟、378 个配置点。1,080 个重叠样本的七项指标与前次复测完全一致，
源文件 SHA-256 校验通过，三组 PNG/PDF 完整。全量回归为 `558 passed, 14 warnings`；
警告为既有依赖的弃用提示。

下面只展示 Uniform 的曲线代表指标；完整 Expert、Ranked、非流水 baseline 均保留在 CSV 和图中。
这里的“最佳”是本次九个测试点中的最大总体 speedup，不表示连续空间的全局最优或统计显著性。

| 模型 | ops/cycle | Uniform 最佳聚合名义带宽 GB/s | 最大预取 speedup | 819.2 GB/s 时预取 speedup |
|---|---:|---:|---:|---:|
| Switch-base-8 | 32 | 12.8 | 1.008702× | 1.000394× |
| Switch-base-16 | 32 | 12.8 | 1.008437× | 1.000394× |
| OLMoE-1B-7B | 32 | 12.8 | 1.012515× | 1.000036× |
| OLMoE-1B-7B | 512 | 204.8 | 1.012657× | 1.000565× |
| Mixtral-8×7B | 32 | 12.8 | 1.000614× | 1.000010× |
| Mixtral-8×7B | 512 | 204.8 | 1.000616× | 1.000158× |
| Mixtral-8×22B | 32 | 12.8 | 1.000439× | 1.000007× |
| Mixtral-8×22B | 512 | 204.8 | 1.000440× | 1.000112× |

所有预取 speedup 的分母都是**同模型、同算力、同带宽的 streaming 无预取**。
Ranked 的最大收益：Switch-base-8/16 为 0.8845%/0.8665%；OLMoE 约 0.46%；两种 Mixtral 小于 0.06%。
不同 backend 的聚合带宽不能直接解释为相同的有用权重吞吐。

另看绝对延迟：从每通道 16 提升到 1024 byte/cycle，即聚合名义带宽从 12.8 提升到 819.2 GB/s，
在 **512 ops/cycle** 下，streaming 无预取的绝对提速为：

| 模型 | 12.8 GB/s 平均 cycles | 819.2 GB/s 平均 cycles | 绝对提速 |
|---|---:|---:|---:|
| OLMoE-1B-7B | 4,777,392 | 1,192,800 | 4.0052× |
| Mixtral-8×7B | 33,077,136 | 8,292,192 | 3.9889× |
| Mixtral-8×22B | 56,685,840 | 14,196,192 | 3.9930× |

因此本轮支持两个不同结论：

- 在原高算力、低带宽配置下，提高带宽可以显著降低总延迟。
- 这不自动扩大预取收益。Uniform 在算力提高 16 倍后，其最佳带宽点也提高 16 倍；
  继续增加带宽则使 streaming baseline 的剩余等待缩小，预取 speedup 逐渐接近 1。

仅凭本次结果不能声称 HBM 板卡上的实际加速比。U280/V80 的真实通道数、接口吞吐、
有效效率和物理实现尚未映射进本轮四通道单因素扫描。

20 样本子集的实际覆盖率：Switch 两模型 Top-2 为 82%；OLMoE candidate recall 为 83.125%；
两种 Mixtral 为 81.5%。没有将这些子集冒充完整原样本的 84.4%。

实际输出位于 `results/real_model_bandwidth_sweep_20261007/`，完整阈值和曲线见其中 `REPORT.md`、
`crossover.csv`、`comparison.csv` 及三组 PNG/PDF。
