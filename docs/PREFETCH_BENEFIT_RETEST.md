# 历史预取收益复测与收益消失路径

## 1. 已完成范围

2026-10-07，使用当前 `UnifiedSequenceSimulator` 完成 186 个配置点、4,650 次模拟。
没有修改主模拟器、Scheduler 算法、RTL 或 XDC，没有覆盖历史实验。

- 历史复测：9 个硬件点，各含无预测与 Uniform，共 450 次模拟。
- 扩展扫描：168 个配置点，共 4,200 次模拟。
- 5 个原 seed，每 seed 5 条独立冷启动单 token 序列，每序列 5 个 MoE 层。
- 完整回归：**578 passed, 14 warnings**；警告为既有依赖弃用提示。
- 检查了来源 SHA-256、样本唯一性、保存 trace 一致性、OCM 上限、字节守恒、
  baseline 无 speculative bytes、历史逐样本差值和 PNG/PDF 文件。

## 2. 固定条件与控制变量

H=128、F tile=64、E=16、INT8 SwiGLU、单逻辑顺序 GEMM Engine、32 ops/cycle。
4 内存通道、200 MHz、DMA/GEMM startup 各 8 cycles，非线性 64 elements/cycle。
预测由前一层 Router 完成后 20 cycles 触发，不等待上一层全部权重完成。
实际 byte OCM、整矩阵 fragment DMA 不可抢占，不启用小 burst 仲裁。
这是合成有效性能配置，不是 U200/HBM 实测吞吐。

使用归一化 payload 比例：

\[
\rho=2B_{aggregate}/C_{aggregate}
\]

历史复测 rho=0.9/1/2；扩展扫描 rho=0.9/2，因此每通道速率分别为 3.6/8 byte/cycle。
rho 不包含 startup、Router、非线性和实际仲裁，不能直接作为真实 bound 分类。

扩展轴：

| 轴 | 取值 | 说明 |
|---|---|---|
| Expert F groups | 2/4/8/16/32 | F=128/256/512/1024/2048；H 不变 |
| Top-K | 1/2/4 | 候选数量随 K 变成 2K；不是孤立改变 K 的因果实验 |
| 浅预取 | GU depth=1 | 固定每个候选的预取 bytes，不随 F 增长 |
| 比例预取 | GU depth=ceil(groups/2) | GU 占全 Expert 权重的 1/3；不是最优策略搜索 |
| scaled_expert OCM | 4 个当前 Expert | 随 F 增长；不能称固定硬件容量实验 |
| fixed_bytes OCM | 196,608 bytes | 固定为最小 Expert 的 4 倍；最多组数时仍可容纳 tile 工作集 |

历史 Top-1 使用保存的 `results/prefetch_dense/inputs.json`，没有重抽样。
扩展 truth 是历史 truth 的嵌套扩展：K=1 的 truth、候选与 ranking 保持原样；
同 seed 的 K=2 truth 是 K=4 的前缀。所有硬件/深度/容量点共享相同激励。

每 seed 的 20 个预测决策中，18 次所有 K 个真实 Expert 都命中且排名在前，
2 次全部 miss；始终保留错误候选。真实 Expert 的 candidate recall 为 90%。
这是**层内相关的 all-hit/all-miss 模型**，不是 K 个独立预测，也不是 CEPR 实测 trace。
因此不能从本轮外推真实模型 Top-K 的 hit 分布，或把 K=4 结果单归因于 K。

## 3. 历史收益和负谷值全部复现

450 个历史样本，逐一核对 total cycles、OCM peak、demand/speculative/useful/wasted bytes：
**全部零差值**。新增驱动 K=1 的重建执行与原始执行函数也有回归测试核对。

下表是 baseline 总周期 / predictive 总周期，baseline 均为同配置 streaming 无预测：

| F groups | rho=0.9 | rho=1 | rho=2 |
|---:|---:|---:|---:|
| 2 | 1.212208× | 0.985359× | 1.110521× |
| 4 | 1.110729× | 0.848104× | 1.054798× |
| 16 | 1.028618× | 0.951651× | 1.014188× |

这里报告 pooled ratio；历史报告主表用 seed ratio 均值，2 groups/rho=1 的均值为
0.985374×，与 pooled 的细小差异来自统计定义，不是模拟器漂移。

## 4. 第一条变化路径：Expert 变大，浅预取收益稀释

固定 K=1、depth=1、rho=0.9：两种 OCM 方案的这条曲线完全一致。

| F groups | F | baseline 平均 cycles | 预取平均节省 cycles | speedup |
|---:|---:|---:|---:|---:|
| 2 | 128 | 22,790 | 3,989.6 | 1.212208× |
| 4 | 256 | 40,020 | 3,989.6 | 1.110729× |
| 8 | 512 | 74,480 | 3,989.6 | 1.056598× |
| 16 | 1024 | 143,400 | 3,989.6 | 1.028618× |
| 32 | 2048 | 281,240 | 3,989.6 | 1.014390× |

**不只是定性猜测：净节省绝对周期保持不变，而 baseline 随 F 增长。**
本路径未出现 future eviction，预取 GU 覆盖率从 1/3 降到 1/48，
所以“大 Expert 收益小”在此路径是明显的分母增长/浅预取稀释现象。
并非所有模型或 tile 形状都必然遵循同一数值规律。

## 5. 第二条变化路径：增加 Top-K，会改变 DMA 竞争

固定 rho=0.9、scaled_expert 容量、浅预取：

| F groups | K=1 | K=2 | K=4 |
|---:|---:|---:|---:|
| 2 | 1.212208× | 1.106810× | 1.163010× |
| 4 | 1.110729× | 0.967562× | 0.966704× |
| 16 | 1.028618× | 0.977500× | 0.971874× |

收益不随 K 单调变化。K 增加也意味着当前层真实权重、候选数、预测流量和排名变动；
不能把差值写成单一 Top-K 效应。但这条配对路径已经展示：
**从小 Expert/Top-1 走向更多真实 Expert 时，之前的收益区域可以消失。**

rho=2 下仍存在收益：4 groups 的 K=1/2/4 分别为 1.054798/1.028223/1.057006×。
所以不能概括为“Top-K 大于 1 就没有收益”。

## 6. 第三条变化路径：盲目增加深度不能恢复收益

固定 K=1、scaled_expert 容量：

| rho | F groups | depth=1 | depth=groups/2 |
|---:|---:|---:|---:|
| 0.9 | 4 | 1.110729× | 0.927789× |
| 0.9 | 16 | 1.028618× | 0.829639× |
| 2 | 16 | 1.014188× | 0.995439× |

以 rho=0.9、16 groups 为例，25 条序列合计：

- depth=1：净节省 99,740 cycles。
- depth=8：净**增加** 736,154 cycles，虽然首 Expert 启动等待之和仍减少 97,642 cycles。
- depth=8：额外逻辑传输 10,108,928 bytes，记录 future-speculation blocking 1,017,916 cycles。
- 两者均没有 future eviction，不能把该点变慢归因于 OCM 驱逐。

这是流量和阻塞压力的诊断证据，不是精确的互斥时间账本。
各等待/阻塞计数可能重叠，不能直接相加扣减来解释 makespan。
也没有用本轮数据证明所有深预取负收益都由同一具体请求造成。

## 7. 第四条变化路径：更多 OCM 不保证策略更快

rho=0.9、K=4、4 groups、depth=1：

| 容量模式 | OCM bytes | speedup | 额外 transfer bytes | future blocking cycles |
|---|---:|---:|---:|---:|
| fixed_bytes | 196,608 | 1.029131× | 2,293,760 | 0 |
| scaled_expert | 393,216 | 0.966704× | 7,208,960 | 324,000 |

容量更大时当前策略允许更多 speculative 流量进入，反而拖慢总执行。
固定小容量点还发生 4,915,200 bytes 的 future eviction，所以这不是纯“更小容量更好”的结论；
它说明 **admission 与预取预算必须受控，物理空间充足不等于应填满它**。
必须同时查看流量、驻留和关键路径，不能仅比较 first-tile readiness。

## 8. 当前可支持的结论与后续方向

可以确认：

1. 历史 5%–21% 额外收益在当前主模型仍存在，不是旧模型偶然结果。
2. 保持浅 prefix 时，Expert 增大会稀释相对收益。
3. 候选/真实 Expert 数量与容量 admission 的变化会使收益变成负值。
4. 深度按尺寸线性放大不足以解决稀释，可能显著加重竞争。
5. rho=0.9/2 两条曲线不同，不能推导通用 compute-bound 阈值。

尚未验证的是：固定候选集合与固定全局 byte 预算的独立 K 消融、独立而非相关的
Top-K 命中分布、burst-safe admission、多层 lookahead、不同 H/tile 形状、真实模型 trace。
这些是后续实验轴，不冒充本轮完成项。
尤其当前四通道整 fragment 结果不能直接推广到已接入 burst 仲裁的 RTL 服务模型。

后续更有针对性的方向是：先固定全局 speculative byte 预算，控制 candidate breadth/depth，
再测试不侵占当前层关键 demand 的准入时序；不是继续单独提高 DSP 或带宽。

## 9. 产物与复现

新增配置：`configs/experiments/prefetch_benefit_retest.yaml`。
驱动：`examples/prefetch_benefit_retest.py`。
测试：`tests/test_prefetch_benefit_retest.py`，新增 8 个测试。

输出目录：`results/prefetch_benefit_retest_20261007/`：

- `summary.json`：完成状态、原始配置、当前执行规则与源 hashes。
- `inputs.json`：原始历史激励及各 K 的扩展 truth。
- `per_sample.csv`：4,650 个样本结果。
- `comparison.csv`：186 个配置的配对统计及诊断指标。
- `historical_deltas.csv`：450 个历史样本的六项指标差值，全部为 0。
- `REPORT.md`：全部配置表。
- `growth_rho_0.9.png/pdf`、`growth_rho_2.png/pdf`：两组曲线。

```bash
PYTHONPATH=src python examples/prefetch_benefit_retest.py \
  --workers 8 --output results/prefetch_benefit_retest_new
pytest -q -o addopts=''
```

驱动拒绝覆盖已有输出；worker 数量只表示主机并行实验，不增加模拟硬件并行度。
speedup 均为配对 baseline 总 cycles / policy 总 cycles；另存 seed 均值/标准差/min/max。
图的阴影是 seed 范围，不是置信区间；线段不证明未测点的连续收益边界。
