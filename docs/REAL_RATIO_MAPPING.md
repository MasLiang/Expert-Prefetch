# 合成收益配置向真实MoE结构的比例映射

## 完成范围与结论

使用主 `UnifiedSequenceSimulator`，保持真实模型H/F/E/Top-K与Expert架构，
将此前合成扫描中的PE/带宽、OCM/Expert、tile深度比例映射过来。
完成5个模型结构、640个配置点、1920个seed汇总点、19200条样本、57600条预测诊断。
模拟器核心、RTL及已有调度算法未改动；新增的是实验与分析脚本。

**结论：相对同粒度baseline，比例映射能在部分真实模型结构上产生较大预测收益；
但粗tile通常也削弱了无预测streaming，不能将这个百分比直接当作整体改进。**
相对原始细tile无预测baseline，本轮可行模型的最佳预测收益仍约0.7%–1.8%。
Mixtral的大OCM对照可有较大收益，但超出当前U200片上预算，不是部署结果。

## 配置

每点3个seed（20261005/06/07），每seed10个独立单token、5个连续MoE层序列。
第一层冷启动，不模拟完整模型attention/KV或真正checkpoint推理。
Top-K candidate recall=90%，不是CEPR实测；INT8为部署假设，未验证量化准确率。
两种候选排序共享相同truth与candidate set：

- `uniform_random`：命中随机混入候选，score按排名递减。
- `uniform_hits_first`：使用truth把已命中候选排在前面，是理想化排名消融，不能当成已实现的predictor。
- `oracle_all_gu`：只输出真实Expert、请求全GU；诊断参考，不是最优调度下界。
- `no_prediction`：相同tile配置下的精确按需streaming。

候选数保持2×Top-K；静态recall按所有真实Expert决策计，而非any-hit。
每seed Switch为36/40次命中，OLMoE为288/320，Mixtral为72/80。

PE=1/4，每PE32 ops/cycle，全部PE用于当前GEMM，Expert顺序执行。
4通道；按rho=0.75/0.9/1/1.5/2反推有效带宽：

\[
\rho=2B_{aggregate}/C_{aggregate},\quad
b_{channel}=\rho\times PE\times32/(2\times4)
\]

200 MHz；DMA setup=8、GEMM startup=8 cycles，非线性16 elements/cycle。
预测在前一层Router完成后20 cycles可用，不等待前一层全部权重加载。
actual-byte OCM，整tile不可抢占，未启用burst仲裁。
所有吞吐是合成有效配置，不是U200测量或PE资源综合结论。

| 模型 | H | F | E | K | 候选数 | INT8 Expert | OCM=4 Experts | 当前24 MiB预算 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Switch-base-8 | 768 | 3072 | 8 | 1 | 2 | 4.5 MiB | 18 MiB | 容量内 |
| Switch-base-16 | 768 | 3072 | 16 | 1 | 2 | 4.5 MiB | 18 MiB | 容量内 |
| OLMoE | 2048 | 1024 | 64 | 8 | 16 | 6 MiB | 24 MiB | 容量内 |
| Mixtral-8×7B | 4096 | 14336 | 8 | 2 | 4 | 168 MiB | 672 MiB | 超预算理论配置 |
| Mixtral-8×22B | 6144 | 16384 | 8 | 2 | 4 | 288 MiB | 1152 MiB | 超预算理论配置 |

Switch是两矩阵ReLU，GU字段代表单input projection，不是fused双投影。
其他三模型是SwiGLU。官方摘录来源与固定revision见既有model catalogue。
OCM容量内仅说明模拟器容量约束可满足，不证明当前RTL支持这些大tile或可综合映射。

## 粒度对照

- `mapped_4_groups`：F tile=F/4，Uniform请求1个GU tile。
- `mapped_2_groups`：F tile=F/2，Uniform请求1个GU tile。
- `fine_original`：F tile=128，Uniform请求2个GU tiles。
- `fine_equal_half_gu`：F tile=128，请求F/256个GU tiles，与mapped_2的GU byte预算相同。

Switch/OLMoE四种均跑；Mixtral仅两种粗tile，且使用理论大OCM。
不在24 MiB内强行运行装不下的Mixtral粗tile，不将其伪装为可行硬件。
本轮没有为Mixtral生成原始细tile分母，其跨layout指标留空，避免借用历史不同配置结果。

## 相对同layout无预测streaming

以下是每种策略在本轮测试点中的最高seed均值，不是一个统一配置：

| 模型 | 随机排序最佳 | 命中优先排序最佳 | 随机排序最佳layout / PE / rho |
|---|---:|---:|---|
| Switch-base-8 | 1.08680× | 1.15116× | 2 groups / 1 / 2 |
| Switch-base-16 | 1.08669× | 1.15201× | 2 groups / 1 / 2 |
| OLMoE | 1.03519× | 1.06095× | 4 groups / 4 / 1.5 |
| Mixtral-8×7B | 1.13854× | 1.13854× | 2 groups / 1 / 1.5 |
| Mixtral-8×22B | 1.13855× | 1.13855× | 2 groups / 1 / 1.5 |

Switch命中优先最佳位于2 groups / PE4 / rho0.9。
OLMoE随机排序最佳seed范围1.03450–1.03605×。
Mixtral-8×7B随机排序最佳范围1.13453–1.14236×，8×22B为1.13454–1.14237×。
Mixtral需要672/1152 MiB OCM，必须与当前可行模型分开解读。

## 原始细tile分母核对

对于Switch/OLMoE，另将每个case与**相同模型、PE、带宽、OCM、seed**的
`fine_original/no_prediction`配对。它排除了通过粗化tile、增加baseline等待来
放大预测百分比的误读。

| 随机排序case | 对自身layout baseline | 对原始细tile baseline |
|---|---:|---:|
| Switch-8，2 groups / PE1 / rho2 | 1.08680× | 0.97747× |
| Switch-16，2 groups / PE1 / rho2 | 1.08669× | 0.97749× |
| OLMoE，4 groups / PE4 / rho1.5 | 1.03519× | 0.96878× |

即使获得明显预测收益，这些case仍慢于同硬件的细tile无预测streaming。
粗tile增大启动所需的数据，并减少可交错的DMA/计算边界；因此不能仅凭第一列
判断整套方案更快。等byte细tile对照用于保留更多细粒度计算边界，预算也不会
因为tile更细而自动减少。

在本轮所有策略/粒度测试点中，相对原始细tilebaseline：

- Switch-8/16随机排序最佳分别1.00744× / 1.00743×，都是fine_original / PE1 / rho2。
- OLMoE随机排序最佳1.01397×，为fine_original / PE4 / rho1.5。
- OLMoE命中优先排序最佳1.01823×，为fine_equal_half_gu / PE1 / rho2。

这些分母都是本轮90%recall、5层配置，不直接与此前84.4%recall、6层结果相减。

## 解释与边界

合成比例可以迁移，但不足以完整决定速度：

1. Top-K改变工作集与时序。合成Top-1的4-Expert OCM很宽裕；OLMoE Top-8下，
   相同OCM/Expert比例并不能容纳全部真实Expert。
2. 候选数与排名改变实际admission和请求顺序，保持recall不等于保持tile命中率。
3. 大tile提高预取覆盖比例，同时削弱baseline streaming，二者必须分开计账。
4. rho忽略Router、startup、非线性和整数时间/仲裁，因此不是精确相似律。
5. 大Expert需要更大绝对OCM；Mixtral理论结果不能直接外推为单卡U200收益。

本轮确认了部分比例映射case确实有预测收益，但尚未找到在当前可行预算下、
相对原始细tilestreaming超过5%的配置。不声称“真实模型预取无用”，仅说明
这些映射变量和现有Uniform策略尚不能带来该幅度的整体收益。

## 验证和产物

405 tests通过，新增覆盖等GU byte预算、Top-8保持、候选集合不变、确定性、
容量与字节守恒、跨layout分母正确配对。
产物检查通过：640/1920/19200/57600行数、样本键唯一、OCM上限、
speculative=useful+wasted、无预测speculative为0、Oracle wasted为0、来源SHA256与PNG/PDF。

输出目录 `results/real_ratio_mapping/`：

- 每模型`*_inputs.json`、summary配置与源码SHA256。
- `per_sample.csv`、`per_seed.csv`、`per_prediction.csv`、`comparison.csv`。
- `cross_layout_comparison.csv`：细tile分母配对结果，Mixtral保持空值。
- `uniform_random.png/pdf`、`uniform_hits_first.png/pdf`：同layout分母图。
- `cross_layout_vs_fine.png/pdf`：可行模型的原始细tile分母图。

```bash
python examples/real_ratio_mapping.py --workers 4
python examples/analyze_real_ratio_mapping.py
pytest -q -o addopts=''
```

未下载权重，未做真实预测器训练或精度校准，未运行RTL，不声称完整模型部署。
