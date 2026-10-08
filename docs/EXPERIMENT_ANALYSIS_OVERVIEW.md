# MoE 预取与资源感知 tile 调度：数据分析总览

整理日期：2026-10-07。本文是历轮分析的中文阅读入口，不重新运行实验，不改写历史数据或模拟器语义。
“当前推荐”表示解读同一研究问题时优先采用的结果，不表示所有旧矩阵都已用当前源码重跑。
配套机器可读索引：[EXPERIMENT_ANALYSIS_INDEX.csv](EXPERIMENT_ANALYSIS_INDEX.csv)。

## 1. 一页结论

我们的研究已经从“预测几个 Expert”转向：

> 在有限 OCM 与内存服务能力下，如何把预算分配给候选 Expert 的 tiles，并安排传输，使整体完成时间真正减少？

现有数据共同支持：

1. tile allocation 可扩大候选覆盖，但**计划覆盖、实际传输、驻留可计算和最终加速是四件不同的事**。
2. streaming 本身是强优化，必须与预测预取的额外收益分开。
3. 预测准确/排名更好通常有帮助，但不能保证预取不伤害当前层。
4. 小 Expert、合适窗口与资源比例下，预取相对 streaming 可有显著收益；大 Expert 的固定浅前缀通常被稀释。
5. 更多 OCM、带宽或 MAC 不保证预取 speedup 单调提高；绝对延迟仍可能明显改善。
6. baseline 内存利用率高时，带宽下界压缩预取空间；利用率很低也可能只是 baseline 已无多少 memory stall。
7. 没有一个当前 allocator 在全部配置上最优。OFF/LIGHT/更深预取应是资源与时序相关的选择。
8. 真实模型结构矩阵目前只支持较小、条件性的额外预取收益，不支持“大模型普遍明显加速”的结论。

**工程验证与研究结论的边界：**已有 RTL 功能/综合及指定夹具对齐证据，但没有把这些假想硬件扫描变成板级性能实测。
2026-10-08单token粒度实验后的最近全量回归为 **591 passed, 14 warnings**。最初资料整理未重跑DSE；下列新增粒度节来自后续实际实验。

## 2. 三类结果必须分开

| 类别 | 主要研究 | 执行范围 | 能回答什么 | 不能直接声称 |
|---|---|---|---|---|
| 单层分配研究 | M8、M9、M9.1–9.3 | `AdvancedSimulator`，代表性单 decode MoE 层 | 固定候选、预算和单层窗口内的allocator比较 | 多层共享资源下相同收益 |
| 多层架构研究 | M7.1–7.3、真实模型对照、密集扫描与近期复测 | `UnifiedSequenceSimulator`，源层里程碑因果预测 | 当前层与未来层的DMA/OCM/计算竞争、整序列makespan | 完整Transformer或实测板卡延迟 |
| RTL对齐研究 | M10接口/资源规则与六层AXI夹具 | 声明RTL规则的主模型与XSim | 同激励的请求、字节、顺序与周期对照 | 所有研究profile都周期精确、真实DDR已校准 |

单层 `AdvancedSimulator` 不等于已弃用的 `M10AlignmentSimulator`。前者仍是有效的局部方法研究工具；
后者不应恢复为当前多层对齐主线。历史 M6 target-local 窗口用于微架构探索，也不等于真正跨层预测。

## 3. 统一 baseline 和术语

建议以后每张图明确写出下列分母，而不要只写“speedup”。

| 记号 | 定义 | 解释 |
|---|---|---|
| A | 整个真实 Expert 权重到齐后计算，无预测 | whole-ready 弱baseline |
| B | tile streaming，无预测 | 判断预取额外贡献的强baseline |
| C | 请求完整候选 Expert，真实 Expert 全权重到齐后计算 | 非streaming完整预取 |
| D | tile streaming + 预测引导tile预取 | 融合方案 |
| A/B | A cycles / B cycles | streaming本身的加速 |
| A/C | A cycles / C cycles | 完整预取相对whole-ready的收益 |
| A/D | A cycles / D cycles | 合并收益，不能全部归因于预测 |
| B/D | B cycles / D cycles | 预测相对streaming的额外收益 |

注意不同早期脚本不都使用这些case名称；需先查看启动门槛再映射，不能按policy字符串猜。

还必须区分：

- `full_expert_prefetch`可能仅表示**全部tiles的请求目标**，不意味着全部驻留再启动。
  Mixtral在当前OCM下不能完整驻留，只有all-tiles target + streaming。
- `ranked_prefix`旧规则通常是depth减rank；M8/M9的Ranked allocator是另一算法；
  旧Hardware Greedy与冻结RTL的HardwareGreedy-v1也不是同一个实现。
- M9.3的quality是**真实Expert排名移动参数**，不是命中率；本轮headroom的accuracy才是Top-1命中比例。
- ready recall必须按实际驻留与启动依赖计算，不能用静态candidate recall替代。
- Expert memory-wait之和与blocking计数可能重叠，不是makespan的互斥时间分区。
- `memory occupancy`含setup；payload bandwidth utilization按实际bytes/带宽/时间定义。
- 512 ops/cycle相当于256 MAC lanes；512 MAC lanes相当于1024 ops/cycle，不能混用。

## 4. 历轮分析导航与主要发现

### 4.1 M7：统一因果模型、指标审计和政策相图

M7.1、7.1a/b/c建立了源层里程碑、真实Router authority、请求生命周期、startup与最终关键路径的解释框架。
M7.1c未发现其所测metamorphic invariant对应的因果错误；提前启动有时被后续瓶颈吸收，makespan不变。
这不代表后续实现永远无bug；M10后来的FIFO可见性问题有独立修复记录。

M7.2：240个控制配置、两种预算比较模式、12,249个可行plan评估。
unconstrained模式中：OFF赢147/240，BDS赢89，Full赢3，固定Ranked赢1。
125个cell有精确延迟并列，只有67个winner有至少1% margin。
BDS相对最佳可行简单策略在36/212点超过1%，最大18.34%；不是处处优于简单策略。

M7.2a清理了winner/tie/可行性图示，属于重新解读已有数据，不应与模拟次数重复相加。
M7.2b补3个实际机会点（约1.066/2.316/3.316 Expert等价），中高质量都选BDS，低/错误质量OFF；
这条切片没有观察到机会量驱动的新策略转折，BDS speedup均为1.01872×。

M7.3审计确定了边界：参数主要为假设或推导，不能把VEK280 theoretical映射当测量。
它对当时profile的描述有历史时点性；后续主模型加入了可选FIFO/slot/burst等规则，不能把旧“未建模”表当最新能力清单。

阅读：

- [M7.1总览](../M71_UNIFIED_CHARACTERIZATION.md)
- [M7.2政策相图](../M72_POLICY_PHASE_REPORT.md)
- [M7.2b机会边界细化](../M72B_OPPORTUNITY_REFINEMENT_REPORT.md)
- [M7.3保真度审计](M73_SIMULATOR_FIDELITY.md)

### 4.2 M8–M9：tile预算分配与硬件友好allocator

M8在324个配置、1,296个策略结果中，Ranked相对No Prefetch平均1.1239×，相对Full平均1.0207×。
Ranked比Full快242点、相同7点、慢75点；优势是条件性的，尤其受容量和排序影响。

M9增加Budget Proportional、Hardware Greedy和离线Oracle：48个点、336个报告结果，
另有3,966个Oracle候选plan模拟。Hardware Greedy距离该离线Oracle平均约28.87%，
说明不能把“整数/无除法易实现”当作“接近最优”的证明。

**历史文字勘误：**M8 `SUMMARY.md`第4条说useful-byte efficiency从0.391“上升”到0.156/0.183，
数值实际是下降，不能引用为“tile必然降低浪费”的证据。本文保留原报告，明确指出不一致。
广度覆盖与useful/transferred效率不同，也不直接决定最终延迟。

阅读：[M8结果](../results/m8_tile_scheduler/SUMMARY.md)、[M9结果](../results/m9_hardware_scheduler/SUMMARY.md)。

### 4.3 M9.1：代表性规模研究，不是真实模型checkpoint

3 workloads（Small/Medium/Large）、3理论/合成硬件、4quality、4OCM、3BW，
共432配置、3,024策略报告行，另评估21,526个Oracle候选。
每Expert为16个logical tiles；单层Top-2；层数只是模型metadata，不乘到报告的单层latency。

平均No Prefetch相对策略的speedup：Full1.02475×、Uniform1.04665×、Ranked1.04582×、
Budget Proportional1.04919×、Hardware Greedy1.04570×、Oracle1.27501×。
小负载、较高带宽与准确排名时Full可赢；更大工作集通常削弱固定完整Expert策略。
这里的均值是配置速度比的均值，不与近期合并周期比直接拼接。

阅读：[M9.1规模分析](M91_REALISTIC_WORKLOAD_EVALUATION.md)。

### 4.4 M9.2：预测损失与分配损失拆分

144配置、1,296报告行，比较Full Oracle和固定候选/排序下的离线plan搜索。
固定预测Oracle仍由离线真实latency选择，非可部署策略，也不是所有可能调度的全局最优证明。

Budget Proportional平均1.04753×、Hardware Greedy1.04518×，新增Criticality Aware仅1.02151×。
Criticality相对Greedy快11/144点、并列53、慢80，**提出的criticality公式没有在整体胜出**。
中低rank quality下预测损失更重要；perfect ranking下仍有明显allocation优化空间。

阅读：[M9.2关键路径分析](M92_CRITICALITY_AWARE_TILE_SCHEDULING.md)。

### 4.5 M9.3：ranking quality与confidence calibration

19,440次模拟；9级quality、4种保持ranking不变的calibration、5个固定部署策略。
well-calibrated子集4,860行，robustness汇总4,860行；这些是同一矩阵的切片/汇总，不额外相加成模拟数。

- quality从0.2到1，Budget/Greedy/Ranked匹配latency降低8.45%/6.64%/6.57%。
- 固定quality时三个tile allocator平均latency spread仅0.96%。
- calibration下Greedy的speedup标准差0.00197，Budget为0.00544：Greedy更稳，不等于处处更快。
- q=0.2时Greedy/Ranked仍约1.0108×/1.0110×，但这是排名变量，不是20% recall。
- Full在972个well-calibrated点中只24点为唯一最优，集中小负载、高quality、高带宽。

阅读：[M9.3预测敏感性](M93_PREDICTOR_SENSITIVITY_ANALYSIS.md)。

### 4.6 FLAME相关比较与Switch真实结构

我们没有复现N:M pruning、CEPR训练、samsum真实route trace、跨tokencache及论文有效吞吐。
只借用其模型结构和统计结论构造受控假设；历史1.164×/1.2486×来自特定旧case，不能作为通用FLAME与本方案对比。
同名模型/相同候选recall不等于论文复现，也不能把CPU基准组合加速归因于纯预取。

当前四case主要输入：Switch-base-8/16，H768/F3072、ReLU、Top-1、INT8 Expert4.5MiB，
6个decoder MoE层、100个独立序列；32ops/cycle、16MiB OCM、四通道、4KiB burst、payload倍率2。
Top-1=75.8%、Top-2=84.4%为人工联合排序假设，非实测联合分布。

Switch-8、每通道名义8B/cycle代表点：

| Case | 当前平均cycles | 相对A | 相对B |
|---|---:|---:|---:|
| A | 3,569,613 | 1.0000× | 0.5135× |
| B | 1,832,877 | 1.9475× | 1.0000× |
| C | 2,449,911.67 | 1.4570× | 0.7481× |
| D | 1,827,152.46 | 1.9536× | 1.0031× |

**大部分A→D提升是streaming贡献，不是预测贡献。** C虽相对A明显提速，仍慢于B。

宽候选实验固定9MiB预算：Full覆盖2个Expert，Uniform/Ranked覆盖全部8/16个。
覆盖确实扩大，但低带宽仍可负收益，高带宽额外收益约0.8%–0.9%。
不要把分配覆盖广度当成“这些Expert已经在Router前驻留”。

阅读：[Switch四case](SWITCH_REAL_MODEL_FOUR_CASES.md)、[宽候选分配](SWITCH_WIDE_TILE_PREFETCH.md)、
[FLAME条件对照及历史状态](FLAME_CONDITION_COMPARISON.md)。

### 4.7 OLMoE与Mixtral真实结构

均为单token、6层截取、人工predictor，非完整模型数值推理。8/16/24MiB OCM、32/512ops/cycle；
矩阵fragment非抢占DMA，没有沿用Switch的burst后端。

| 模型 | H/F | E/K | INT8 Expert | whole-ready可行性 |
|---|---|---|---:|---|
| OLMoE-1B-7B | 2048/1024 | 64/8 | 6MiB | 单Expert可行，不是全部Top-8同时驻留 |
| Mixtral-8×7B | 4096/14336 | 8/2 | 168MiB | 当前OCM不可行，仅streaming |
| Mixtral-8×22B | 6144/16384 | 8/2 | 288MiB | 当前OCM不可行，仅streaming |

16MiB下Uniform相对streaming：OLMoE在32ops/cycle为1.01294×，512ops/cycle为0.93199×；
两种Mixtral在32ops/cycle仅约1.00061×/1.00043×，提高compute后可为负。
不能把负收益视作仿真失败，也没有“最差自动回退到OFF”的性能保证。

阅读：[OLMoE/Mixtral矩阵](OLMOE_MIXTRAL_CASES.md)。

### 4.8 当前版本五模型复测：推荐真实结构主入口

9,300次重新模拟、138配置，复用原激励，不复用checkpoint、不覆盖旧结果。

- Switch四case2,400样本cycles全部不变。
- OLMoE/Mixtral4,500样本cycles全部不变。
- Switch宽候选2,400样本中78个cycles改变，最大0.299432%；另有投机byte变化。

对这些三套矩阵，推荐使用当前复测目录，而不是把旧报告中的四舍五入数字当最新依据。
这**不是全部M7–M9 DSE已重跑**。

阅读：[复测范围](CURRENT_REAL_MODEL_RETEST.md)、[实际总表](../results/current_model_retest_20261007/REPORT.md)。

### 4.9 真实结构带宽扫描

7,560次模拟、378点；各模型复用20个保存样本，OCM16MiB；名义四通道聚合3.2–819.2GB/s。
这是改变外存服务率，不是U280/V80真实HBM接口模型，RTL也未实现这么宽的通路。

- 在512ops/cycle下，提高每通道16→1024B/cycle，OLMoE/Mixtral的B绝对提速约4×。
- Uniform最佳额外收益：Switch约0.87%，OLMoE约1.27%，Mixtral小于0.07%。
- 继续提高带宽，额外预取收益通常又靠近1×；不等于总延迟没有改善。

阅读：[带宽单因素扫描](REAL_MODEL_BANDWIDTH_SWEEP.md)。

### 4.10 MAC/DSP扫描

最新5,400次模拟、270点：固定内存、16MiB OCM，16–512假想MAC lanes。
当前RTL未物理扩宽到512lanes；假设OCM供数/归约也能相应扩宽，不能把算术DSP预算当综合通过。

- B在16→512lanes：Switch约1.97×，OLMoE约4.06×，Mixtral约4×。
- 已测点中B距离最快点1%以内的最小规模：Switch/Mixtral64lanes，OLMoE192lanes。
- 512lanes时Uniform相对B：Switch约1.00065×，OLMoE0.92994×，Mixtral0.98980×/0.99387×。

阅读：[最新DSP敏感性](U200_DSP_LANE_SWEEP.md)。
旧[RTL规则/PE扫描](RTL_RULES_PE_SWEEP.md)保留为历史消融；其“RTL仍有weights_done屏障”是当时状态，已非现状。

### 4.11 合成收益相图、密集扫描与当前复测

首轮/补充合成扫描1,782点、8,910样本；密集扫描922配置、23,050样本、5seed。
H128/Ftile64，改变F groups、算力/带宽比例rho、容量、质量和GU depth。
结果发现多个收益岛，而不是单一compute-bound门槛；rho=1局部谷值有逐请求非抢占证据。

最新4,650次模拟先复测9个历史硬件点、450个样本，六项指标全部零差；再扩展尺寸/Top-K/深度/容量。
固定Top-1、rho0.9、GU depth1时：groups2/4/8/16/32分别1.2122/1.1107/1.0566/1.0286/1.0144×。
每序列净节省保持3,989.6cycles，分母随Expert增大，直接显示浅预取稀释。
16groups加深到8时降为0.8296×；更多OCM也可能因放入更多有害投机而变慢。

阅读：[机制相图](PREFETCH_BENEFIT_PHASE.md)、[多seed密集扫描](PREFETCH_DENSE_SCAN.md)、
[当前复测与收益消失路径](PREFETCH_BENEFIT_RETEST.md)。

### 4.12 最新内存利用率×准确率方向实验

2,340次模拟、156配置、72相图点；固定H128/F256/Top-1、单候选、384KiB OCM。
通过改变带宽产生不同实测B利用率；LIGHT为GU1，PREFETCH为完整目标但仍streaming。

- B占用率97.91%时perfect两策略都仅1.0170×。
- 占用率6.98%时perfect都仅1.0091×：内存已快，baseline没多少等待。
- 86.11%时perfect LIGHT1.1250×，但84.12%时LIGHT0.9468×：正确预测仍有时序风险。
- 相图选择需所有seed都过1%门槛；不是连续边界，也不是已实现的运行时开关策略。

在必需传输W与带宽B不变、计时窗口外不提前传输时，有必要上限：

\[
speedup\le 1/U_{payload}
\]

这个上限解释高payload利用率的headroom限制；不保证低U就有收益，不能无条件用含setup的occupancy代替。
研究还需同时考虑**baseline关键路径等待量**和**空档是否因果可利用**。

阅读：[利用率与预取空间](PREFETCH_MEMORY_HEADROOM.md)。

### 4.13 Tile粒度扫描（2026-10-08新增）

固定headroom H128/F256/OCM384KiB，F_tile=8/16/32/64/128，六个带宽切片，perfect/90%单候选。
3,150次模拟、210配置，含GU1、固定32KiB GU、完整96KiB目标三种策略。
rho0.9、90% GU1：最佳OFF宽度16，最大B/D宽度128，预取后总体最快宽度32。
最大同tile B/D为1.1986×，但最佳对最佳只有1.0443×；不能以较差baseline抬高增量收益。
36个策略切片中29个overall最优与OFF最优不同，均为有限测试集合，不是全局最优证明。
保存setup fraction、payload/AIE利用率、baseline反事实slack、requests和blocking。
safe-slack fit针对单请求且依赖事后baseline时间线，不是在线安全保证。

阅读：[tile粒度完整报告](PREFETCH_TILE_SIZE.md)。

## 5. RTL对齐证据单列，不混入策略平均收益

完整尺寸Switch六层四通道AXI行为夹具中，D此前2.60%偏差主要来自reserved FIFO credit被错误当作已入队请求。
修复后主模型/RTL：

| Case | 主模型cycles | RTLcycles | 差值 |
|---|---:|---:|---:|
| A | 3,013,531 | 3,013,249 | +282 |
| B | 1,821,517 | 1,821,517 | 0 |
| C | 1,998,389 | 1,998,342 | +47 |
| D | 2,006,406 | 2,006,224 | +182 |

D的431个首次DMA与288个compute阶段顺序、byte统计和slot峰值匹配；剩余182cycles未完全逐控制边界归因。
这是一个指定夹具和资源profile的证据，不是研究byte OCM矩阵的全体周期精确验证。
当前长时真实DDR器件实验已停止；此表不包含板级MIG/DDR校准。

阅读：[请求级对齐修复](SWITCH_D_ALIGNMENT_DEBUG.md)、[独立allocator/DMA顺序配置](PREFETCH_ALLOCATOR_DMA_ORDER.md)。

## 6. 为什么旧分析与最近结果看起来不同

1. **单层与多层：**孤立层可利用Router窗口；统一多层未来DMA也会阻塞当前真实权重。
2. **分母不同：**相对A的1.95×不等于相对B的额外1.95×。
3. **quality不同：**真Expert在候选尾部、candidate miss和confidence变换是三个不同实验轴。
4. **预算不同：**GU-only prefix、GU/DOWN logical tile、all-tiles计划的实际传输量不同。
5. **资源规则不同：**byte容量与fixed slot/pair、无限与有限credit、fragment与burst不能直接拼图。
6. **计数不同：**报告行、独立模拟、候选plan、汇总切片和离线搜索不是同一单位。
7. **统计不同：**早期配置speedup均值、seed均值、近期总cycles比不应混作一个数字。
8. **版本不同：**promotion/FIFO可见性修正前数据须保留历史标签，不能继续用于解释已修复偏差。

不建议把全部目录的runs相加成“总实验数”；其中包含复测、缓存Oracle、重复汇总和旧版探索。

## 7. 推荐阅读顺序与论文可用结论

若准备下一轮研究或写论文，建议按下面顺序：

1. 先读本文baseline/术语表。
2. 当前真实结构复测：确认真实规模下强baseline的剩余空间。
3. 历史收益复测：看到哪些配置有大收益及如何消失。
4. 内存利用率实验：给headroom上限和时序反例。
5. M9.2/9.3：说明ranking、calibration和allocator的不同影响。
6. M7.2/BDS：说明策略区域、near-tie、OFF和离线搜索空间。
7. 最后看RTL对齐与保真度审计，明确硬件验证边界。

**目前可写：**条件性的fine-grained allocation优势、预算/时序的重要性、强baseline剩余空间、
ranking质量敏感性、错误/正确投机都可能竞争、多层统一事件模型与指定RTL夹具验证。

**目前不能写：**普遍大幅优于streaming、精确FLAME复现、HBM设备实测、更高DSP自动增加预取收益、
单一utilization/recall阈值决定最优策略、criticality公式整体胜出、所有研究profile都已经RTL验证。

## 8. 未完成的研究空白（不是整理任务未完成）

- 固定全局speculative bytes，隔离breadth/depth与Top-K效果。
- 对比有相同平均U、但空档位置不同的负载。
- 对照burst-safe或更谨慎admission，量化正确预取造成的时序伤害。
- 真实模型routing/predictor trace、独立Top-K hit分布与多层lookahead。
- 宽MAC真实供数/归约/综合与板级内存有效参数校准。

本次新增总览与索引，没有重新模拟、没有移动/删除旧目录，也没有修改旧报告或实验结论。
