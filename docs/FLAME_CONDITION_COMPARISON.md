# FLAME条件启发对照：Switch完整Expert与tile预取

## 范围

本轮从OLMoE Top-8切回FLAME使用的Switch-base-8/16、Top-1；
比较四种执行方式，并单独消融“上一层真实权重全部完成才放行”的条件。
主预测比较明确设置`wait_for_previous_layer_weights=False`。
使用已有`UnifiedSequenceSimulator`，不修改核心算法，不修改RTL。

这是**FLAME条件启发对照，不是FLAME数值复现**。继续遵循项目决定：
无N:M剪枝、不实现CEPR训练或预测器，只借用其双候选覆盖率结论。

## 当前RTL核验

`hardware/m10_u200/rtl/cross_layer_prefetch.sv`的eligible仍包含
`weights_done[layer-1]`，所以当前工作区的该RTL路径尚未放开屏障。
此前“已放开”的说法不能替代当前源码证据。
模拟器的`UnifiedPredictionSpec`默认早已为False；上一轮四case实验显式开启，
本轮主比较解除，旧开启规则只保留为单独参考，历史结果不覆盖。
本报告不声称RTL已做相同修改或已跑RTL联合仿真。

解除屏障不等于取消硬件约束：demand优先、burst非抢占、有限OCM、
Router truth、single Engine和计算依赖仍然保留。
该主模型会在burst边界重选请求，不会在已开始的burst中途抢占。

## 论文中正确的比较对象

FLAME全文第6页Table 1：FPGA baseline=31.27 ms，仅CEPR=21.08 ms，
对应31.27/21.08=1.4834×。4.12×是剪枝与CEPR组合相对CPU，不能当作纯预取收益。
第7.3节报告CEPR延迟收益1.48×–1.98×，MoE latency评估排除了其他层。
论文使用U200、Switch-base-8/16及samsum；CEPR使用跨token的环形激活路径信息。

本轮没有samsum真实route trace、跨token环形预测、PA-BUF/N:M实现或FLAME实测
有效GEMM/DMA吞吐校准。INT8、带宽和控制时序全部是本项目配置假设，
不能因模型名称、板卡或candidate recall相同就将cycles称为论文复现。

RTL当前Expert后端是SwiGLU，本轮Switch ReLU为模拟器负载；不声称RTL已经具备
对应ReLU整模型数值路径。

## 输入与共享硬件

Switch：H=768，F=3072，ReLU两矩阵Expert，E=8/16，K=1，权重/激活INT8假设。
每Expert权重4.5 MiB；六个decoder MoE层，索引1/3/5/7/9/11；不模拟attention/encoder。
F tile宽度128；每矩阵tile=96 KiB；24个F groups。

每seed50个独立冷启动序列，后五层共250个预测目标，211命中，恰好84.4%。
候选2个，命中rank在0/1之间人为均衡，confidence=0.6/0.4；不是CEPR实测排名分布。
source Router done+20 cycles产生预测，预测allocator延迟由配置结构公式决定。
第一层没有预测；不模拟前一token对第一层的预测或权重复用。

200 MHz，4通道，单逻辑GEMM Engine，DMA/GEMM startup均8 cycles，
DMA burst=4096 bytes；请求、选择、payload倍率2、返回、控制和非线性成本四case一致。
标称每channel 4/8/16/32 bytes/cycle；payload窗口实际速率为标称的一半，
startup/仲裁/返回还会进一步降低服务速率。最高档不是U200实测能力声明。
MAC groups=1/4/16，等效16/64/256假想lanes，32/128/512 ops/cycle。
不是增加Expert并行数，更宽OCM/MAC尚未物理实现。

主OCM=16 MiB，共享当前层工作权重和未来缓存；这能容纳当前4.5 MiB Expert加
两个4.5 MiB候选，**不等于PA-BUF两个独立cache硬件的精确模型**。
另在Switch-base-8、64 lanes、16 B/cycle/channel下扫描9 MiB容量。

主扫描每点一个固定seed；参考点64 lanes、16 B/cycle/channel及9 MiB容量探针
分别额外跑两个seed。不能将单seed主曲线标为三seed统计结果。

## 六组case

| Case | 计算门槛 | 预取 | 上一层权重屏障 |
|---|---|---|---|
| A | 整Expert全部GU/DOWN resident再计算 | 无 | 无预测，不适用 |
| B | 按tile就绪streaming | 无 | 无预测，不适用 |
| C-open | 与A相同 | 完整candidate Expert请求 | 关闭，主比较 |
| D-open | 与B相同 | Uniform GU prefix depth=2 | 关闭，主比较 |
| C-barrier | 与A相同 | 与C-open相同 | 开启，消融 |
| D-barrier | 与B相同 | 与D-open相同 | 开启，消融 |

所有case统一byte OCM、无限request credits、expert_locked；非streaming路径尚不支持
fixed-slot/atomic-pair/finite credits，所以不把不同容量或credit模型拼成横向比较。
C请求完整candidate，但可能部分admission/被拒；任何真实Expert计算仍必须等完整权重。
D不是RTL HardwareGreedy。本轮不增加调度算法。

## 必须区分的指标

- `candidate_recall`：真实Expert出现在候选中的比例。
- `full_true_ready_recall`：C中Router resolved时真实Expert全部权重实际resident的比例。
- `full_true_transfer_recall`：真实Expert全部speculative传输已完成的比例；不同于resident。
- `useful_physical_bytes_before_router`：Router时刻真实候选完成与部分in-flight的payload。
- `release_delay_per_target`：predictor ready到预取放行；不等于DMA开始等待。
- `speedup_vs_A`：同模型、MAC、带宽、OCM、seed的A总cycles/current总cycles。
- `speedup_vs_B`：相对已有streaming的净预取收益，不能把它和相对A的收益混淆。
- `speedup_vs_barrier_on`：同一策略开启屏障时间/当前时间，隔离放行政策变化。
- `net_saved_cycles_per_sequence`：配对A-current的实际makespan净减少，可为负。
- `aggregate_wait_reduction_per_sequence`：Expert等待汇总差；不是DMA/控制的严格因果分解。

“实际隐藏加载时间”不能只以candidate hit乘Expert bytes推算。本轮使用真实驻留、
传输进度与配对等待/makespan差共同分析；不将理论payload时间全部称为已隐藏等待。
负收益点不删除，取消屏障也不预先假定单调提速。

## 复现

```bash
python examples/flame_condition_comparison.py --workers 4
pytest -q -o addopts=''
```

输出`results/flame_condition_comparison/`：输入JSON、per_sample/per_seed/per_prediction/comparison CSV、
summary与两组PNG/PDF（收益、完整驻留recall）。所有实验来源哈希在summary中保存。

全部实验已完成，测量结果及验收记录如下。

## 首候选质量的独立敏感性

原来的84.4%双候选覆盖率并未定义排名分布。均衡hit时，50样本主矩阵的
第一名命中率约42.4%；这不是FLAME所报告的单cache最高75.8%。
论文中的两个最大值不一定出自同一配置，不能据此声称真实联合分布已知。
补充实验因此明确标为独立敏感性假设：100样本、500目标、Top-1=379/500=75.8%、
Top-2=422/500=84.4%；匹配参考Top-1=211/500=42.2%、Top-2仍84.4%。

两组复用相同truth和candidate集合，只改变hit在rank0/1的位置；
固定seed随机打散hit位置，较高Top-1组保留较低Top-1组已有的rank0 hit。
confidence数值仍为0.6/0.4，只作为排序输入，不声称概率已校准。
各有两个供数点，16 lanes、16 MiB OCM，六case均重跑；没有只选有利case。

```bash
python examples/flame_condition_comparison.py --config configs/experiments/flame_head_quality.yaml --output results/flame_head_quality --workers 8
python examples/flame_condition_comparison.py --config configs/experiments/flame_head_quality_reference.yaml --output results/flame_head_quality_reference --workers 8
```

## 当前burst/FIFO行为的定向诊断

本节及后续原始表格为修正前历史记录。该 promotion 问题现已修复，并完成
主模型和 RTL 四 case 验证，详见 [修正与验证记录](M101_PROMOTION_FOUR_CASES.md)。

在Switch-base-8、64 lanes、标称16 B/cycle/channel、首seed第一个序列，C-open的
layer7 Router resolved=678,352，Expert start=1,060,244。
该层真实Expert仅剩GU tile23这一已开始、后被命中的descriptor迟迟没有完成：
它首次开始于677,414，最后完成于1,060,239。
同一期间未来layer9传输了9,437,184 bytes（2304个burst），全部发生在当前Expert启动前。

这不是整个Expert必需等待未来层的数学依赖，而是当前队列行为：
`MultiChannelMemory.promote`只给尚未开始的hit增加promoted token；
已开始的speculative hit可能在burst让出后回到speculative FIFO尾部，
排在已入队未来请求之后。可用的未来带宽会因此挤占当前关键请求。
RTL的`burst_dma_channel.sv`也区分原active class和descriptor promotion，
但本轮未进行RTL联合验证或修改，不能将此Python样本当作RTL逐cycle实测。

因此“解除屏障”的性能效应包含该既有队列行为，不等于理想的critical-demand调度。
本轮如实保留结果，未为追求论文速度比偷偷改变DMA/allocator算法。

## 已完成的配对排序实验

以下Switch-base-8、16 lanes、16 MiB OCM的两组使用相同100序列truth、candidate集合、
硬件配置及Top-2=84.4%，只改变Top-1质量。均为关闭屏障。

| 标称B/cycle/channel | Top-1 | A无stream/无预取 | B stream/无预取 | C无stream/完整预取 | D stream/tile预取 | D/B |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 42.2% | 1.0000× | 1.9475× | 1.2107× | 1.9000× | 0.9756× |
| 8 | 75.8% | 1.0000× | 1.9475× | **1.4570×** | 1.9182× | 0.9849× |
| 16 | 42.2% | 1.0000× | 1.4956× | 1.2604× | 1.5068× | 1.0075× |
| 16 | 75.8% | 1.0000× | 1.4956× | 1.2856× | 1.5068× | 1.0075× |

8 B/cycle/channel点的C：A=3,569,613 cycles/sequence；
均衡排序C=2,948,379.62；首候选75.8%时C=2,449,912.77。
相对A实际净减少1,119,700.23 cycles，延迟降低约31.37%。
只改变排名分布，C自身加速约1.2035×；不是更换PE/BW/容量后拼接的比较。

该点完整驻留recall为0，但多数rank0 hit的权重已在Router前完成绝大部分传输，
最后一部分略晚于Router。因此计算仍等待整Expert，不违反C语义；
也不能因接近1.48×就声称论文的完整cache-hit机制已复现。
16 B/cycle/channel时较高Top-1组的完整resident recall=76.0%，仍不等于84.4%候选覆盖率。

当窗口只够加载首候选时，Top-1质量对C影响很大；窗口能够覆盖更多候选时影响缩小。
这解释为何只复制一个84.4%的Top-2数字、却把hit均分到两名，会低估某些完整预取场景。
不过不能从论文两个独立最大值推断真实联合排名分布，这是本轮明确控制的敏感性假设。

该点D相对A为1.9182×，但B本身已有1.9475×，所以当前D的tile预取不是正净收益。
必须同时报告D/B，不能把streaming已有收益全部归给预测。

首候选较高组C在8 B/cycle/channel下，屏障开启/关闭为1.4570×/1.4570×，基本相同。
Top-1非streaming当前层会一次性排入全部demand，DMA优先级已经使future等待，
单独取消硬屏障不一定创造更多实际DMA窗口。
16 B/cycle/channel下C开屏障为1.3103×，关屏障为1.2856×；负向变化同样保留。

## 主矩阵结果及容量参考

均衡排名主矩阵中，关闭屏障的C最佳点均为16 lanes、标称16 B/cycle/channel、16 MiB：
Switch-base-8为1.248336×，Switch-base-16为1.248856×；完整resident recall均45.6%。
这仍是一个固定seed的扫描点，不应与三seed参考点混称。

64 lanes、标称16 B/cycle/channel的三个seed配对均值如下，分母为各自相同配置的A：

| 模型 / OCM | B stream无预取 | C-open完整预取 | C-barrier完整预取 | D-open tile预取 | D-barrier tile预取 |
|---|---:|---:|---:|---:|---:|
| Switch-base-8 /16 MiB | 1.4759× | 0.9510× | 1.1305× | 1.4547× | 1.4702× |
| Switch-base-8 /9 MiB | 1.4759× | 1.0134× | 1.1305× | 1.4547× | 1.4702× |
| Switch-base-16 /16 MiB | 1.4756× | 1.1308× | 1.1309× | 1.4548× | 1.4703× |

Switch-base-8的16 MiB C-open完整resident recall为18.67%，9 MiB为0；
虽然前者就绪hit更多，总时间却更慢。未来流量、队列位置和关键partial descriptor等待
决定净收益，不能只用容量/ready hit单一指标预测延迟。增大OCM也不保证既定策略单调更快。
两个Switch的Router耗时不同，burst/FIFO相位效应会被当前active-hit队列行为放大，
不将这些微时序敏感点描述成普遍模型规律。

## 综合结论

1. “只把双候选准确率设成84.4%”不是FLAME预测条件的充分对照：首候选质量与
   实际prefetch窗口需要一起匹配。受控首候选75.8%组能出现1.457×，接近但不等于论文1.4834×。
2. 原先Switch的约1.164×来自内存加载占baseline比例较低；在本轮较平衡供数点、
   较高首候选质量下，完整Expert预取可以明显受益。不是必须引入剪枝才可能有收益。
3. 取消屏障只是让请求更早获得admission机会，demand优先可能已使其DMA等待，
   也可能增加未来流量竞争；不能保证回退到不预取的性能或必然优于屏障开启。
4. 当前D相对弱baseline可有较大速度比，但相对已有streaming的净收益小、部分为负。
   不声称tile预取在全部匹配条件下优于完整Expert预取；本轮D固定Uniform，未做allocator优化。
5. 已开始后才被truth命中的partial descriptor的优先级/回队位置值得后续单独修正验证。
   本轮没有更改它或RTL，不能用“获得接近论文速度比”掩盖这个已定位的工程问题。

## 最终验收与文件

- 主扫描：150个配置点、186个seed点、9,300次模拟、31,000个预测时序记录。
- 两个配对排序扫描：各12个配置点、1,200次模拟、4,000个预测时序记录。
- 合计：174个配置点、210个seed点、**11,700次模拟、39,000个预测时序记录**，无失败样本跳过。
- 完整回归：**437 passed**；新增7项覆盖率、确定性、驻留诊断、放行与分母测试。
- 产物验证：每组六case配对完整；payload计算周期完全相同；OCM不超容量；
  speculative=useful+wasted；A/B零speculation；预测source因果成立；
  主预测释放延迟均为allocator的8 cycles，不再等待上一层权重完成。
- 高/低首候选组的truth和candidate集合逐项相同；较高组保留较低组已有rank0 hit；
  Top-1分别379/500和211/500，两组Top-2均422/500。
- 三个结果目录的源码及配置SHA256均校验一致，PNG/PDF及CSV全部生成。

主要修改：`examples/mac_four_cases.py`只增加可选诊断返回，原默认输出及数值行为不变；
新增实验驱动、三个YAML和定向测试，更新本报告及README。没有修改模拟器核心或RTL。
复现命令见上文，源参数、所有trace及逐点统计均保存在各结果目录。
