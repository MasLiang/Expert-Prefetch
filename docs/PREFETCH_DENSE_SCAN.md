# 预测预取收益转折：密集扫描与多seed验证

## 完成范围

在此前合成相图的基础上完成收益转折加密，不修改主模拟器、RTL或任何调度算法。
本轮使用 `UnifiedSequenceSimulator`，复用上一轮的执行函数。

| 项目 | 数量 |
|---|---:|
| 硬件配置点 | 264 |
| 硬件/策略/质量/有效深度配置点 | 922 |
| 配置/seed汇总点 | 4610 |
| 配置/seed/样本结果 | 23050 |
| 随机种子 | 5 |
| 每seed独立单token序列 | 5 |
| 每序列连续MoE层 | 5 |

本轮每seed有20次预测决策；质量0.5/0.75/0.9/1分别精确命中10/15/18/20次。
truth、错误候选和命中位置随seed变化；相同seed下所有硬件和策略共享truth。
第一层冷启动，没有跨token权重cache复用。

## 配置与加密方式

H=128、F tile=64、E=16、Top-1、INT8 SwiGLU；Expert为2/4/16个F groups。
PE=1/4，每PE为32 ops/cycle，所有PE用于当前GEMM，Expert仍顺序执行。
OCM为1/4个Expert等价容量，4内存通道，200 MHz，DMA和计算startup均8 cycles。
非线性吞吐64 elements/cycle，预测在前一层Router完成后20 cycles可用。
DMA为整矩阵tile不可抢占，actual-byte OCM，不启用burst仲裁。

按既有归一化指标扫描22个rho点：

\[
\rho=\frac{t_{compute,payload}}{t_{memory,payload}}
=\frac{2B_{aggregate}}{C_{aggregate}}
\]

0.1、0.2、0.3、0.4、0.5、0.6、0.75、0.9、1、1.1、1.25、1.5、1.75、2、
2.25、2.5、3、4、6、8、12、16。
对于每个PE数，反推每通道bytes/cycle，因此不是给不同PE使用相同绝对带宽。
所有吞吐为合成有效速率；非整数bytes/cycle也允许，时间由模拟器向上取整为整数cycle。
rho忽略startup、Router、非线性和通道仲裁，不等同于真实设备实测bound分类。

主矩阵固定quality=0.9、GU depth=1，另外运行无预测和true-only all-GU Oracle。
局部消融固定PE=1、OCM=4，选择4/16 groups、rho=0.5/1/1.5/2/4，扫描
quality=0.5/0.75/0.9/1及GU depth=1/2/4/8。
超过Expert大小的深度截断并按有效深度去重；baseline和Oracle也不重复运行。

质量同时改变候选recall和rank-1精度：命中时truth在首位，miss时两个候选均错误。
不是confidence calibration实验。Uniform仍加载两个候选，所以quality=1仍有错误候选。
Oracle只输出truth，但请求全GU；它不是最优调度，也不是严格延迟下界。

## 配对统计

每seed先计算baseline总cycles/当前策略总cycles，再跨seed报告均值、样本标准差、
min/max。另提供所有seed总cycles比 `pooled_speedup`，不混淆平均比值与比值平均。

曲线阴影是5个seed的min/max，不是95%置信区间；局部消融误差棒为seed样本标准差。
部分点5个seed结果完全相同：同质Expert、固定命中次数和对称striped放置会消除
Expert ID或命中位置对makespan的影响。零波动不证明现实部署也无波动。

## 固定策略实测结果

以下固定PE=1、OCM=4个Expert、quality=0.9、GU depth=1；加速比相对无预测streaming。

| rho | 2 groups均值 | 4 groups均值 | 16 groups均值 |
|---:|---:|---:|---:|
| 0.5 | 1.05580× | 1.02855× | 1.00727× |
| 0.75 | 1.15508× | 1.08034× | 1.02064× |
| 0.9 | 1.21221× | 1.11073× | 1.02862× |
| 1.0 | 0.98537× | 0.84810× | 0.95165× |
| 1.25 | 1.13284× | 0.94622× | 0.98445× |
| 1.5 | 1.14482× | 1.02995× | 1.00804× |
| 2.0 | 1.11052× | 1.05480× | 1.01419× |
| 4.0 | 1.05732× | 1.02960× | 1.00759× |
| 8.0 | 1.02985× | 1.01533× | 1.00391× |

2 groups、rho=1的seed范围为0.98254–0.99201×，5个seed都负收益。
4 groups、rho=1的5个seed均为0.84810×；负谷值不依赖单一随机激励。
固定浅预取在大Expert上收益被稀释：16 groups的全部主矩阵采样点未达到1.05×。

在本轮全部预测配置中，2 groups的最高均值为1.21513×，4 groups为1.11228×，
16 groups为1.02903×。这是事后报告最优测试点，不代表一个统一部署策略可达到这些值。

## 为什么出现不连续负谷值：逐请求证据

另外导出同一seed（20261005）、sample=0、4 groups、PE=1、OCM=4、quality=0.9、
depth=1的rho=0.9与1.0时间线，并分别跑对应无预测对照。
开启trace后的makespan与扫描执行函数逐case核对完全一致。

| 事件 | rho=0.9 | rho=1.0 |
|---|---:|---:|
| 第一条下一层GU预取开始 | 7028 | 2720 |
| 第一条下一层GU预取完成 | 8174 | 3752 |
| 首条被阻塞真实请求 | 下一层DOWN tile0 | 当前层DOWN tile1 |
| 请求就绪/阻塞开始 | 8140 | 2721 |
| 该请求阻塞cycles | 34 | 1031 |
| 预测序列makespan | 35572 | 43096 |

在rho=1时，cycle2720没有待启动demand，DMA可以启动下一层预取；
仅1 cycle后当前层真实demand到来，整tile不可抢占，因此必须等到3752。
该预取最终还是命中的：**即使预取正确，也可能先拖慢当前层。**
rho=0.9时，相对时序不同，首条预取推迟至当前层尾部才开始。
这提供了具体的非单调调度证据，不是“更高带宽必然更慢”的硬件物理定律。
目前没有证明全部负谷值都由同一个事件解释，也没有进行burst/保守准入的反事实消融。

在4 groups、PE=1、OCM=4、rho=1处，局部质量/深度消融为：

| quality | depth=1 | depth=2 | depth=4 |
|---:|---:|---:|---:|
| 0.5 | 0.81709× | 0.82115× | 0.88026× |
| 0.9 | 0.84810× | 0.88127× | 0.99882× |
| 1.0 | 0.85623× | 0.89726× | 1.02995× |

提高预测质量有帮助，但固定浅depth下仍无法保证净收益。
此表保持硬件和算法不变；质量与深度是显式实验变量，不修改scheduler逻辑。

## 稳定通过的采样段

`benefit_regions.csv`以5个seed的**最小**speedup是否达到阈值为判断条件，
只连接相邻且通过的采样点，失败谷值会切断采样段。
例如PE=1、OCM=4、quality=0.9、depth=1，所有seed均达到1.05×的采样点：

- 2 groups：{0.5,0.6,0.75,0.9}，以及{1.25,1.5,1.75,2,2.25,2.5,3,4}。
- 4 groups：{0.75,0.9}，以及{2,2.25}。
- 16 groups：无。

这些不是闭区间保证，不推断未采样点，不等于统计置信度，也不能直接外推到Top-K、
其他tile尺寸或DMA粒度。更准确的结论是：**当前执行规则下存在多个收益岛；
纯算力/带宽比例不足以表达充分条件，必须把预取覆盖量与不可抢占窗口一起考虑。**

## 验证与产物

全套398 tests通过，包含去重、独立seed校验、配对统计、采样段不跨失败点、
diagnostic trace/makespan一致性和谷值请求时序检查。
结果检查通过：完成状态、922/4610/23050行数、样本键唯一、OCM不超容量、
speculative=useful+wasted、无预测speculative为0、Oracle wasted为0、源码SHA256匹配。

输出位于 `results/prefetch_dense/`：

- `inputs.json`、`summary.json`：激励、配置、来源校验码。
- `per_sample.csv`、`per_seed.csv`、`comparison.csv`：原始与配对统计。
- `benefit_regions.csv`：稳定通过阈值的采样段。
- `dense_curves.png/pdf`：固定策略曲线及seed范围。
- `quality_depth_sensitivity.png/pdf`：质量/深度局部消融。
- `diagnostics/*.json`：四个代表case的timeline、请求和阻塞证据。

复现：

```bash
python examples/prefetch_dense_sweep.py --workers 4
python examples/prefetch_dense_diagnostics.py
pytest -q -o addopts=''
```

下一步若优化策略，应该对这里识别出的“短空档启动长speculative DMA”进行干预，
例如小burst仲裁或更保守的准入，并在同样激励上比较；本轮没有实现这些新机制。
