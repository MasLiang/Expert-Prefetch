# Tile粒度：Streaming最优与预取最优是否相同？

## 1. 范围与结果概述

2026-10-08 使用当前 `UnifiedSequenceSimulator` 完成 **3,150次模拟、210个配置点**。
保持headroom workload不变，单独扫描tile宽度，并取六个原带宽切片；没有修改模拟器、DMA算法或RTL。

结论：**确实出现不同的最优粒度，但“增量speedup最大”与“总延迟最低”也必须区分。**
在6个带宽×2种质量×3种固定预取策略的36条切片中：

- 36/36：B/D最大tile与stream-only最优tile集合不相交。
- 29/36：预取后总latency最优tile与stream-only最优集合不相交。

这不是36个独立统计样本，也不证明所有模型如此；有相同预算/计划的等效策略点。
这些是离散测试点的事后最优，不是全局最优或部署策略。

## 2. 固定配置

H=128，**F始终256**，E=16，Top-1，INT8 SwiGLU；5个连续MoE层、单token、首层冷启动。
固定OCM384KiB；32ops/cycle、单逻辑顺序GEMM Engine；4通道、200MHz。
DMA/GEMM startup各8cycles，非线性64elements/cycle。
矩阵fragment不可抢占，demand优先，不启用burst仲裁；控制资源使用默认研究profile。
这些是合成有效参数，不是FPGA板卡吞吐实测。

预测为前一层Router done+20cycles，提前一个MoE层，无上一层全部权重屏障。
始终一个候选；accuracy=1为perfect prediction，0.9为受控90% Top-1命中，**不是实测神经predictor**。
复用headroom相同的3个seed×每seed5个保存样本；每seed20次决策分别20/18次hit。

带宽取rho=0.5/0.9/1/1.5/2/4，每通道B/cycle=2/3.6/4/6/8/16。
rho=2Baggregate/Caggregate是payload比例，不含startup和仲裁。

## 3. tile几何与byte大小

改变F_tile时，groups=F/F_tile随之变化；不能固定groups而改变F。

| F_tile | groups/Expert | GU tile | DOWN tile | 一个GU+DOWN logical tile |
|---:|---:|---:|---:|---:|
| 8 | 32 | 2KiB | 1KiB | 3KiB |
| 16 | 16 | 4KiB | 2KiB | 6KiB |
| 32 | 8 | 8KiB | 4KiB | 12KiB |
| 64 | 4 | 16KiB | 8KiB | 24KiB |
| 128 | 2 | 32KiB | 16KiB | 48KiB |

整Expert始终96KiB，未因tile改变而改变工作量。
本轮采用用户建议的F_tile序列；未另跑独立的1/2/4/8/16/32KiB GU序列。
注意“1KiB tile”在F_tile=8时指DOWN而非GU，不能混用矩阵定义。

## 4. 三种预取与OFF

- OFF：同tile配置的stream-only baseline。
- LIGHT：只请求候选的首个GU tile。深度始终1，**byte预算随tile增大**。
- GU32：固定请求32KiB GU权重；depth=128/F_tile，所有tile宽度预算和F覆盖一致。
- FULL：请求候选整Expert的GU+DOWN，96KiB目标。

所有策略计算都streaming，不要求整Expert到齐；FULL不是whole-ready FLAME缓存路径。
实际请求可能受truth cancel、DMA竞争和容量影响，目标预算不是已传输或resident bytes。
每个tile×带宽仅运行一次OFF，不因accuracy或预取策略重复baseline。
GU32用于区分“LIGHT更宽意味着预取量更多”的混杂；不能把LIGHT跨粒度结果称为纯预算不变消融。

## 5. 三张图

- [A：stream-only latency](../results/prefetch_tile_size_20261008/A_streaming_baseline.png)
- [B：预取增量B/D](../results/prefetch_tile_size_20261008/B_prefetch_incremental.png)
- [C：各策略总latency](../results/prefetch_tile_size_20261008/C_overall_latency.png)

每张均有6个带宽子图，PNG/PDF双格式。
线是15序列的合并周期均值；B/D是同tile OFF总cycles / policy总cycles，CSV另存seed范围。
没有画事后最优包络冒充一个部署策略。

## 6. 关键代表点：rho=0.9、LIGHT

| F_tile | OFF平均cycles | q=90%平均cycles | q=90% B/D | perfect平均cycles | perfect B/D |
|---:|---:|---:|---:|---:|---:|
| 8 | 38,285 | 37,741.4 | 1.01440× | 37,681 | 1.01603× |
| 16 | **37,450** | 36,395.2 | 1.02898× | 36,278 | 1.03231× |
| 32 | 37,940 | **35,862.8** | 1.05792× | 35,632 | 1.06477× |
| 64 | 40,020 | 36,030.4 | 1.11073× | **35,572** | 1.12504× |
| 128 | 44,700 | 37,293.6 | **1.19860×** | 36,380 | **1.22870×** |

因此90%时：

- stream-only最优：F_tile=16。
- 同tile B/D最大：F_tile=128。
- 带预取总体最快：F_tile=32。
- 对公平优化后的baseline，37,450/35,862.8=**1.04426×**，不是1.19860×。

perfect时LIGHT总体最优64，相对最佳OFF为37,450/35,572=1.05279×。
这说明仅优化speedup比值会奖励较慢的baseline。**C图比B图更接近最终设计目标。**

其他代表最佳粒度：

| rho | OFF最佳tile | 90% LIGHT整体最佳 | perfect LIGHT整体最佳 | perfect FULL整体最佳 |
|---:|---|---:|---:|---:|
| 0.5 | 16/32并列 | 32 | 64 | 128 |
| 0.9 | 16 | 32 | 64 | 128 |
| 1 | 16 | 16 | 16 | 128 |
| 1.5 | 16 | 32 | 32 | 64 |
| 2 | 32 | 32 | 64 | 64 |
| 4 | 32 | 64 | 64 | 64 |

**最优不同不代表预取必胜。** rho=1、perfect FULL在128宽达到1.20759×同tile收益，
但总cycles34,664仍慢于最佳OFF的34,080，最佳对最佳仅0.98315×。

## 7. setup、请求数与计算startup

OFF每个序列的必需权重始终491,520bytes，started矩阵DMA请求数随tile变化：

| F_tile | started请求/序列 | rho=0.9 DMA setup fraction | OFF payload utilization |
|---:|---:|---:|---:|
| 8 | 320 | 6.93% | 89.16% |
| 16 | 160 | 3.60% | 91.14% |
| 32 | 80 | 1.84% | 89.97% |
| 64 | 40 | 0.93% | 85.29% |
| 128 | 20 | 0.47% | 76.36% |

小tile增加DMA setup和GEMM startup次数；大tile减少固定开销，但推迟首次可算并降低overlap。
两者形成baseline的中间最优。GEMM payload量不变、startup数改变，所以也必须看AIE payload利用率，
而不是将含startup的allocated利用率上升一概解释为更高算术效率。

比如rho=4的8宽DMA setup占25%，128宽约2.04%；setup影响依赖带宽，不能只看tile bytes。
本轮没有真实DDR事务或宽MAC流水的校准。

## 8. safe-slack fit ratio的精确定义

这是**事后、baseline反事实诊断**，不是在线策略使用的未来信息或安全准入保证。

对每个预测目标：

1. 取OFF时间线的窗口：[前一层Router done+20，目标层Router resolve)。
2. baseline已issue但未完成的demand区间不可用，含queued和in-flight，不只减DMA busy。
3. 在剩余空档中判断能否装下一个GU DMA：startup+ceil(GUbytes/(4×BW))。
4. `safe_slack_fit_ratio`=能容纳一条GU请求的空档数/全部空档数；无空档记0。
5. 另存按空档cycles加权的`safe_slack_cycle_fit_ratio`，避免仅靠大量小空档解释结果。

所有请求striped跨4通道，空档需要四通道同时可用；这套定义不能直接套round-robin多channel。
不检查该反事实请求的OCM保留能力，也不保证整预取plan都能放进窗口，故“safe”只是此声明的baseline demand-free口径。
现实预测策略改变后，真实demand到达时序也会改变。

rho=0.9时，8/16/32宽fit ratio均1，64/128宽为0，但大的LIGHT有更高B/D。
原因之一是提前推进、跨Router的部分传输也可有收益；**完整装进Router前空档不是收益的必要条件**。
GU32同点在小tile虽fit=1仍负收益，因为fit针对一条请求，不是整个32KiB计划。
不能据此直接设计fit=1就允许全部预取的策略。

raw CSV中slack字段只对OFF测量，其他policy为0占位；comparison里所有policy使用配对OFF测得的slack。
其他metric（请求/阻塞/利用率）均是各策略自身真实执行的读数，不是baseline反事实。

## 9. metrics口径与验证

已记录用户要求的全部metrics：stream-only latency、perfect/90% speedup、DMA setup fraction、
payload utilization、safe-slack fit ratio、请求数、demand blocking、AIE utilization。

- latency：inference makespan，不把投机DMA尾部当推理时间；另存drain_cycles。
- setup fraction：各通道setup占用cycles / 各通道busy cycles，观察到全部fragment启动。
- payload utilization：transferred bytes / (4×配置每通道速率×资源drain时间)。
- request_count：真正started的矩阵DMA descriptor，不是AXI beats或四通道参与次数。
  issued数、cancel-before-start数和channel intervals数另外保存。
- demand_blocking_cycles：主模型future speculation引起的demand阻塞汇总，可能重叠。
  另存critical_future_blocking_cycles，均不能直接与idle相加为总时间。
- AIE utilization：单逻辑GEMM资源allocated（含startup）/推理时间；另存payload利用率。
  “AIE”是原API字段，不代表此研究硬件真的有AIE。

只读观测通过实验子类保留已有事件state，不写队列/请求/资源；
F_tile=64与旧headroom的450个重叠样本四项metrics完全一致。
7个新增测试通过：固定几何、预算、区间合并/裁剪、request conservation、确定性、
旧驱动等价、增量speedup最优不等于总latency最优。
全量 **591 passed, 14 warnings**，警告为既有依赖弃用提示。
来源hash、3,150键唯一、byte守恒、OCM容量、利用率范围和三组PNG/PDF检查通过。

## 10. 文件与复现

- 配置：`configs/experiments/prefetch_tile_size.yaml`。
- 驱动：`examples/prefetch_tile_size.py`。
- 测试：`tests/test_prefetch_tile_size.py`。
- 输出：`results/prefetch_tile_size_20261008/`。
  - `summary.json` / `inputs.json`：配置、定义、源hash和保存trace。
  - `per_sample.csv`：3,150行。
  - `comparison.csv`：210行，全部配对metrics。
  - `optima.csv`：36行，分别保留最佳baseline/最佳B/D/最佳overall及并列集合。
  - A/B/C三组PNG/PDF。

```bash
PYTHONPATH=src python examples/prefetch_tile_size.py \
  --workers 8 --output results/prefetch_tile_size_new
pytest -q -o addopts=''
```

驱动拒绝覆盖已有目录。这里只跑固定小headroom workload的离散粒度；
没有扫描真实模型或引入新调度器、不同硬件DMA后端、真实predictor、prefill和板级验证。
