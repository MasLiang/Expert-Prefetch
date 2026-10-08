# M10 RTL / Simulator：统一功能边界契约 v1

## 1. 范围与结论

本轮仅统一观察事件和统计口径，不校准耗时，不改变RTL/XDC、硬件配置、
服务时间参数、主UnifiedSequenceSimulator或Scheduler算法。
新增`alignment_boundaries.py`作为无后端依赖的共用分析器，便于随后主模型复用。
当前接入后端仍是RTL自检平台和M10AlignmentSimulator，不能称主模型已经对齐。

实际8组完整默认配置XSim语义/边界契约检查通过，235个Python测试和52个RTL回归点通过。
每组总周期、原weight_wait和speedup均与上一轮完全一致，26份设计RTL源码hash一致。
本轮未重新综合、运行网表/SDF或PnR，不代表200MHz时序闭合或板级带宽测量。

## 2. 同一事件定义

所有时间是从layer begin handshake起的整数cycle，区间统一为`[start, end)`。
RTL在posedge、NBA写回之前观察当前可见状态；握手事件在valid和ready均有效的edge记录。
状态写回后的可用性在后续首次可见采样点记录，不混用DMA最后beat和payload发布。
不再用running上升边沿推断请求握手，也不把逐拍计数的最后一拍直接当区间长度。

| 事件 | 统一定义 |
|---|---|
| ROUTE_RESOLVED | 全部真实Expert已接收，execution首次可见truth_ready |
| REQUEST_QUEUED | 一个GU或DOWN逻辑stage的descriptor被接受入DMA队列；RTL映射ADMIT |
| MEMORY_SERVICE_START | stage获得非抢占DMA所有权；RTL使用claim_valid，不使用第一个burst_start |
| PAYLOAD_READY | 完整stage的OCM payload首次可见完整驻留，ownership有效且无cancel/quarantine |
| TILE_ISSUE_WINDOW_OPEN | truth已知，未running、未combine，仍有真实工作待调度；窗口首次打开 |
| COMPUTE_REQUEST_ACCEPTED | execution到compute controller的实际valid/ready握手 |
| GU_START / GU_DONE | GU计算启动与完成事件 |
| DOWN_START / DOWN_DONE | DOWN计算启动与完成事件 |
| TILE_RETIRED | execution接受compute_done，推进prefix并消费该tile的时刻 |
| EXPERT_COMBINE_START | Expert最后tile退休，启动gating/combine |
| EXPERT_COMBINE_DONE | combine结果写回后首次可见；RTL不以乘法开始代替最终写回 |

PAYLOAD_READY观察按slot generation维护状态；被取消slot复用时valid不一定先降为0，
不能只在valid下降时清除“已记录”标记。本轮严格生命周期检查发现并修复了这一只读观察器漏记。
当前契约只处理本轮单层、无flush/error fixture；不将同key跨epoch重试混入一个生命周期。

## 3. 统一等待区间

每个tile执行记录与一个issue window配对。前一个tile未退休或Expert combine未完成，
不能打开后续窗口；同Expert必须按prefix顺序执行。

在window打开后，分析器检查所有真实Expert当前下一prefix的GU+DOWN就绪时间，
选择最早可执行pair，避免把“Router序第一个未就绪但其他Expert已就绪”误记为内存等待。

```text
window_open ── 内存等待 ── 任意下一prefix pair可用 ── ready→accept ── 请求握手 ── accept→GU ── GU_START
```

实际某个pair可能早于window驻留；此时内存等待为0，ready→accept从window开始计。
以下恒等式由同一个分析器生成，两端采用相同含义：

```text
issue_window_cycles = memory_wait_cycles + ready_to_accept_cycles
ready_to_gu_cycles = ready_to_accept_cycles + accept_to_gu_cycles
```

`down_done_to_retire_cycles`、`previous_retire_to_window_cycles`和Expert combine单独输出。
前者反映完成结果进入execution的退休路径，后者反映两次发射窗口之间的阻塞。
combine可能包含在retire→next window中，不能把所有诊断量直接相加当layer total。
GU/NL/DOWN正在执行、Router以及最后结果输出路径不属于issue等待。

旧`weight_wait_cycles`、上一轮的`ready_control_wait_cycles`和全部原门槛保留，
新数据放在summary的`canonical`字段及comparison的`canonical_*`字段，避免覆盖历史含义。

## 4. 未建模阶段明确标记，不拟合

M10原来只有固定8-cycle tile_dispatch，未表达请求接受后的独立启动阶段。
默认保持这个总时间，把COMPUTE_REQUEST_ACCEPTED放在GU_START：
**accept→GU = 0，表示collapsed/unmodeled，不表示RTL没有此延迟。**

适配层允许可选`request_accept_to_gu_cycles`划分已有dispatch时间，范围为0到tile_dispatch，
但本轮YAML没有加入或调整它，实际矩阵仍采用默认0。
测试中设2仅验证分段不会改变原timeline/总周期，不是硬件校准值。

原每tile3-cycle阻塞也没有删除。模型TILE_RETIRED仍在DOWN完成时，后续窗口在3cycles后打开；
仅最后tile另标Expert combine区间。因此非尾tile的旧3-cycle阻塞现在可见，
不能因为两端Expert combine合计恰好同为6cycles就认为退休路径也已准确。

## 5. 实际结果

| 激励类别 | 内存等待RTL/Sim | ready→accept RTL/Sim | accept→GU RTL/Sim | ready→GU RTL/Sim |
|---|---:|---:|---:|---:|
| demand_fast / wrong_fast | 115 / 110 | 63 / 256 | 64 / 0 | 127 / 256 |
| demand_gaps / wrong_gaps | 126 / 119 | 63 / 256 | 64 / 0 | 127 / 256 |
| hit及second_hit，两种DDR模式 | 0 / 0 | 63 / 256 | 66 / 0 | 129 / 256 |

fast首tile的实际事件是：

| 边界 | RTL cycle | Sim cycle |
|---|---:|---:|
| issue window打开 | 685 | 684 |
| GU/DOWN pair payload可用 | 800 | 794 |
| compute请求接受 | 802 | 802 |
| GU开始 | 804 | 802 |

首tile统一ready→accept为2/8，accept→GU为2/0。
旧逐拍控制计数包含握手采样拍且在running后停止；所以旧95与模型256不可直接比较。
统一后完整ready→GU累计为127–129/256cycles。

全部32tiles的DOWN done→retire为RTL32cycles、模型0；
前次retire→下次window累计为RTL33、模型93cycles；首window没有前一tile。
这些是新的可定位差异，不是已经校准的参数。

内存等待仍有5–7cycles偏差，超过原4-cycle门槛；旧性能综合门槛保持失败。
本次通过的是统一契约与语义检查，不是完整性能对齐。

## 6. 产物与复现

从仓库根目录运行：

```bash
PYTHONPATH=src python hardware/m10_u200/scripts/compare_u200_default_alignment.py \
  --output results/m10_boundary_alignment
python -m pytest -q
make -C hardware/m10_u200 rtl-test
```

每case新增：

- `rtl_boundary_events.csv` / `simulator_boundary_events.csv`：规范化功能边界。
- `rtl_tile_timing.csv` / `simulator_tile_timing.csv`：32个真实tile的关键事件和等待分段。
- `rtl_stage_timing.csv` / `simulator_stage_timing.csv`：每stage的入队、DMA所有权、payload就绪与时长。
- 原timeline追加`CANONICAL_ISSUE`区间；原事件与旧指标不删除。
- summary保留`canonical`分析结果；comparison报告两端分项值，不把不同口径混为一项。

严格检查缺失/重复事件、负数/非整数cycle、倒序生命周期、pair未就绪提前发射、
prefix违反、未退休就重开window及combine因果。零长度阶段合法，必须明确模型近似。
原总周期10%、等待4cycles和speedup 0.005容差均不变。
`--strict-timing`仍会失败，不用边界契约通过替代性能通过。

后端：Vivado/XSim 2023.2，默认U200配置，debug off、无全量波形/coverage/SDF。
本机Python回归仍为3.9.19，未声称已经验证3.11环境。
结果写入新的`results/m10_boundary_alignment`，不覆盖以前两轮的结果目录。
