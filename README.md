# edge-moe-sim

## 单 token 源码 release 候选（2026-10-08）

当前 Simulator / U200 RTL 的源码发布范围、文件清单、复现与Git状态见
[发布准备说明](docs/SINGLE_TOKEN_RELEASE.md)。运行
`python scripts/prepare_single_token_release.py` 生成独立源码快照与SHA256清单；
不打包Vivado生成工程、凭据目录或完整历史实验结果，不删除本地文件。
本候选不声称板级部署或200MHz时序闭合。

## 真实 MoE 模型尺寸 pilot

新增 OLMoE-1B-7B、Mixtral-8×7B 和 Mixtral-8×22B 的官方配置摘录与主模型实验：
`python examples/real_moe_pilot.py --plot`。
比较四个连续 MoE 层、有限 OCM 和不同计算吞吐；真实架构尺寸搭配合成路由/预测，
不是完整 LLM 推理、量化准确率验证或板上实测。配置、复现与部署限制见
[真实模型 pilot](docs/REAL_MOE_MODEL_PILOT.md)。
完整 Expert 目标预取补跑：`python examples/real_moe_full_expert.py`，
容量截断与完整传输检查见 [补充比较](docs/REAL_MOE_FULL_EXPERT_COMPARISON.md)。

## U200 RTL 与主模型共同激励（M10.1）

[当前 RTL 工程](hardware/m10_u200)
包含真实共享 INT8 Router/Expert MAC、四 DDR AXI 通路、OCM payload 与多层跨层预取。
当前行为级验证不代表板级实测或 200 MHz 时序闭合。
运行 `make -C hardware/m10_u200 alignment-main` 使用主模型 `UnifiedSequenceSimulator`
进行共同夹具诊断。`M10AlignmentSimulator` 对当前 RTL **已失效**，默认拒绝使用，
仅保留显式 opt-in 的历史回归。
当前 fixed-slot、pair admission、tile ready-first 与请求重试/FIFO 的配置及验收范围
见 [执行规则统一](docs/M101_EXECUTION_RULE_ALIGNMENT.md)。
新增有限 descriptor/FIFO credits、burst 边界仲裁和控制时间诊断：
`make -C hardware/m10_u200 alignment-control`；主模型敏感性扫描：
`make -C hardware/m10_u200 control-sensitivity`。
验收与剩余周期差异见 [控制资源对齐报告](docs/M101_CONTROL_RESOURCE_ALIGNMENT.md)。
九组执行规则比较通过，**尚非完整资源或逐周期性能对齐**。
最新接口时间补充和四 DDR 空拍优化使用
`make -C hardware/m10_u200 alignment-interface`，见
[接口时间与 RTL 优化](docs/M101_INTERFACE_TIMING_OPTIMIZATION.md)。
等待来源与临界窗口检查：`make -C hardware/m10_u200 boundary-validation`，见
[边界等待验证](docs/M101_BOUNDARY_WAIT_VALIDATION.md)。
历史配置发现六个晚预测点分类不一致。最新逐 descriptor 时序补齐使用
`make -C hardware/m10_u200 descriptor-boundary`，见
[逐 descriptor 可见时序与 admission 修正](docs/M101_DESCRIPTOR_VISIBILITY_ALIGNMENT.md)。
修正配置消除了六点分类差异；仍不是所有内部握手/资源峰值的逐周期等价。
层间实际 ingress/控制资源退休与 FIFO credit/occupancy 拆分使用
`make -C hardware/m10_u200 ingress-boundary`，见
[层间退休与 FIFO 统计](docs/M101_INGRESS_FIFO_ACCOUNTING.md)。
四路 DDR4 controller/RDIMM 数字模型验证入口为
`make -C hardware/m10_u200 ddr4-validation`；校准与预加载不计入 decode 周期。
见 [DDR4 控制器验证](docs/M101_DDR4_CONTROLLER_VALIDATION.md)。

`edge-moe-sim` 是一个 trace-driven、event-driven 的通用 MoE 权重调度模拟器。它同时支持原有 stage-level 模型、tile-level weight streaming，以及 M8 prediction-aware resource allocation，用于研究有限 OCM、带宽和 DMA channels 下的 Expert tile budget 分配、传输顺序与计算重叠。核心模型不绑定 VEK280。

Milestone 5 新增同一 token 的连续 MoE layer timeline，用来研究在 source MoE layer 期间预测下一个 MoE layer、并提前读取 target Gate-Up 启动 tiles 的系统收益。

> **所有 bundled 性能参数均为 synthetic placeholder，不是 VEK280 实测值、标称值或理论峰值。**

## Resource-aware tile scheduling（M8）

M8 把问题从“选择多少个 Expert”推广为“给每个候选 Expert 分配多少个 F-axis logical tiles”。新增四个可插拔策略：`no_prefetch`、`full_expert`、`uniform_tile` 和 `ranked_tile`。计划仍由原有 `MultiChannelMemory`、有限 `CapacityOCM`、`AIEPool` 和事件依赖执行，因此没有复制或绕过 M7 资源语义。

完整设计和结果见 [docs/TILE_LEVEL_SCHEDULING.md](docs/TILE_LEVEL_SCHEDULING.md)。运行 1,296-point synthetic sweep：

```bash
python examples/m8_tile_scheduler_experiment.py
```

输出位于 `results/m8_tile_scheduler/`，包括 CSV、summary、artifact validation 及三组 PNG/PDF 图。

## Hardware-friendly online scheduling（M9）

M9 新增定点 `BudgetProportionalScheduler`、零除法的 `HardwareAwareGreedyScheduler`、离线 exhaustive latency Oracle，以及 comparisons/multiplications/divisions/state entries 的定性复杂度模型。Oracle 只用于上界分析，绝不代表在线 PL 实现。

```bash
python examples/m9_hardware_scheduler_experiment.py
```

设计、限制和结果解释见 [docs/HARDWARE_FRIENDLY_TILE_SCHEDULER.md](docs/HARDWARE_FRIENDLY_TILE_SCHEDULER.md)。输出位于 `results/m9_hardware_scheduler/`。这些结果是 simulator methodology validation，不是 RTL/HLS、FPGA synthesis 或硬件验证。

## 支持范围

Stage 和 Tile 两种 granularity 都支持 S0 `demand_serial`、S1 `exact_pipeline`、S2 `predictive`、S3 `oracle` 和 S4 `ideal`，并共享单 AIE、单 non-preemptive DMA、独立 Nonlinear Engine、无限 OCM、ReadyFirst Expert 调度及 demand/speculative commit/cancel。

默认保持旧行为：

```yaml
simulation:
  granularity: stage
```

Tile mode：

```yaml
simulation:
  granularity: tile
tile:
  tile_size_bytes: 4096
  min_tiles_per_matrix: 1
```

## Tile partition

`FixedByteTilingPolicy` 按最大 byte 数切分；`FixedTileCountPolicy` 用于小型解析测试。最后一个 tile 可以更小。每个矩阵严格保证：

```text
sum(tile.size_bytes)  == matrix weight bytes
sum(tile.compute_ops) == matrix GEMM operations
```

Operation 先按 tile byte 比例做整数分配，最后一个 tile 接收精确余数，不会因逐 tile rounding 重复计算工作量。

## AIE streaming

每个矩阵依次执行 tile 0、1、…。Tile 0 resident 后即可启动 AIE，DMA 同时传输后续 tile。Gate-Up 全部结束后才执行完整 Nonlinear；Down 只能在 Nonlinear 完成后计算。

```text
tile 0: matrix_startup_cycles + switch + ceil(tile_ops / rate)
tile j: tile_pipeline_cycles + ceil(tile_ops / rate), j > 0
```

缺少新字段时，`matrix_startup_cycles=startup_cycles` 且 `tile_pipeline_cycles=0`，所以单 tile 矩阵与 stage 模式完全兼容。

Tile starvation 只计算：

```text
max(0, tile_ready_cycle - compute_eligible_cycle)
```

并分为 first-tile、inter-tile、Gate-Up 和 Down stall。Router、Top-K/Dispatch、Nonlinear、前一 tile 计算以及 AIE 执行其他合法操作的时间不会计入 memory stall。

## Tile OCM 与 DMA

Tile key 为 `(layer_id, expert_id, matrix_type, tile_id)`。每个 tile 独立经历 queued、in-flight、resident 状态，并独立支付：

```text
startup_cycles + ceil(tile_bytes / bytes_per_cycle)
```

所以小 tile 可以更早启动 AIE，但也可能因更多 DMA startup 而更慢。旧单层/sequence OCM 提供 resident count/fraction、first-tile ready 和 contiguous ready prefix 查询且保持无限容量；Milestone 6 advanced mode 另行提供有限 OCM 与 speculative eviction。

## Tile Critical Path

`TileCriticalPathAwareScheduler` 依次优先当前矩阵最早缺失 tile、当前 Expert Down prefix、ReadyFirst 下一 Expert Gate-Up prefix、下一 Expert Down，再按 demand、criticality、tile ID、token count、gating weight、score 和 Expert ID 确定性排序。Stage 模式继续使用原来的矩阵级 scheduler。

## Progressive predictor 和 policies

新 JSONL 格式：

```json
{
  "request_id": 0,
  "token_position": 0,
  "layer_id": 0,
  "ready_offset_cycles": 0,
  "candidates": [
    {"expert_id": 2, "score": 0.9, "gate_up_tiles": 3, "down_tiles": 0}
  ]
}
```

旧 `candidate_ids` / `candidate_scores` 格式仍兼容。实现的策略：

- `FixedDepthPolicy`
- `FixedByteBudgetPolicy`，只分配完整 tile
- `ProgressiveRankPolicy`
- `PredictorSpecifiedPolicy`

Tile speculative 请求在 resolve 时逐 tile commit/cancel；in-flight true tile 不会重复 demand，false active tile 不可抢占。

## OracleSet 与 OracleOptimalDepth

OracleSet 知道真实 Expert 集合，但仍使用普通 policy/scheduler。`optimize_oracle_depth()` 针对 `K<=4`、每 Expert Gate-Up `<=8` tiles 的小 synthetic case 枚举 byte budget 内的 Gate-Up prefix depths。搜索包含 depth 0，并输出 `optimal_depths`、`optimal_cycles` 和 `evaluated_combinations`。

## Multi-layer timeline

`Simulator.simulate_event()` 保留单层隔离行为；`SequenceSimulator.simulate_layer_sequence()` 使用一份贯穿 sequence 的：

- global integer cycle 与 `EventQueue`
- `ResourceManager`、单 AIE、单 non-preemptive DMA、Nonlinear Engine
- demand/background/speculative queues
- 无限容量 OCM resident state
- timeline 与计数器

因此 target speculative DMA 可以在 source expert、inter-layer compute 和 target Router 期间传输，也可以保持 active 或 resident 跨越 layer boundary。实现不是把多个从 cycle 0 开始的单层结果相加。

`ModelConfig.moe_layer_indices` 定义通用 MoE 映射；例如 `(3, 7, 11)` 对应 `3→7→11`，不假设 target 是 `source_layer_id + 1`。`LayerSequence` 要求同一 request/token、严格递增且属于该映射。

## Inter-layer resource segment

`InterLayerSegment` 将 Attention/Norm/Dense 等未展开算子抽象为 AIE、Nonlinear、Controller 和 background DMA 工作。三种计算资源从 source layer done 并行启动；segment 结束是这些资源与 background DMA 的 makespan。background DMA 的默认优先级低于 current-layer demand、高于 future speculative；没有 background DMA 时，future prefetch 可继续占用 inter-layer DMA 窗口。

## Cross-layer prediction semantics

`PredictionEvent` 明确记录 source/target token 和 layer、实际 feature ready cycle、predictor latency 与 absolute prediction ready cycle。支持触发点：

```text
source_layer_start, source_router_done, source_resolve,
source_expert_start, source_gate_up_done, source_expert_done, custom_offset
```

事件只在对应 source milestone 真正发生时生成，不能在 sequence 开始时预知未来硬件 cycle。future target speculative tile 只能由 target layer exact Router resolve commit/cancel；source resolve 没有该 authority。

DMA 默认优先级是：

```text
current-layer demand > inter-layer background > future-layer speculative
```

已开始的 future burst 不可抢占。随后到达的 demand 等待时间计入 `total_demand_blocked_by_future_speculative_cycles`。

## SyntheticCrossLayerPredictor

这个类是 **oracle-informed synthetic generation for sensitivity analysis only**。它读取 target routing truth 来构造具有受控统计质量的候选，绝不是可部署 predictor，也不读取未来 OCM/DMA state、Expert 执行顺序或 OracleOptimalDepth 结果。

两种生成模式：

- `hit_distribution`：先采样整个 Top-K 的 hit count，能直接控制 zero/partial/full-hit 事件分布。
- `independent_recall`：每个真实 Expert 独立按 `target_recall` Bernoulli 采样；实际 mean recall、any-hit 和 full-hit 由数据集统计，不假设等于配置概率。

支持 `true_first`、`random`、`noisy_score`、`perfect_router_order` ranking；false candidates 从非真实 Experts 中均匀无放回选择。候选不足或分布非法会抛出 `ValidationError`。相同输入与 seed 会产生完全相同的 candidates、scores、summary 和 timeline。

连续场景包括 `sequence_exact`、`synthetic_cross_layer`、`cross_layer_oracle_set`、`cross_layer_oracle_optimal` 和 `sequence_ideal`。OracleOptimalDepth 对小 sequence 枚举预算内真实 Expert GU prefix，按 sequence makespan 选择确定性最优解。

## 安装与测试

需要 Python 3.11+：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

## CLI

Tile Predictive：

```bash
edge-moe-sim run \
  --model configs/model.yaml \
  --hardware configs/hardware.yaml \
  --trace examples/routing_trace.jsonl \
  --scenario predictive \
  --predictor examples/predictor_trace.jsonl \
  --granularity tile \
  --tile-size-bytes 4096 \
  --prefetch-policy fixed_byte_budget \
  --prefetch-byte-budget 16384 \
  --dma-scheduler tile_critical_path \
  --expert-scheduler ready_first \
  --output results/tile_predictive/
```

Stage mode 使用 `--granularity stage`，也是默认值。

Sweep：

```bash
edge-moe-sim sweep \
  --model configs/model.yaml \
  --hardware configs/hardware.yaml \
  --trace examples/routing_trace.jsonl \
  --scenario oracle \
  --granularity tile \
  --tile-sizes 4096,16384,65536,262144,1048576 \
  --byte-budgets 4096,8192,16384,32768,65536 \
  --ready-offsets 0,7,14,21,28 \
  --output results/sweep/
```

## M9.1 realistic-scale characterization

M9.1 adds three representative synthetic MoE workload scales and three
theoretical/synthetic accelerator mappings without changing simulator or M9
scheduler semantics. Run the complete 3,024-row matrix and regenerate all CSV,
PNG, and PDF artifacts with:

```bash
python examples/m91_realistic_evaluation.py
```

See [the M9.1 evaluation report](docs/M91_REALISTIC_WORKLOAD_EVALUATION.md) for
configuration provenance, quantitative results, validation, and the explicit
boundary between supported simulator conclusions and unclaimed real-hardware
performance.

## M9.2 criticality-aware scheduling

M9.2 adds a hardware-friendly criticality-weighted allocator and a bounded
Fixed Prediction Oracle that preserves the original candidate set and ranking.
It decomposes Full-Oracle distance into prediction and allocation losses without
changing the event simulator or earlier scheduler semantics:

```bash
python examples/m92_criticality_scheduler.py
```

The [M9.2 report](docs/M92_CRITICALITY_AWARE_TILE_SCHEDULING.md) documents the
integer score formula, Oracle search boundary, full 1,296-row experiment, five
figures, and the measured negative result for this particular criticality
heuristic.

## M9.3 predictor sensitivity

M9.3 holds all schedulers fixed while sweeping nine deterministic predictor
quality levels and four rank-preserving confidence-calibration modes over the
complete M9.1 workload/hardware/OCM/bandwidth matrix:

```bash
python examples/m93_predictor_sensitivity.py
```

See [the M9.3 report](docs/M93_PREDICTOR_SENSITIVITY_ANALYSIS.md) for quality
thresholds, calibration variance, scheduler robustness, the six PNG/PDF
figures, and limitations of the synthetic predictor model.

输出 `sweep.csv`，不生成图。

连续三层 synthetic example：

```bash
edge-moe-sim run-sequence \
  --model configs/model.yaml \
  --hardware configs/hardware.yaml \
  --simulation configs/sequence_simulation.yaml \
  --tile configs/tile.yaml \
  --inter-layer configs/inter_layer.yaml \
  --trace examples/multilayer_routing_trace.jsonl \
  --scenario synthetic_cross_layer \
  --synthetic-predictor configs/synthetic_predictor.yaml \
  --output results/cross_layer/
```

Cross-layer Cartesian sweep：

```bash
edge-moe-sim sweep-cross-layer \
  --model configs/model.yaml --hardware configs/hardware.yaml \
  --simulation configs/sequence_simulation.yaml --tile configs/tile.yaml \
  --inter-layer configs/inter_layer.yaml \
  --trace examples/multilayer_routing_trace.jsonl \
  --synthetic-predictor configs/synthetic_predictor.yaml \
  --candidate-counts 2,4 --recalls 0,0.25,0.5,0.75,1 \
  --predictor-latencies 0,10,50,100 \
  --source-points source_router_done,source_resolve,source_expert_start,source_expert_done \
  --prefetch-depths 1,2,4 --inter-layer-cycles 0,100,500,1000 \
  --seeds 1,2,3,4,5 --output results/cross_layer_sweep/
```

`summary.json` / `per_sequence.jsonl` / `timeline.csv` 包含 prediction lead、source/target readiness、future useful/wasted/canceled bytes、target stall 和 active future burst 阻塞 demand 的时间。Timeline 还携带 source/target layer/token、prediction stage/scope/id。

## Synthetic experiments

```bash
python examples/run_synthetic.py
python examples/run_cross_layer_synthetic.py
python examples/run_cross_layer_sweeps.py
```

后两条命令生成 `results/synthetic_cross_layer/` 与七个 focused sweep CSV。当前 bundled cross-layer synthetic placeholder 结果为：

```text
Sequence Exact                  6216 cycles
Synthetic configured recall 0  6376 cycles
Synthetic configured recall .5 6228 cycles
Synthetic configured recall .75 6228 cycles  (observed recall .5 in this tiny sample)
Synthetic configured recall 1  5784 cycles
Cross-layer OracleSet           5655 cycles
Cross-layer OracleOptimalDepth  5488 cycles
Sequence Ideal                  1778 cycles
```

Recall 越高不保证延迟严格单调：candidate ranking/count、错误 non-preemptive burst、source demand 竞争、第一个关键 Expert 是否命中和 target ReadyFirst 顺序都可能改变结果。这里 recall .5 与 .75 的单样本结果相同正是统计采样，不是模拟器强制单调。

旧单层脚本仍生成 `results/synthetic/`、`results/synthetic_tile/` 和 `results/sweeps/`。一个历史 synthetic placeholder 结果示例：

```text
Stage Exact                 1741 cycles

Tile Ideal                   506 cycles
Tile OracleOptimalDepth     1818 cycles
Tile OracleSet              1818 cycles
Tile Predictive             1818 cycles
Tile Exact                  1854 cycles
Tile Demand Serial          2282 cycles
```

这个配置体现 4 KiB tile 的多次 DMA startup，因此 Tile Exact 可以比 Stage Exact 慢。Predictive 也不保证快于 Exact；错误 candidate、晚 ready 或 false active tile 都可能抵消收益。

## 尚未实现或仍受限

- 跨 layer 持久 cache reuse、LRU 等复杂 OCM policy
- 真实 prefill trace 和 continuous batching
- 每 channel 多个同时传输的 DMA、DMA 抢占
- DDR bank/NoC、功耗、真实 VEK280 calibration
- Previous-token/heatmap/Fate/MLP/LUTNet predictor
- 真实 token t→t+1 时序、真实 cross-layer/previous-token/MLP/LUTNet predictor 训练
- 多阶段 temporal + spatial refinement
- Chrome Trace

## Milestone 7：Unified Cross-Layer Advanced Simulator

`UnifiedSequenceSimulator` 是独立的新入口，保留旧 `Simulator`、`SequenceSimulator` 和 `AdvancedSimulator`。它在一个全局事件队列上共享 M6 的 multi-channel memory、AIE compute-group pool、finite OCM 和 F-tile dataflow，并采用 M5 的 source milestone → predictor latency → future target request → target Router commit/cancel 因果语义。相邻层还可通过 `InterLayerConfig` 注入明确标记为 synthetic 的 AIE、nonlinear/controller 和 background-DMA 工作。详见 `docs/UNIFIED_SEQUENCE_SIMULATOR.md`。

## Milestone 6：Batch、Multi-channel、Multi-AIE 与 Roofline

Milestone 6 以增量模块实现，不改变旧单 AIE/single-DMA simulator 的语义。旧 `HardwareConfig` 可通过 `HardwareProfile.from_legacy()` 自动映射为一个 memory channel 和一个 AIE unit。

主要模块：

- `hardware_profile.py`：显式 theoretical/effective bandwidth、compute peak 与有限 OCM。
- `batch_routing.py`：逐 token Top-K routes、`n_e` 聚合和 reuse statistics。
- `f_tile.py`：沿 F 维的物理 tile group；每组执行 `GU weight → GU → nonlinear → Down weight → Down → accumulate`。
- `hardware/multi_channel_memory.py`：独立 channel、single/round-robin/striped placement、burst coalescing和三档优先级。
- `hardware/aie_pool.py`：capacity-based AIE allocation 和并行 Expert compute。
- `hardware/capacity_ocm.py`：有限 byte capacity、running protection、false/low-priority speculative eviction。
- `advanced_simulator.py`：batch/F-tile shared-event DAG。
- `roofline_metrics.py`：board/configured Roofline、ridge point和 regime classification。
- `roofline_prefetch_dse.py`：breadth/depth/AIE/parallelism/bandwidth grid 与 latency-byte Pareto。

Batch 路由中，多个 token 共享同一 Expert 时只创建一份 weight tiles；GEMM/nonlinear work 按 `n_e` 增长。F-tile 的字节和 operations 分别严格守恒，不使用没有 compute dependency 的任意 byte tile。

新增场景：

```text
exact_full_expert_load
exact_tile_streaming
predictive_full_expert
predictive_uniform_prefix
predictive_ranked_prefix
predictive_DSE
```

同一 `BatchPredictorOutput` 可以在 full-expert、uniform/ranked prefix 和 DSE 场景复用。Exact Router 仍是唯一 authority；active speculative burst 仍不可抢占。

Roofline 使用模拟器精确 accounting：

```text
AI = useful expert ops / unique required expert weight bytes
board roofline = min(effective memory BW × AI, effective board compute peak)
```

同时输出 configured AIE subset ceiling，以区分“board memory-bound”和“所选 AIE 子集与带宽匹配”。带宽 utilization 与 useful-byte fraction 分开报告；DMA busy 不自动视为收益。

### Roofline/DSE CLI

下面的脚本是独立 CLI，所有值均可覆盖：

```bash
python examples/roofline_sweep.py \
  --batch-size 1,4,16,64 \
  --top-k 1,2,4,6 \
  --num-experts 64 --hidden-size 2048 --ffn-size 1408 --weight-bits 8 \
  --memory-channels 3 --bandwidth-gbps 25,50,75,102.4 \
  --bandwidth-efficiency 0.70 --dma-setup-cycles 200 \
  --aie-total-units 64 --aie-units-per-expert 8 \
  --aie-ops-per-cycle 8 --aie-efficiency 0.60 \
  --max-parallel-experts 4 --f-tile-size 352 \
  --candidate-count 4 --prefetch-depth 1 --prefetch-policy uniform \
  --spec-byte-budget 16777216 --ocm-bytes 67108864 --seed 1234 \
  --output results/roofline/
```

`--full-grid` 增加中间 batch 和完整 sweep。输出：

- `results/roofline/roofline_sweep_results.csv`
- `results/roofline/dse_results.csv`
- `results/roofline/summary.json`

Bundled `hardware_profile.yaml` 中 102.4 GB/s 只是三通道 aggregate theoretical study input；它不是已实现或实测 VEK280 带宽。

## Milestone 6.5：表征与验证

Milestone 6.5 不改变 event-driven 调度器；它以可重复的 controlled routing 和同一 predictor trace 驱动 M6，并把结果写为 CSV 后再独立绘图。这里始终使用 **AIE compute group**（内部兼容字段仍可能叫 `aie_unit`）。

```bash
MPLCONFIGDIR=/tmp/edge-moe-mpl python examples/m6_characterization.py --smoke
MPLCONFIGDIR=/tmp/edge-moe-mpl python examples/m6_make_figures.py \
  --input-dir results/characterization
```

完整 synthetic grid：

```bash
python examples/m6_characterization.py --full
python examples/m6_make_figures.py --input-dir results/characterization
```

可用 `--experiment reuse-bandwidth|reuse-aie|prefetch-window|breadth-depth|policy-comparison|topk|memory-channel|tile-size|dse` 只运行一个研究。原始数据位于 `results/characterization/raw/`；九个 PNG/PDF 图、六张 CSV/Markdown 表、`figure_manifest.json`、自动 accounting warning 和 `CHARACTERIZATION.md` 都可由已保存的 CSV 重建。

`RooflineMetrics` 现在明确同时报告 board 与 configured subset：`board_compute_peak_ops_s`、`configured_compute_peak_ops_s`、两个 ridge/ceiling/efficiency，以及 `board_regime` 和 `configured_regime`。因此一个小 compute-group 分配可以是 configured compute-bound，同时 board 仍是 memory-bound，报告不会将两者混为一谈。

所有 characterizer hardware 数字均标记为 **SYNTHETIC / THEORETICAL INPUT**，不是 VEK280 实测结果。当前 M6 eager demand-window issuer 不适合容量小于活跃工作集的极小 OCM 点；此限制会在表征输出中保留为模型限制，而不是把这些点当作硬件结论。
## M7.2 policy phase study

Run the complete unified, paper-oriented policy DSE and regenerate all figures:

```bash
python examples/m72_policy_phase.py --full
python examples/m72_make_figures.py --input-dir results/m72_policy_phase
```

For a quick validation use `python examples/m72_policy_phase.py --smoke`. If a
run completed simulation and only post-processing needs recovery, use
`python examples/m72_policy_phase.py --full --resume`; it reads the deterministic
CSV cache and does not rerun simulation. Results and the 18-section analysis are
in `results/m72_policy_phase/` and `M72_POLICY_PHASE_REPORT.md`. The controlled
hardware profile is a synthetic placeholder. The separate `vek280_theoretical`
subset is explicitly theoretical and is not measured VEK280 performance.

## 可选 Expert 架构

模型 YAML 支持 `expert_architecture: swiglu`（默认）或 `relu_ffn`（两矩阵 ReLU）。
Switch 配置、工作量公式和适用边界见 [中文说明](docs/EXPERT_ARCHITECTURES.md)。
运行主模拟器六层 synthetic 示例：

```bash
python examples/switch_workload.py --model configs/models/switch_base_8.yaml
```

该示例硬件为 synthetic placeholder；不是 FLAME 精确复现或 U200 实测结果。

未剪枝 Switch 的无预取/完整 Expert/tile 预取对照：

```bash
python examples/switch_prefetch_comparison.py
```

使用共享的双候选 84.4% 覆盖率合成激励，输出 CSV、JSON 和 PNG/PDF。
当前默认允许预测就绪后资源驱动预取，不等待上一层全部权重加载完成。
新结果输出到 `results/switch_prefetch_flexible/`；加 `--reference results/switch_prefetch`
可比较历史 barrier 结果，并验证无预取 baseline 不变。
历史假设见 [原始报告](docs/SWITCH_PREFETCH_COMPARISON.md)，
当前结果见 [灵活预取报告](docs/SWITCH_FLEXIBLE_PREFETCH.md)。

当前脚本还输出完整 Expert 按需加载和 tile 串行两个弱 baseline，
默认新输出目录为 `results/switch_prefetch_baselines/`。
使用 `--reference results/switch_prefetch_flexible` 核对既有四种策略周期未改变。
定义及各 speedup 分母见 [弱 baseline 与部分 tile 对照](docs/SWITCH_BASELINE_COMPARISON.md)。

严格的“非流水 + 完整 Expert 预取”对“非流水 + 不预取”实验：

```bash
python examples/switch_nonpipeline_prefetch.py
```

双方都要求当前 Expert 全部权重 ready 才计算，仅预测版本允许跨层预取。
结果保存至 `results/switch_nonpipeline_prefetch/`，见
[非流水预取报告](docs/SWITCH_NONPIPELINE_PREFETCH.md)。

OLMoE 七种 case，以及 Mixtral 两个规模的 streaming-only 对照：

```bash
python examples/olmoe_mixtral_cases.py
```

输出到 `results/olmoe_mixtral_cases/`。Top-K 合成 recall 固定84.4%，
真实模型结构、容量不可行项和 speedup 分母见
[中文报告](docs/OLMOE_MIXTRAL_CASES.md)。

不限定真实模型的PE/带宽/容量收益区域扫描与合并曲线：

```bash
python examples/prefetch_phase_sweep.py --workers 4
python examples/prefetch_phase_sweep.py --config configs/experiments/prefetch_phase_refinement.yaml --output results/prefetch_phase_refinement --workers 4
python examples/analyze_prefetch_phase.py --inputs results/prefetch_phase results/prefetch_phase_refinement --output results/prefetch_phase_analysis
```

全部为合成配置；固定策略曲线与事后最优深度相图分开，结论适用范围见
[预取收益区域报告](docs/PREFETCH_BENEFIT_PHASE.md)。

收益转折附近的22个归一化带宽点、5个seed及局部质量/深度消融：

```bash
python examples/prefetch_dense_sweep.py --workers 4
python examples/prefetch_dense_diagnostics.py
```

输出到 `results/prefetch_dense/`，见
[密集扫描报告](docs/PREFETCH_DENSE_SCAN.md)。

将收益比例映射到真实Switch/OLMoE/Mixtral结构，并核对原始细tile分母：

```bash
python examples/real_ratio_mapping.py --workers 4
python examples/analyze_real_ratio_mapping.py
```

Mixtral使用超出当前片上预算的理论大OCM，不能视为U200部署结果。见
[真实结构比例映射报告](docs/REAL_RATIO_MAPPING.md)。

按现有RTL的slot/pair、tile-ready-first、burst/FIFO及层间退休规则对照，
扫描单Engine内部16–256个假想MAC lanes（不修改RTL）：

```bash
python examples/rtl_rules_pe_sweep.py --workers 4
pytest -q -o addopts=''
```

见[规则对照与MAC规模扫描](docs/RTL_RULES_PE_SWEEP.md)及
[DSP MAC待实施方案](docs/CONFIGURABLE_DSP_MAC_PLAN.md)。
本实验不是RTL联合仿真；更宽MAC的供数能力、DSP映射和板级带宽尚未实现或测量。

同一MAC档位比较非streaming无预取、streaming无预取、非streaming完整Expert预取、
streaming tile预取（四case统一byte OCM及供数时序）：

```bash
python examples/mac_four_cases.py --workers 4
```

需要上一轮保存的输入trace。分母、可行性及与finite-slot规则实验的边界见
[四case同配置收益报告](docs/MAC_FOUR_CASES.md)。

FLAME条件启发的Switch Top-1四case与预取放行屏障对照（主比较取消屏障）：

```bash
python examples/flame_condition_comparison.py --workers 4
```

另有首候选质量敏感性配置。输入、硬件假设、与FLAME/RTL的差异见
[FLAME条件对照报告](docs/FLAME_CONDITION_COMPARISON.md)。

主模型现支持独立配置 Uniform / Ranked / HardwareGreedy-v1 分配器与
Up-first / Pair-first DMA 顺序，保留历史默认行为。有限 RTL 原子 pair 模式
支持 Pair-first，并检查 OCM、descriptor、FIFO credit 与重试顺序。
配置与边界见 [分配器与 DMA 顺序说明](docs/PREFETCH_ALLOCATOR_DMA_ORDER.md)。

```bash
python examples/prefetch_allocator_order.py
```

完整 Switch-base MoE 尺寸的 RTL 四 case 验证（H=768/F=3072/E=8、六层，
真实 DMA/OCM/MAC，DDR 后端为可配置 AXI responder，不是 MIG/RDIMM）：

```bash
python hardware/m10_u200/scripts/run_switch_full_size.py --jobs 4 \
  --output results/m10_switch_full_size_new_run
```

运行目录请使用新的路径。完整尺寸、预算、预测质量和验收边界见
[完整 Switch RTL 对照说明](docs/RTL_SWITCH_FULL_SIZE_COMPARISON.md)。
