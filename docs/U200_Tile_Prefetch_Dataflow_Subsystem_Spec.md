# M10 U200 Tile-Level Predictive Expert Prefetch 微架构规格

**版本：** v0.9（冻结前行为规格）  
**状态：** 未达到交付 RTL→PnR agentic flow 的条件  
**参考平台：** AMD/Xilinx Alveo U200  
**权威性：** 本文档高于 `hardware/m10_u200/` 下的探索性原型 RTL；后者不能作为
任何行为、接口或时序决策的依据。

## 1. 状态、用途与冻结门槛

本文档是交给 FPGA agentic flow 的唯一微架构规格。实现者的职责是依此完成 RTL、
验证、综合和 PnR，而不是重新做研究决策。

本文当前为 v0.9。只有第 23 节的全部 freeze gate 获得证据后，才能改为
`v1.0 frozen`。在此前，flow 可以完成原型和验证，但不得报告最终 RTL sign-off、
U200 PnR 成功，或与 simulator bit-exact 对齐。

## 2. 设计范围与非目标

实现 tile-level predictive Expert-weight prefetch 子系统，默认服务连续六个 MoE
layer（`0..5`）。它是参数化 PL 控制平面与 U200 适配层，不是完整 Transformer
accelerator。

M10.1 固定范围：

- 同一时间仅执行一个 sequence；single-token、batch size 1。
- `NUM_LAYERS=6`；Top-K 可配置，默认 `TOP_K=2`。
- predictor 由 trace 提供 candidate；不实现 predictor hardware。
- 使用 cycle-counted fake DMA 和 fake compute backend 验证控制语义。
- 预留连接 AXI DMA、banked payload RAM、HLS/RTL GEMM 的稳定接口。
- 主部署策略为 `HARDWARE_GREEDY`；比较策略为 `NO_PREFETCH`、
  `FULL_EXPERT`、`BUDGET_PROPORTIONAL`。

明确不实现：Attention、完整 LLM execution、multi-token packing、跨无关 token
sequence 的 persistent cache、truth oracle、CriticalityAware、learned predictor、
dynamic striping、复杂 eviction optimizer。`FULL_EXPERT` 仅称为
**FLAME-like whole-Expert predictive-prefetch baseline**，不得声称复现 FLAME。

## 3. 术语与基本规则

本文中 **必须（MUST）**、**不得（MUST NOT）**、**应当（SHOULD）** 与
**可以（MAY）** 为规范性用语。

- **logical F-tile**：沿 Expert intermediate dimension `F` 切分的 prefix-ordered
  logical object。
- **physical tile**：一个可独立寻址、传输和驻留的 `GU` 或 `DOWN` weight object。
- **descriptor**：描述物理传输的不可变对象。
- **epoch**：sequence transaction version；epoch 不匹配的 event 是 stale event。
- **commit**：Router truth 证明 speculative tile 有用。
- **cancel**：Router truth 否定尚未启动的 speculative tile。
- **drop-on-complete**：Router truth 否定已启动 DMA，但 DMA 必须完成。
- **evictable**：false resident tile 可以被 allocator 回收。

所有 cycle、byte、ID、counter 与 latency 均为 unsigned integer。硬件不得推断
floating-point datapath。

## 4. 六层系统执行时间线

```text
MoE_0 -> transition_0 -> MoE_1 -> transition_1 -> ... -> transition_4 -> MoE_5
```

六层共享 candidate table、descriptor manager、DMA channel array、OCM tile
residency store、scoreboard 和 compute backend。不得例化六份彼此独立的
accelerator。

当 layer `L` 在 `t_done[L]` 完成，对于 `L < NUM_LAYERS-1`：

```text
router_input_ready[L+1] = t_done[L] + compute_block_cycles[L]
bandwidth_ready[L+1]    = t_done[L] + bandwidth_block_cycles[L]
```

两个 timer 并行运行。

- `router_input_ready` 表示 non-MoE compute block 完成，下一层 Router input 与
  activation input 可用；它不允许跳过 Router 直接启动 Expert compute。
- `bandwidth_ready` 仅允许 target layer 的 Expert-weight DMA issue。
- candidate ingress 可以早于两个 gate；但 descriptor 不得在 `bandwidth_ready`
  之前消耗 target-layer memory bandwidth。

必须验证 `compute_block < bandwidth_block`、`>` 和 `=` 三种关系。

## 5. prediction 与 Router 的时间接口

每个 layer 的 runtime trace 或寄存器配置必须提供绝对 sequence cycle：

```text
prediction_release_cycle[layer]
route_release_cycle[layer]
```

两者分别是 candidate transaction、Router truth transaction 最早能被 core 接受
的时间。对 layer `L`：

\[
t_{prediction}=prediction\_release\_cycle[L]
\]

\[
t_{router\_resolved}=\max(route\_release\_cycle[L],t_{router\_input\_ready})
\]

\[
t_{prefetch\_start}=\max(t_{prediction},t_{bandwidth\_ready})
\]

Expert compute 的启动必须满足：

\[
t_{expert\_start}\ge\max(t_{router\_resolved},t_{router\_input\_ready},
t_{required\_tile\_ready})
\]

Router 可以在 candidate 到达前 resolve；此时所有 true required tiles 由 demand
fallback 处理。所有 trace event 必须携带 epoch 与 layer。提前 event 可等待，延迟
event 可延后，但绝不得影响其他 epoch。

`route_release_cycle` 是 trace/mock Router 对外产生 truth 的最早 cycle；它
**不是**真实 Router GEMM 的 latency model，也不等价于已实现或已校准的 Router
hardware。后续接入 Router backend 时，backend 必须显式产生该 release event，
不得通过解释这个字段隐式引入 Router GEMM latency。

## 6. 精确 MoE、F-tile 与输出归约语义

单层因果图固定为：

```text
Router -> Top-K resolve -> Expert Gate-Up -> nonlinear -> Expert Down -> output
```

logical F-tile 是 **weight object**，不是计算表达式。对 F 区间
`[F_t, F_{t+1})`：

```text
GU weight tile:
  W_gate[:, F_t:F_{t+1}] 与 W_up[:, F_t:F_{t+1}]
DOWN weight tile:
  W_down[F_t:F_{t+1}, :]
```

对应计算为：

\[
G_t=XW_{gate,t},\quad U_t=XW_{up,t}
\]

\[
A_t=SiLU(G_t)\odot U_t,\quad Y_t=A_tW_{down,t}
\]

\[
Y=\sum_tY_t
\]

因此每个 FTileGroup 必须扩展为两个独立 physical descriptor：

```text
FTileGroup(epoch, layer, expert, tile_id)
  |- GU physical tile
  `- DOWN physical tile
```

GU 与 DOWN 具有独立 external address、byte count、OCM slot、ready bit、DMA
state 与 completion event。不得将 FTileGroup 当作一个 monolithic DMA object。
末尾 F-tile 可以更小，所有物理 byte count 必须精确。

每个 `(epoch, layer, expert)` 必须维护 partial-output accumulator 与 tile-complete
bitmap。M10.1 可以在最后一个 required tile 完成后，用固定 `combine_cycles`
模拟归约；它必须计入 layer latency。真实 Top-K gating-weight combination 不属于
M10.1。

`ENFORCE_TILE_ORDER` 是参数，默认 `1`。为 1 时同一 Expert 的 compute 以递增
`tile_id` dispatch，便于确定性验证；为 0 时任何 ready tile 可先执行，但
scoreboard/accumulator 必须保证每个 `Y_t` 恰好累加一次。递增顺序只是 M10.1
实现简化，不是 HardwareGreedy 的算法要求。

## 7. Predictor 与 Router 内容接口

candidate input 为 ordered valid/ready transaction：

```text
prediction_begin_valid/ready
  epoch_id[15:0], sequence_id[15:0], target_layer_id[2:0]
  candidate_count[4:0], policy_mode[1:0], explicit_budget_bytes[31:0]

candidate_valid/ready
  epoch_id[15:0], target_layer_id[2:0], candidate_expert_id[9:0]
  candidate_rank[4:0], candidate_confidence_q12[11:0], candidate_last
```

同一 transaction 内 Expert ID 必须唯一，rank 必须唯一且连续 `0..count-1`。输入
端在 `valid && ready` 前必须保持 stable。`confidence_q12` 必须夹紧为
`[1,4095]`。

Router truth 也为 transaction：

```text
route_begin_valid/ready
  epoch_id, sequence_id, layer_id[2:0], top_k[4:0]

route_expert_valid/ready
  epoch_id, layer_id, true_expert_id[9:0], route_last
```

必须接收恰好 `top_k` 个 unique true Expert。stale epoch/layer 的 route transaction
必须被标记为 stale，且不得改变 candidate、descriptor、OCM、scheduler 或 compute
状态。

## 8. HardwareGreedy-v1：唯一的规范算法

`HardwareGreedy-v1` 是新的硬件黄金定义。必须新增独立 Python
`hardware_greedy_v1` 位精确模型供 RTL 对照。历史 M7–M9.3 的
`HardwareAwareGreedyScheduler` 与已有实验结果不得被静默改写。

每个 candidate entry 保存：

```text
valid, expert_id, candidate_rank, confidence_q12, next_depth,
tiles_per_expert, mapped_channel, next_logical_bytes, exhausted
```

`mapped_channel = expert_id mod NUM_MEMORY_CHANNELS`。候选在 invalid、exhausted、
达到末尾 depth 或不满足全局/OCM/channel credit 时不可选。

令 `C=confidence_q12`、`S=4095`、`d=next_depth`、
`p=priority_shift_per_depth`，默认 `p=1`：

\[
P(d)=\max(1,S >> (d\times p))
\]

\[
N(e,d)=C_e\times P(d)
\]

\[
D(e,d)=B(e,d)\times(1+L_{ch(e)})
\]

`B(e,d)=GU_bytes+DOWN_bytes`；`L_ch` 是当前 prediction transaction 开始以来、
已被接受的 logical allocation 数量，不是 bytes，也不是 DMA completion 数量。

比较 a、b 必须以完整位宽 cross multiplication 完成：

```text
N_a * D_b > N_b * D_a
```

相等时选择字典序最小的 `(candidate_rank, next_depth, expert_id)`。每次
descriptor-manager handshake 成功后仅更新 winner：

```text
remaining_credit -= B(winner, next_depth)
channel_load[mapped_channel] += 1
next_depth[winner] += 1
```

它是 k-way prefix merge，不得 global sort `M*T` tiles。无 eligible candidate 时
停止，结果必须 prefix-closed。single-tile candidate 也使用同一整数定义；历史
floating RankedTile 特例不属于 v1。

默认 `MAX_CANDIDATES=16`，必须参数化支持至少 32。16 输入 comparator tree 有
4 个 compare level。实现报告必须说明 pipeline stage、first-allocation latency、
initiation interval 和 backpressure 行为。

### 8.1 HardwareGreedy-v1 算术位宽

实现不得凭经验截断 cross multiplication。默认最小位宽为：

```text
CONF_W    = 12
PRIO_W    = 12
N_W       = CONF_W + PRIO_W = 24
BYTE_W    = 32
CHLOAD_W  = ceil(log2(MAX_ALLOCATIONS + 1))
D_W       = BYTE_W + CHLOAD_W + 1
CMP_W     = N_W + D_W
```

`N=N(e,d)` 使用至少 `N_W` bit；`D=D(e,d)` 使用至少 `D_W` bit；两个 cross
product 均使用至少 `CMP_W` bit 的 unsigned full-precision result。参数化实现
可以扩大但不得缩小。综合/PnR report 必须列出最终 `MAX_ALLOCATIONS`、
`CHLOAD_W`、`N_W`、`D_W`、`CMP_W` 与所有明确 truncation point；不允许隐式
overflow 或 signed promotion。

## 9. Policy mode 与 resource credit

- `NO_PREFETCH`：不发出 speculative logical allocation。
- `FULL_EXPERT`：按 rank 递增扫描；只有该 Expert 的完整 logical tile set 同时
  满足 global、OCM 与对应 channel credit，才可接纳。
- `BUDGET_PROPORTIONAL`：以 floor 实现 `budget*q/sum(q)`，再按 rank 递增做
  residual round。因为 `confidence_q12>=1`，`sum(q)` 不会为零。
- `HARDWARE_GREEDY`：第 8 节。

未实现 `BUDGET_PROPORTIONAL` 时，端口必须返回 `POLICY_UNSUPPORTED`，不得静默
替换为其他策略。

`FULL_EXPERT` admission 是 Expert-atomic：在创建任何 physical GU/DOWN
descriptor 前，必须预先计算该 Expert 的**全部** logical tile bytes。若全部 tile
映射到同一 channel，必须以总 logical bytes 一次性检查该 channel credit，不能
逐 tile 边发送边检查而部分接纳 Expert。若未来允许多 channel mapping，则必须
先分别累加每个 channel 的总需求，并原子检查全部 channel credit。

credit 定义如下：

```text
speculative_ocm_credit = max(0, free_ocm_bytes - demand_reserve_bytes)
channel_bw_credit[c]  = max(0, opportunity_bytes[c] - reserved_bytes[c])
policy_credit          = explicit_budget_bytes，未启用时为 all-ones
effective_budget       = min(speculative_ocm_credit,
                             sum(channel_bw_credit), policy_credit)
```

一个映射到 channel `c` 的 allocation 必须同时满足 `effective_budget` 与
`channel_bw_credit[c]`。demand reserve 对 speculative 不可用。credit 是 byte
counter，overflow 时饱和；telemetry counter 独立以 64-bit wraparound 计数。

## 10. 不可变 Descriptor ABI

physical object 的 canonical key 为：

```text
Key = (epoch_id, layer_id, expert_id, tile_id, matrix_kind)
```

同一个 Key 最多存在一个 live metadata entry 与一个 live descriptor request。

| 字段 | 最小位宽 | 含义 |
|---|---:|---|
| `epoch_id`, `sequence_id` | 各 16 | transaction/version identity |
| `layer_id` | 3 | 默认 0..5 |
| `expert_id` | 10 | 最多 1024 Experts |
| `tile_id` | 8 | 最多 256 F tiles |
| `matrix_kind` | 1 | `0=GU`，`1=DOWN` |
| `external_addr` | 64 | byte address |
| `byte_count` | 32 | 精确 physical bytes |
| `ocm_bank`, `ocm_slot` | 4，16 | 动态 OCM 位置 |
| `memory_channel` | 4 | 映射 channel |
| `request_class` | 1 | `0=SPECULATIVE`，`1=DEMAND` |
| `candidate_rank`, `confidence_q12` | 5，12 | 不可变 prediction metadata |

descriptor 中不得放 mutable state、timestamp、credit、queue position、useful 或
false flag。host address table 定义：

```text
base = matrix_base[layer][expert][matrix_kind]
external_addr = base + tile_byte_offset[layer][expert][matrix_kind][tile]
memory_channel = expert_id mod NUM_MEMORY_CHANNELS
```

core 不得假设 U200 AXI width、DDR map 或 burst alignment。

## 11. Descriptor manager、去重与 backpressure

manager 接收一个 logical allocation，按确定顺序输出 GU 后 DOWN 两个 physical
descriptor。两个 OCM slot 与 metadata 必须均 reserve 成功后，任一 descriptor 才能
入 DMA queue。若资源或 queue capacity 不足，manager 不握手，scheduler state 不得
推进。

去重规则：

- speculative Key 不存在：建立 speculative request。
- demand Key 不存在：建立 demand request。
- demand Key 已是 speculative queued/inflight/resident：promote 或 commit 现有项。
- speculative Key 已 live：不得 duplicate。
- stale completion：不得释放或复用其他 epoch 的 slot。

所有 streaming interface 使用 valid/ready。除非 `valid && ready`，任何 ownership
和 state 都不得推进。

## 12. 可综合的 cancel/promote queue 表示

普通 FIFO 不支持删除中间项。禁止 shift FIFO、rebuild FIFO 或无限 CAM arbitrary
delete。冻结如下实现：

```text
DescriptorTable[request_id]:
  valid, generation, key, descriptor, queue_class,
  cancelled, promoted, inflight, drop_on_complete, slot_generation

DemandFIFO:    {request_id, generation}
SpecFIFO:      {request_id, generation}
PromotedFIFO:  {request_id, generation}
```

FIFO 只存 ID/generation，不复制 descriptor。arbiter priority 固定为：

```text
DemandFIFO valid head > PromotedFIFO valid head > SpecFIFO valid head
```

取 head 时查表。若 entry invalid、generation 不同或 `cancelled=1`，丢弃该
token 并继续扫描，称为 tombstone skip。false queued spec 只设置
`cancelled=1`，不得删除 FIFO 中间项。

true queued spec promotion 必须：

1. 将旧 SpecFIFO token 逻辑 tombstone；
2. 在 table 设置 `promoted=1` 和 demand class；
3. 向 PromotedFIFO 写入新的 `{request_id,generation}`；
4. 若 PromotedFIFO 满，route resolver 必须 backpressure，不能丢 promotion。

选中 token 后 entry 标记 inflight；残余 burst 由同一 entry 持有。spec burst 被
demand 打断时，以新 generation/token 回到 SpecFIFO 尾部，保证确定性顺序。

`request_id` 的 table entry 不得在仅看到 `cancelled`、FIFO pop 或 descriptor
completion 后立即回收。它只有在同一 `(request_id,generation)` 的所有潜在引用均
退休后才可重用：DemandFIFO、SpecFIFO、PromotedFIFO token 已被消费或成为
tombstone，DMA ownership 已释放，所有 burst response/completion path 已退休，且
没有 route/compute sideband 仍可引用它。回收时必须递增 generation；任何旧 token
或 completion 因 generation 不匹配只能被 discard。

## 13. DMA：burst 边界仲裁与 fake timing

DMA 仲裁单位是 **burst**，不是完整 descriptor。一个 physical descriptor 先执行
一次不可中断的 `setup_cycles`，再进入可重选 burst 序列：

```text
setup(desc) -> burst_0(desc) -> arbitrate -> burst_1(desc) -> arbitrate -> ...
```

定义 `MAX_BURST_BYTES[channel]` 与 `bytes_per_cycle[channel]`；前者在 M10.1
必须为 2 的幂。若 descriptor 剩余 `R` bytes，则下一 burst 为：

```text
burst_bytes = min(R, MAX_BURST_BYTES[channel])
```

其 payload latency 为：

\[
T_{burst}=\left\lceil\frac{burstBytes}{bytesPerCycle[channel]}\right\rceil
\]

当前 setup 或当前 burst 都不可抢占；每个 burst 完成后重新仲裁。若存在
demand/promoted request，必须在下一 burst 先于 speculative request 运行。被打断的
spec descriptor 保持 remaining bytes，并按第 12 节规则重入 SpecFIFO。descriptor
总完成时间由实际 interleaving 决定，不得使用单一
`setup + ceil(descriptor_bytes/rate)` 假装 descriptor-level non-preemption。

`demand_stall_cycles` 从 required demand 进入可服务 queue 后累计，到其第一个
setup 获得 ownership 为止。它可包括正在进行的 speculative setup/burst，但不得把
bandwidth gate 尚未开启的时间计入 stall。

## 14. OCM residency store 与 metadata state machine

residency store 包括 metadata RAM、payload RAM（M10.1 可 metadata-only）、
per-bank free list 与可选 victim queue。它不是 direct-mapped Expert cache。

`NUM_OCM_BANKS` 在 M10.1 必须是 2 的幂，`BANK_BITS=log2(NUM_OCM_BANKS)`：

```text
h = epoch_id ^ zero_extend(layer_id) ^ zero_extend(expert_id) ^
    zero_extend(tile_id) ^ matrix_kind
bank = h[BANK_BITS-1:0]
slot = free_list[bank].pop()
```

不得综合一般 modulo divider。bank 内 slot 必须动态分配，live Key 会 pin slot 直到
安全释放。

| 当前状态 | trigger | 下一状态 |
|---|---|---|
| `FREE` | allocate speculative | `QUEUED_SPEC` |
| `FREE` | allocate demand | `QUEUED_DEMAND` |
| `QUEUED_SPEC` | spec DMA start | `INFLIGHT_SPEC` |
| `QUEUED_DEMAND` | demand DMA start | `INFLIGHT_DEMAND` |
| `INFLIGHT_SPEC` | matching DMA done | `RESIDENT_SPEC` |
| `INFLIGHT_DEMAND` | matching DMA done | `COMMITTED` |
| `RESIDENT_SPEC` | true route | `COMMITTED` |
| `QUEUED_SPEC` | true route | `QUEUED_DEMAND` |
| `INFLIGHT_SPEC` | true route | committed-inflight flag |
| `QUEUED_SPEC` | false route | `FREE` |
| `INFLIGHT_SPEC` | false route | `DROP_ON_COMPLETE` |
| `RESIDENT_SPEC` | false route | `EVICTABLE` |
| `DROP_ON_COMPLETE` | matching DMA done | `EVICTABLE` 或 `FREE` |
| `COMMITTED` | compute start | `RUNNING` |
| `RUNNING` | consumer release | `FREE` |
| `EVICTABLE` | allocator reclaim | `FREE` |

committed-inflight 可由 inflight state 加 flag 表示，但 matching completion 后必须成为
`COMMITTED`。每个 command 与 completion 必须检查 epoch；不匹配时只增加
`stale_event_count`，不得改状态或 free slot。

## 15. soft flush、inflight DMA 与 slot ownership

`soft_flush(epoch)` 绝不可释放 DMA setup、burst 或 response path 中的 physical
OCM slot：

| flush 时状态 | 必须动作 |
|---|---|
| queued | 设置 `cancelled`；descriptor token 被安全回收后才能释放 slot |
| inflight/setup/burst | 设置 `drop_on_complete=1`；slot 进入 quarantine，不得复用 |
| resident/running | 依赖 consumer；running tile 完成后才可释放 |

每个 physical slot 必须有 `(epoch_id, slot_generation)` ownership tag。每个 DMA
burst write response 也必须携带相同 tag。只有 live metadata tag 完全匹配时，OCM
write guard 才能写 payload；旧 tag write 必须 discard，并计入
`stale_event_count` 和 `discarded_dma_write_count`。

主规则：inflight descriptor 的 slot ownership 直到最后一个 burst response retire
后才可释放。write guard 只是第二道防线，不能替代 quarantine。这样新 epoch 不会被
old DMA payload 覆盖。

## 16. Router commit/cancel 与 demand fallback

对每个 physical tile，non-stale route resolve 后执行：

| truth | 现有状态 | 必须动作 |
|---|---|---|
| true | `QUEUED_SPEC` | promote 为 demand priority，保留 slot/descriptor |
| true | `INFLIGHT_SPEC` | 设置 committed flag，保留 active DMA |
| true | `RESIDENT_SPEC` | 标记 committed/useful |
| true | absent | required 时创建 demand descriptor |
| false | `QUEUED_SPEC` | cancel，安全后 free metadata/slot |
| false | `INFLIGHT_SPEC` | 不抢占，标记 drop-on-complete |
| false | `RESIDENT_SPEC` | 标记 evictable，永不 compute-ready |

demand fallback 的粒度是 physical GU/DOWN tile。required Key absent 或 unusable 时
触发。demand 有严格优先级，但仍可能等待当前 active speculative burst 结束。

## 17. Replaceable compute backend

core/backend ABI 固定为：

```text
compute_req_valid/ready
  epoch_id, layer_id, expert_id, tile_id, matrix_kind, ocm_bank, ocm_slot,
  activation_addr

compute_done_valid/ready
  epoch_id, layer_id, expert_id, tile_id, matrix_kind
```

M10.1 fake compute 使用 configurable GU/nonlinear/DOWN latency，并 obey 第 6 节
dependency/backpressure。未来 HLS GEMM backend 必须保留此 ABI，不得改变
scheduler、descriptor、DMA 或 OCM semantics。

## 18. reset、flush 与 epoch rollover

power reset 清除 candidate state、queue、metadata、free list、compute state 和
telemetry。`sequence_start` 必须分配 fresh epoch。host 不得在相关 event 仍 live 时
复用 epoch。

`soft_flush(epoch)` 只影响 matching epoch，具体 slot 行为见第 15 节。epoch 按
`2^16` wrap；host 在 wrap 后复用旧值前必须 drain 或 flush 所有旧工作。stale route、
DMA 与 compute completion 都必须可由 telemetry 观察，且仅能被忽略。

## 19. Telemetry 架构与精确定义

所有 telemetry counter 都是 64-bit unsigned wraparound，只有 power reset 或明确
telemetry reset 清零。

| counter | increment 时机 |
|---|---|
| `total_cycles`, `per_layer_cycles[L]` | 每个 active sequence/layer cycle |
| `speculative_allocated_bytes` | physical metadata reservation 成功 |
| `speculative_issued_bytes`, `demand_issued_bytes` | matching DMA setup start |
| `useful_speculative_bytes` | true speculative transfer completion |
| `wasted_speculative_bytes` | false speculative transfer completion |
| `cancelled_before_issue_bytes` | false queued descriptor removal |
| `inflight_wasted_bytes` | false inflight 在 Router resolve 时被标记 |
| `demand_stall_cycles` | required demand 被 active DMA/gate 阻塞 |
| `compute_stall_waiting_gu/down` | compute eligible 但 required tile 未 usable resident |
| `ocm_peak_occupancy` | 最大 reserved payload bytes |
| `channel_dma_busy_cycles[c]` | 每 channel busy cycle |
| `demand/spec_fifo_peak_depth[c]` | 每类 FIFO 最大深度 |
| `scheduler_cycles`, `allocation_count` | scheduler active cycle / accepted logical allocation |
| `commit_count`, `cancel_count`, `stale_event_count` | transition / stale rejection |

`inflight_wasted_bytes` 是 `wasted_speculative_bytes` 的子集；报表不得把两者相加
称作 total waste。useful/wasted bytes 由 Router truth 与 DMA completion 共同确定，
不得在 speculative issue 时提前计数。

若 DMA completion 先于 Router resolve 到达，则该 speculative byte 处于
**unclassified** 状态：不得在 completion 当下计入 useful 或 wasted，必须保留
completion accounting，直至 matching non-stale Router truth 到达后再做 deferred
finalization。该 truth 为 true 时计入 `useful_speculative_bytes`；为 false 时计入
`wasted_speculative_bytes`。这不会改变 `speculative_issued_bytes` 或
`channel_dma_busy_cycles` 的已发生事实。

## 20. 参数、U200 binding 与 provenance

compile-time 参数包括：

```text
NUM_LAYERS=6, MAX_CANDIDATES=16（必须支持至少 32）, TOP_K_MAX,
NUM_MEMORY_CHANNELS, NUM_OCM_BANKS, OCM_SLOTS_PER_BANK,
DMA_FIFO_DEPTH, DESCRIPTOR_TABLE_ENTRIES, MAX_BURST_BYTES,
TELEMETRY_W=64
```

runtime 参数包括 epoch/sequence、policy、candidate count、Top-K、每层
compute/bandwidth block、prediction/router release cycle、explicit budget、
demand reserve、per-channel bandwidth credit、fake DMA rate/setup/burst、fake
compute latency、`combine_cycles` 和 tile/address geometry。

core 为 platform neutral。U200-specific AXI、DDR、XRT、PCIe、clock 和 physical
memory binding 只能出现在 U200 adapter。U200 channel count、usable OCM、AXI width、
clock、device part、bandwidth 与 Fmax 均不得臆造。每一项必须标为 `VERIFIED`、
`ASSUMED` 或 `PLACEHOLDER`。generic FPGA、VEK280 theoretical configuration、U280
XPFM 均不得作为 U200 数值替代来源。

## 21. Python 黄金模型对应关系

| Python reference | RTL responsibility |
|---|---|
| 新增 `hardware_greedy_v1` | candidate table、comparator tree、winner update |
| `FullExpertPrefetchScheduler` | whole-Expert policy mode |
| `NoPrefetchScheduler` | policy bypass |
| `BudgetProportionalScheduler` | optional exact proportional mode |
| `MultiChannelMemory` | DMA channel array 与 priority arbiter |
| `CapacityOnChipMemory` | residency metadata、reservation、reclamation |
| `FTileGroup` | logical-to-GU/DOWN descriptor expansion |
| `UnifiedSequenceSimulator` | six-layer causal sequencing的参考语义 |

flow 必须提供 deterministic trace generator 与 machine-readable comparison report。
对照项至少包括 allocation order、Expert ID、tile ID、matrix kind、channel、
request class、credit evolution、DMA issue/completion order、commit 与 cancel。RTL
pipeline latency 可以不同，但 decision/resource semantics 不得不同。历史 M7–M9.3
的已有 golden/result 不得被该新模型改变。

## 22. 强制验证矩阵

每项必须有命名 deterministic assertion：

1. zero budget；2. budget 小于一 tile；3. exact 一 tile；4. OCM full；
5. demand reserve；6. equal confidence；7. deterministic tie；8. one candidate；
9. max candidate；10. multiple channel；11. saturated channel；
12. queued spec 加 demand；13. demand 到达 active spec；14. queued cancel；
15. false inflight completion；16. false resident eviction；17. hit commit；
18. true miss demand fallback；19. stale DMA completion；20. stale router result；
21. GU ready 早于 DOWN；22. DOWN ready 早于 nonlinear complete；
23. compute block 小于 bandwidth block；24. bandwidth block 小于 compute block；
25. equal block；26. six-layer rollover；27. prefix closure；
28. HardwareGreedy-v1 golden match（multi-tile 与 single-tile）；
29. BudgetProportional match（启用时）；30. NoPrefetch/FullExpert correctness；
31. burst 中 demand 到达后下一 burst 获胜；32. full PromotedFIFO backpressure；
33. flush 后 old DMA write 无法覆盖 new slot；34. `ENFORCE_TILE_ORDER=0` 下
accumulator exactly once。

失败日志必须打印 epoch、layer、Key、metadata state、FIFO 内容、credit 和
slot-generation。random test 可以补充，但不可替代命名 deterministic test。

## 23. v1.0 freeze gate

只有下列证据全部存在，本文档才可升级为 `v1.0 frozen` 并交给自动 FPGA flow 做
最终实现：

1. 新增 `hardware_greedy_v1` Python integer golden model，且 M7–M9.3 输出不变。
2. multi-tile/single-tile 的 Python/RTL comparison zero mismatch。
3. prediction 与 Router early/late/same-cycle release 均通过。
4. burst 中 demand 在下一 burst boundary 获胜。
5. tombstone cancel/promotion 在 FIFO full/backpressure 下无丢失、无重复。
6. flush 后 old epoch DMA completion/write 无法改 metadata 或覆盖 new payload。
7. `ENFORCE_TILE_ORDER=0` accumulator exactly-once 与默认顺序模式均通过。
8. 第 22 节所有项目映射为自动化测试。
9. HardwareGreedy 位宽推导及 cross-product overflow assertion 通过；report 列出
   `CHLOAD_W`、`N_W`、`D_W`、`CMP_W`。
10. FullExpert atomic admission、request table retirement、以及 DMA-before-route
    deferred telemetry finalization 均有命名测试。

## 24. RTL、综合与 PnR Definition of Done

成功必须同时满足：

1. 所有强制 RTL simulation PASS。
2. Python/RTL golden report 为 zero mismatch。
3. 无 inferred latch、无 combinational loop。
4. reset/clock constraint 完整，无 unintended unconstrained path。
5. CDC clean，或每个 exception 有文档化理由。
6. 对**已验证**的 U200 part/platform synthesis 成功。
7. PnR 成功且 timing clean；否则报告 WNS、TNS、critical root cause 并标 `BLOCKED`。
8. 生成 LUT、FF、BRAM、URAM、DSP、WNS、TNS、Fmax estimate、constraint coverage
   与 hierarchical utilization report。
9. 未经 board run，绝不得声称 measured speedup、bandwidth 或 latency。

若没有 verified U200 part/XPFM，generic RTL simulation 仍必须完成，但 synthesis/PnR
状态必须是 **`BLOCKED: verified U200 platform absent`**。不得以 U280 或 VEK280
替代并标注为 U200。

## 25. FPGA agentic flow 必须交付的产物

```text
rtl/       synthesizable core、U200 adapter、constraints
tb/        deterministic 与 randomized self-checking tests
scripts/   simulation、golden、synthesis、PnR commands
configs/   provenance-tagged platform/runtime configuration
traces/    fixed inputs 与 expected outputs
reports/   sim、golden、synthesis、PnR、timing、utilization artifacts
docs/      deviation 与 final sign-off report
```

final sign-off report 必须列出 tool version、target part/XPFM provenance、可复现
command、测试结果、对本文的 deviation、timing/resource 和 blocker。缺少必需产物
属于 `FAILED` 或 `BLOCKED`，不得隐式视为成功。

## 26. 交给 FPGA agentic flow 的执行指令

本文是唯一的行为规格。先实现第 21 节所需的 `hardware_greedy_v1` Python golden
与 deterministic trace，再实现 RTL。任何未实现 policy、U200 XPFM 缺失、parameter
provenance 未定义、timing 未闭合或 golden mismatch，都必须在 machine-readable
report 中标为 `BLOCKED` 或 `FAILED`。不得使用默认假设、替换板卡、近似算法、忽略
stale event 或跳过验证来宣称成功。

## 27. 当前 RTL 原型的已确认配置与实现边界

以下配置是当前 `tile_prefetch_predictive_axi_top` 原型的**已确认 ABI**，不是
U200 的实测峰值、时钟频率或带宽承诺：

| 项目 | 当前值 |
|---|---:|
| 权重/activation 数据类型 | signed int8 |
| MAC accumulator | signed int32 |
| DMA/OCM beat | 128 bit，即 16 个 int8 lane |
| physical F-tile | GU 与 DOWN 各一个 descriptor；logical admission 原子保留二者 |
| candidate confidence | Q12，无符号 12 bit |
| Router truth | 有界 KeyResolver 扫描，true commit，false cancel/drop |
| DMA 数据面 | 单 outstanding AXI4 read burst，128-bit INCR |
| `route_release_cycle` | trace/mock Router 释放 truth 的时刻，**不表示 Router GEMM latency model** |

当前端到端顶层的连接为：`control_transaction_parser ->
candidate_prefetch_scheduler -> tile_prefetch_axi_top -> axi_read_adapter ->
ocm_payload_ram`。每个已 physical-admit 的 speculative descriptor 同时进入
`route_token_resolver`；Router truth 仅通过 `(request_id, generation)` token 回流，
不会复制或修改 FIFO 中的 descriptor。

`FULL_EXPERT` 的未来 admission 必须对同一 channel 的全部 logical bytes 先做总额
credit 检查；禁止逐 tile 部分发射。若 DMA 在 Router truth 前完成，speculative bytes
必须保持 unclassified，直到 truth 到达后才归类为 useful 或 wasted。

2026-10-03 更新：上述真实 fused-SwiGLU tile payload layout、true-expert missing-tile
demand fallback、带 Expert key 的 vector partial-output 与 Q15 combine 已接入。
GU bytes 为 DOWN 的两倍；默认 H=16、F_TILE=16 对应 512/256 byte。数值 ABI、
定向回归、dense Python/RTL golden 和剩余 freeze-gate 差距，以
[M101_IMPLEMENTATION_STATUS.md](M101_IMPLEMENTATION_STATUS.md) 的当前代码事实为准。
板级 PnR、全 policy 与全部并发矩阵仍须单独验收，当前没有升级为 v1.0 frozen。
