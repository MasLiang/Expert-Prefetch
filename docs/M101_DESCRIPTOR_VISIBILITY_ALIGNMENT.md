# M10.1：逐 descriptor 可见时序与预取 admission 边界补齐

## 本轮范围与结论

本轮补充的是主模型 `UnifiedSequenceSimulator` 的可配置控制规则。设计 RTL、XDC、
Router truth 和 Scheduler 分配算法均未修改，也没有引入新的 M10 专用 Simulator。

旧配置保留；新配置为 `configs/hardware/u200_descriptor_alignment.yaml`。
这些参数来自当前 RTL 的接口、状态机和流水结构，**不是 U200 板级实测性能**。
仍使用 H=16、F=32、E=6、K=2、INT8、F_TILE=16 的共同功能夹具，而非真实大模型。

验收将功能、请求分类/顺序、总周期、资源峰值分开：不能由一个 PASS 推导另一个。
新模型消除了上一轮 delay=441–446 的六点 speculative 分类差异，双方的首次
预取机会损失点都是 delay=447。五组资源条件的固定计划预取收益也一致。
尚不是所有内部握手、所有资源峰值和全部场景的逐周期等价。

## 1. Payload ready 与 committed ready 分开

过去模型在 truth 解析时立即将所有 hit 提交。现在有限 resolver 表为每个
speculative descriptor 保存独立表项，以最低空闲表项分配，不能依赖 DMA table
提前回收来丢失尚待路由解析的引用。

配置的可见时刻为：

```text
commit_visible_cycle = truth_cycle
                     + route_resolver_drain_cycles
                     + resolver_index * route_scan_cycles_per_entry
                     + route_commit_visibility_cycles
```

当前结构配置是 drain=4、每项扫描=1、token/core/metadata 可见链=3。
第一对 speculative GU/DOWN 在 truth 后第 7/8 拍可见。错误请求也按对应扫描位置
执行 cancel，而非 truth 当拍清空所有 residency。

这是一种接口级事件模型，不是逐寄存器 RTL 模型。表项在模型 logical pair
admission 时预留，而 RTL 在 physical descriptor capture 时登记；表项数、顺序
和生命周期保留，但两个内部登记事件不是同一个物理握手。

promotion FIFO 没有 credit 时，token 保持并重试，后续表项不得越过它。
demand dispatch 等待待解析表项退休。真实路由结果不因 credit 或预测变化。

## 2. 允许选择已到 payload，但不得提前 GEMM

GU payload resident、truth 有效、DOWN ownership 已预留时，可以选择 tile。
dispatch 与 resolver 的 commit 链可以重叠，不能把两段串行相加。

```text
launch_candidate_cycle = max(select_cycle + tile_dispatch_cycles,
                             commit_visible_cycle + committed_launch_cycles)
```

到候选发射事件时再次检查实际 metadata 是否 committed。若 promotion backpressure
延迟了提交，GEMM 不预先占用 AIE；等真实 commit 可见后再发射。所有实际 GEMM 仍由
共享 AIE 资源接口预留，保持互斥。

OCM payload 的可见延迟独立设置为 1 cycle，并生成 `payload_ready` 事件；
`dma_done` 仍表示 DMA 服务完成。该延迟不改变 DMA payload bytes、带宽或 burst
服务时间。等待 CSV 优先使用 `payload_ready`，避免将 metadata/dispatch 等待误记
为 DDR 传输等待。Expert timing 保留 dispatch 前的 request/ready 记录，不在发射时
将其覆盖成 GU_START。

## 3. Truth 刷新与 Expert combine

当前结构配置还包含：

- `truth_scheduler_refresh_cycles=2`：truth transaction 到 scheduler 快照可用。
- `expert_combine_cycles=2`：一个 Expert 完成后的 combine_valid1/2 门控。
- 最后 Expert 的 `layer_combine_cycles=4` 仍单独处理；不重复叠加 Expert combine。

它们是有触发条件的控制依赖，不是每个 layer 末尾增加同一个误差补偿数。

## 4. Allocator 与 admission 截止点

上一版 `prefetch_allocator_cycles=11` 的注释按 16 候选、5 级 score/tree 估计。
但实际 `tb_u200_moe` 显式设置 `MAX_CANDIDATES=2`，其 score/tree 是 2 级。

新共同夹具配置使用：

```text
choice / LAUNCH / FEED     3
selection launch          1
score / tree              2
winner                    1
logical offer             1
合计                      8 cycles
```

这个数适用于当前单候选 transaction、容量为 2 的测试实例，不可直接套到任意
candidate capacity、candidate count 或重复 selection。改变真实 RTL 参数时必须
同步声明相应结构配置，而不是沿用本夹具的常数。

pair admission 保留原子 GU/DOWN 的 OCM、descriptor 与 FIFO 约束，并以 accepted
logical reservation 为边界。预留成功后，GU/DOWN 后续发射不得因层完成而撤销。
同 cycle 事件按既定优先级执行；若 reservation 先于 layer-combine-done 被处理，
它属于已接受请求。层完成事件之后不能在 retirement 期间新增 future reservation。

没有通过允许 retirement 空隙继续偷偷 admission 来获得窗口收益。

## 5. 验证矩阵和结果口径

使用与上一轮相同的 30 组端到端输入：五组 OCM/descriptor/FIFO 资源条件下
no-prefetch/hit；434–450 的逐拍扫描；错误预测、八层和 DDR gaps。
另外重跑 3 组 GU/truth 就绪边界和 2 组 commit 被动观测，共 35 组 XSim 自检。

结果输出到 `results/m10_descriptor_boundary/`，与旧结果分目录保存。

| 资源条件序号 | 主模型节省 cycle | RTL 节省 cycle |
|---|---:|---:|
| 0 | 64 | 64 |
| 1 | 68 | 68 |
| 2 | 76 | 76 |
| 3 | 68 | 68 |
| 4 | 76 | 76 |

默认三层 no-prefetch 为 1398/1398 cycles，hit 为 1322/1322 cycles。
八层 hit 为 3487/3487 cycles。30 组七项执行规则检查全部一致，760 个 GU/DOWN
stage 时长一致；35 组 RTL 功能自检通过。Python 测试为 329 passed。

仍保留的非零总周期残差：

- delay=446：模型 1322，RTL 1323，相差 1 cycle。
- DDR gaps：模型 1338，RTL 1330，相差 8 cycles。当前按有效带宽缩放 bench
  服务节拍，尚未逐 beat 重建 if/else、gap 和 stripe join 的离散时序。

其余 28/30 组总周期一致，不能据此宣称完整微架构等价：默认 demand FIFO 峰值
仍为模型 4、RTL 3；logical pair credit 与 physical FIFO entry 的观测时点不同。
首个 demand 的 dependency/memory 子段仍有 +6/−6 的抵消，SELECT/ACCEPT 的主模型
细分字段仍为空，而非伪造逐拍一致。

`exact_alignment=false` 与 `alignment_passed=false` 因此继续保留。新增摘要明确
列出 `admission_boundary_equal`、`fixed_plan_gain_alignment_passed`、
`all_cycles_equal` 和 `cycle_mismatch_cases`，不把 admission 修复混同于完全对齐。
本矩阵只检查固定 full-expert 计划与 no-prefetch，不证明所有 allocator 的策略排名。

## 6. 兼容性与测试

新增规则默认都为 0，未开启逐 descriptor 时序的通用抽象模式不变。
旧 `u200_interface_alignment` 的边界 30 组配置重新运行主模型后，完整 metrics
与 trace 均与前轮存档一致。旧存档保留其六点差异，不能将新规则倒灌成旧结果。

新单元测试包括 commit/payload 独立、取消表项退休、commit 前不能启动、一次
promotion backpressure 的保序重试、边界两侧、确定性与错误配置检查。
其中 backpressure 注入是模型单元测试，不声称 RTL 矩阵出现了同一个受阻序列。

## 7. 复现

```bash
make -C hardware/m10_u200 descriptor-boundary
pytest -q -o addopts=''
```

只重新分析已有完整运行：

```bash
make -C hardware/m10_u200 descriptor-boundary ALIGNMENT_ARGS='--analyze-only'
```

后处理不是重新仿真。运行保存完整日志、共同输入、分段等待、stage timing、资源
比较及 source hash manifest。关键产物为 `boundary_summary.json`、
`gain_comparison.csv`、`late_boundary.csv` 和各案例 `wait_segments.csv`。

本轮没有新的综合、PnR、板级或大模型 DSE；功能仿真不代表 200 MHz timing closure。
