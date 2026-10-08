# Switch 宽候选 Tile 预取对照

## 实验问题

在相同物理 OCM、候选排序、预测释放时刻、计算能力和 DMA 约束下，完整 Expert 分配与 Uniform/Ranked Tile 分配是否能获得不同的覆盖范围，以及覆盖增加是否转化为净延迟收益。

这不是 CEPR 复现，也不是 U200 板卡实测。旧的双候选实验没有发挥 Tile 分配覆盖更多 Expert 的优势；本实验不覆盖或替换旧结果。

## 配置

- 官方 Switch-base-8/16 的结构尺寸：H=768、F=3072、Top-1，两矩阵 ReLU FFN。
- INT8 权重为部署假设；仅模拟六个 decoder MoE 层，无 attention/encoder、数值推理和跨 token 缓存。
- 单共享计算 Engine，32 operation/cycle；Router、Up、Down 使用同一 Engine。
- 物理权重 OCM 为 16 MiB；每个完整 Expert 为 4.5 MiB。
- 四个内存通道，配置名义带宽为每通道 8/16 byte/cycle；payload cycle multiplier=2，载荷有效速率因此为每通道 4/8 byte/cycle，另有 setup/arbitration/control 成本。
- 每个 F tile 大小为 128，即每 Expert 24 个逻辑 tiles；一个逻辑 tile 含 96 KiB Up 和 96 KiB Down。
- 预取预算上限为两个完整 Expert，即 9 MiB。实际分配预算还受预测释放时空闲 OCM 限制。两种模型均使用相同物理容量和相同 byte 预算。
- 每个模型/带宽/策略 100 个独立序列；共 24 个配置点、2,400 次模拟。每个序列第一层冷启动，后五层使用提前一个 MoE 层产生的预测。

## 共用 Predictor

完整 Expert、Uniform 和 Ranked 都获得同一份完整候选排序：Switch-8 有 8 个候选，Switch-16 有 16 个候选。

保留受控 Top-1=75.8%、Top-2=84.4% 命中次数。剩余 Expert 的顺序使用固定 seed 随机生成；confidence 为 0.758、0.086，以及均分的剩余 0.156。尾部排序和这些 confidence **不是 CEPR 实测**，不能从论文的两个最大命中率推导出来。

完整候选列表包含所有 Expert，因而 candidate-set recall 必然为 1；这不等于首选准确率或实际驻留命中率为 1。真实 Router 结果不受预测影响。

## 六种对照

| 策略 | 预取分配 | 计算启动规则 |
|---|---|---|
| whole_demand | 无预取 | 整个真实 Expert 驻留后启动 |
| stream_demand | 无预取 | 真实 Up tile 就绪即可启动 |
| whole_full_prefetch | 完整 Expert 分配 | 整个真实 Expert 驻留后启动 |
| stream_full_prefetch | 完整 Expert 分配 | 真实 Up tile 就绪即可启动 |
| stream_uniform | 广度优先等深前缀 | 真实 Up tile 就绪即可启动 |
| stream_ranked | confidence × depth_decay^depth / byte，depth_decay=0.5 | 真实 Up tile 就绪即可启动 |

`stream_full_prefetch` 与两种 Tile 策略的比较保持计算启动规则相同，能够避免把整 Expert 等待与 streaming 的收益误归因给 allocator。

Ranked 使用已有 M8/M9 的 `RankedTilePrefetchScheduler`，不是旧主模型的简单 `ranked_prefix = depth - rank`，也不是 Hardware Greedy。

## 接入与 DMA 顺序

主模型新增可选 `tile_prefetch_scheduler` 接口，直接使用已有 allocator。未传入时保持原有路径和默认行为。接入当前明确限制为 byte OCM、非 atomic pair、无限 descriptor/FIFO credit；不声称已支持 finite-slot RTL 配置。

allocator 在预测释放事件取得实际 OCM occupancy，只规划连续前缀。每个逻辑 tile 的 Up/Down 都实际发出 DMA，不把 GU-only 传输当作整个 logical tile。

物理请求沿用 AdvancedSimulator 的 Up-first 接入方式：先按计划顺序提交全部 Up，再按同一顺序提交 Down。Uniform 的 Up 顺序为第一轮所有候选 tile 0、第二轮所有候选 tile 1；Ranked 保留边际价值选择顺序。Full Expert 在此接入中同样是先所选 Expert 的 Up，再 Down，**不是“一个 Expert 的 Up/Down 全部传完再请求下一个”的 FLAME 缓存实现**。因此 whole_full_prefetch 是本实验受控基线，不能与旧驱动或论文速度比直接混用。

真实事件队列继续约束 DMA 带宽、demand 优先、活动 burst 非抢占、OCM eviction、commit/cancel 和共享计算资源。规划不查询目标层未来 Router 完成时间；超出实际窗口的计划可被取消或未能及时完成。

所有策略使用相同固定 allocator 控制成本，仅比较分配/顺序；不建模不同 allocator 的 RTL 面积与真实流水延迟。

## 指标

- `planned_candidate_depths`：计划为各候选分配的逻辑 tile 深度。
- `planned_experts`：至少分配一个逻辑 tile 的候选数。
- `queued_experts`：实际创建请求的 Expert 数，包含后续被取消的请求。
- `started_experts`：至少一个 speculative DMA 实际开始的 Expert 数。
- `first_up_completed_experts_before_router`：tile 0 Up 已完成传输的 Expert 数；不冒充仍然 resident。
- `true_startable_at_router`：根据当时实际 resident 权重和该策略的启动依赖计算。整 Expert 等待策略与 streaming 策略的依赖不同，不能混为同一个 first-tile ready 指标。
- `true_first_up_completed_before_router`：统一比较真实 Expert 的第一个 Up tile 完成情况。
- speedup：先对同一组样本累加 makespan，再取基线总周期/策略总周期，不平均单样本 speedup。

逐预测 CSV 同时提供 byte 预算、深度数组、物理传输量、useful/wasted bytes、释放时刻和 Router 时刻；不将计划覆盖当作实际加速。

## 复现与输出

```bash
python examples/switch_wide_tile_prefetch.py --workers 4
pytest -o addopts='' -q
```

配置：`configs/experiments/switch_wide_tile_prefetch.yaml`。

输出目录：`results/switch_wide_tile_prefetch/`。包含共用输入 JSON、`per_sample.csv`、`per_prediction.csv`、`comparison.csv`、配置/来源校验摘要 `summary.json`，以及 speedup/coverage 两组 PNG/PDF。

新增测试覆盖同预算从 2 扩大到 8 个 Expert、真实两矩阵请求、广度优先顺序、managed/unmanaged DMA、确定性、byte/OCM 守恒、finite-slot 明确拒绝和 Top-1/2 激励保持。

## 结果

完整数值以该目录 `comparison.csv` 为准。以下速度比均相对于同配置的 **stream_demand（无预取 streaming）**，大于 1 表示加速。

| 模型 | 名义 byte/cycle/channel | Full 分配＋streaming | Uniform＋streaming | Ranked＋streaming |
|---|---:|---:|---:|---:|
| Switch-8 | 8 | 0.893154× | 0.935771× | 0.926958× |
| Switch-8 | 16 | 1.007296× | 1.008673× | 1.008867× |
| Switch-16 | 8 | 0.894026× | 0.974995× | 0.949670× |
| Switch-16 | 16 | 1.007311× | 1.008181× | 1.008736× |

相对于 stream_full_prefetch，Uniform/Ranked 在本矩阵四个点均更快；但这不等于均优于无预取 streaming。低带宽下扩宽覆盖缓解了 Full 分配的退化，仍未取得相对无预取的净加速；高带宽下两种 Tile 策略净收益约为 0.82%–0.89%。

### 分配覆盖确实扩大

本轮每次预测计划的预算都是 9 MiB，最多 48 个逻辑 tiles。以下列出按候选排名的典型深度；深度数组与真实 Expert ID 无关。

| 模型 | Full | Uniform | Ranked |
|---|---|---|---|
| Switch-8 | `[24,24,0,0,0,0,0,0]` | `[6,6,6,6,6,6,6,6]` | `[10,7,6,5,5,5,5,5]` |
| Switch-16 | 前2名各24，其余0 | 全部16名各3 | `[9,5,3,3,3,3,3,3,2,2,2,2,2,2,2,2]` |

所以 Full 覆盖 2 个候选，Uniform/Ranked 分别覆盖全部 8/16 个候选。这里测量的是同预算的 allocation，不是只给同两个候选减少深度。

### 计划覆盖与真正就绪不是一回事

下表为被预测的后五层中，真实 Expert 在 Router 时可启动的比例。此表仅比较相同 streaming 启动规则，冷启动第一层不计入该比例。

| 模型 | 名义 byte/cycle/channel | Full 分配＋streaming | Uniform | Ranked |
|---|---:|---:|---:|---:|
| Switch-8 | 8 | 73.6% | 87.2% | 83.4% |
| Switch-8 | 16 | 83.4% | 96.8% | 98.2% |
| Switch-16 | 8 | 73.0% | 19.2% | 75.2% |
| Switch-16 | 16 | 83.2% | 87.8% | 95.8% |

Switch-16 低带宽下 Uniform 的计划覆盖虽为 16，真实 Router-time 启动就绪比例却很低，直接说明“分配到 tile”不能替代真实 DMA 完成/驻留判断。Ranked 的就绪比例更高，但总延迟仍高于 Uniform，说明启动就绪一个指标也不能单独决定全序列 makespan。

在 Switch-8 低带宽下，stream_demand 的总 memory wait 为每序列 56,694 cycles；Uniform 增至 182,498.17，Ranked 增至 201,120.95。Uniform/Ranked 实际完成但最终错误的 speculative payload 分别约为每序列 2.11/2.13 MiB。更广覆盖提高命中机会，但错误流量与共享 DMA 争用仍然存在；“填满 byte 预算”不是最优 prefetch window 分配的保证。上述数据支持这个解释，但未将每个净延迟差严格拆成独立因果贡献。

高带宽下原有无预取 streaming 已接近计算下界，因此更高就绪比例主要消除剩余的少量 startup wait，无法再获得很大的全序列加速。本轮不声称已经找到最优预算/深度/ordering，亦不将结果推广到其它带宽、PE 数量、预测提前量或真实 predictor 的排名分布。

### 验证

- 24 个配置点，`per_sample.csv` 2,400 行，`per_prediction.csv` 8,000 行，`comparison.csv` 24 行。
- 两种模型、两个带宽下的 whole_demand/stream_demand 周期均与之前双候选实验完全一致。
- 完整回归：**472 passed，14 个已有依赖弃用警告**；新增主模型 allocator 测试共 8 项。
- 67 个配置/源码 SHA-256 校验通过，OCM 容量与 speculative useful/wasted byte 守恒检查通过。
- 两组图 `speedup`、`coverage` 的 PNG/PDF 均生成。绘图环境提示默认字体缓存目录不可写，回退到临时目录；产物已成功输出。

结果目录保持独立。旧双候选结果不会因为此次接入自动变成宽候选结论，也没有修改 RTL 或重跑 RTL 仿真。
