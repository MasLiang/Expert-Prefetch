# 主模型与 RTL：四项执行规则统一

后续有限 descriptor/FIFO credits、burst 边界仲裁和控制时间已加入主模型，见
[控制资源对齐报告](M101_CONTROL_RESOURCE_ALIGNMENT.md)。本文记录此前四项规则验证，
下文“尚未模拟有限 credits”的描述属于当时的范围，不是最新实现状态。

## 本次范围

这次统一的是会改变有效容量、请求次序和计算选择的架构规则，而不是用 RTL
测得的总周期反向拟合主模型。实现位于 `UnifiedSequenceSimulator` 中；
没有新增独立性能模拟器，没有修改现有 RTL 调度算法，也没有退回 pair-ready 计算。
RTL testbench 仅新增 OCM peak、admission、DMA burst 的被动观测。

旧 `M10AlignmentSimulator` 继续对当前 RTL 失效，默认拒绝使用。

## 通用配置

`AdvancedExecutionConfig.execution_rules` 使用 `ExecutionRules` dataclass。
默认仍为 byte-capacity、Expert 锁定、历史请求顺序；现有研究配置不自动切换。
非默认规则目前仅由主模型 `UnifiedSequenceSimulator` 支持；
`AdvancedSimulator` 遇到它们会明确报错，不静默忽略。

当前 RTL 共同夹具配置如下：

```yaml
execution_rules:
  ocm_slot_bytes: 512
  ocm_banks: 2
  reserved_demand_slots: 2
  epoch_base: 1
  atomic_tile_pair: true
  compute_policy: tile_ready_first
  retry_future_prefetch: true
  request_order: pair_fifo
```

这些规则可以用于其他固定 buffer 数据流硬件，不表示 U200 有 AIE。
主模型中一个逻辑 AIE unit 在本测试里代表一个共享 MAC Engine。
tile-ready-first 的初版仅支持一个逻辑计算 Engine、一个启动 GU tile；
不支持的组合会报错，不以增大 Expert 并发数冒充 tile 调度。

## 1. OCM：payload 与 allocation 分离

GU payload=512 byte，DOWN payload=256 byte；两者分别占一个 512-byte slot。
DMA 时长、带宽和 speculative bytes 仍只统计 payload。
OCM current/peak 统计 allocated bytes，因此一个 pair 占 1024 byte，而不是 768 byte。

固定模式的 bank 映射为：

```text
epoch = epoch_base + layer_id
bank = (epoch XOR layer_id XOR expert_id XOR tile_id XOR matrix_kind) AND (banks-1)
matrix_kind: GU=0, DOWN=1
```

这与本轮 RTL 夹具的每层 epoch 一致；如果未来使用任意 epoch 分配方式，
必须扩展映射配置，不能假设仍然相同。
总容量必须可均分到 bank；即使全局 byte 足够，也要检查目标 bank 的空闲 slot。
大于单 slot 的 payload 直接报错，不偷偷拆成多个 slot。

## 2. 成对 admission 和 demand 保留

GU/DOWN 同时预检并原子保留，失败时不创建半个 pair，不驱逐现有投机对象。
speculative admission 后必须剩下至少 `reserved_demand_slots` 个全局 slots；
demand 可以使用这些保留 slots。保留额度不等于新增物理容量。

GU 完成时释放 GU slot，DOWN 完成时释放 DOWN slot。
DOWN ownership 存在即可允许 GU 计算，不要求 DOWN payload 已经到达；
DOWN 真正计算仍必须等待自己的 DMA 完成。

当前主模型仍未逐周期模拟 RTL DescriptorTable 和三类 FIFO 的有限 credits，
因此不能把这次 OCM/admission 规则验证等同于全部控制资源等价。

## 3. tile-ready-first 与整 tile ownership

单 Engine 互斥与 Expert 上下文数量分开。
每次 tile 完成后，在 Router 顺序中选择第一个当前 prefix GU resident、
且 DOWN ownership 已存在的真实 Expert。不是按 Expert ID 重排路由。

选中的 tile 保持 ownership，直到其 GU→非线性→DOWN 完成。
之后可换另一个 Expert，不锁定整个 Expert；也不允许在 tile 中途随意切换。
定向测试构造的 GU Expert 顺序为 `2,3,3,2`，证明支持合法交错。
一般 batch 输入按路由中首次出现的 Expert 顺序；当前 RTL 对齐仍限定单 token。

## 4. future 重试和 pair FIFO

有限 OCM 下的 future 计划保存在待发队列中，资源释放后按目标层、候选 rank、
tile prefix 重试。已成功 admission 的 pair 保存去重记录，消费后也不会被重复加载。
前一层所有真实权重完成 barrier 仍成立，不能靠未来 truth 预知来放行。
target 开始时停止该层新 speculative admission，未 admission 的剩余计划单独退休。

计划、实际 admission 和 DMA payload 分开统计：

- `planned_speculative_bytes`：预算生成的计划 payload。
- `initial_allocated_speculative_bytes`：真正成功 admission 的 payload。
- `canceled_unadmitted_bytes`：target 开始时取消、但从未创建 DMA 的计划。
- DMA canceled bytes：已 admission 请求的取消；不能混入未发计划。

当前请求顺序为 `GU0,DOWN0,GU1,DOWN1,...`。
demand 游标按 Router 顺序、Expert 内 tile prefix 推进，遇背压不绕过。
DMA 类优先级为 demand > promoted > speculative，同类为 admission FIFO；
候选 confidence 影响计划，不重新排序已 admission 的 FIFO。

主模型 DMA 当前仍把一个 stage 当成不可抢占的请求；RTL 可以在完成的
physical burst 边界重新仲裁。这里没有掩盖此差异，后续需要单独建模或明确限制。

## 复现与验收

```bash
make -C hardware/m10_u200 alignment-main
pytest -q -o addopts=''
make -C hardware/m10_u200 rtl-test lint golden
```

结果写入 `results/m10_unified_rules_alignment/`，不覆盖上轮
`results/m10_unified_alignment/` 的历史诊断。

XSim 用当前 sources 重建，要求实际数值/路由正确、所有 tile 恰好执行一次、
正常结束且无 unexpected fatal/error。新增观测验收要求两侧：

- 完整 DMA descriptor 首次执行 key 顺序一致。
- 全部成功 admission 的 GU/DOWN descriptor 顺序一致，不能隐藏重复 admission。
- 全部 GU/DOWN 计算 key 顺序一致。
- allocated/transferred/useful/wasted/canceled speculative payload 总量一致。
- OCM peak allocated bytes 一致。

任一观测不一致，诊断脚本直接失败，不能靠配置声明标成通过。
`execution_rule_alignment_passed` 是上述范围内的验收。
`alignment_passed` / `resource_equivalence` 仍保留 false：全部控制资源、
prediction transaction 到达周期、DMA burst 仲裁和总性能周期尚未完全等价。
`--require-alignment` 仍返回失败，不把部分完成当成整体完成。

## 本次实际验证结果

Vivado MCP 确认版本为 2023.2，非工程模式显式 sources、行为级 XSim、
每个 case 有 1 ms watchdog/runtime 上限。五组仿真全部正常完成，
数值/路由/tile 验收及以上五项观测比较均通过。

| 场景 | 主模型 cycle | RTL cycle | OCM peak（两侧相同） | speculative transferred（两侧相同） |
|---|---:|---:|---:|---:|
| 三层无预取 | 1026 | 1410 | 2048 byte | 0 byte |
| 三层命中 | 1010 | 1326 | 2048 byte | 1536 byte |
| 三层全错 | 1026 | 1410 | 2048 byte | 1536 byte |
| 八层命中 | 2680 | 3491 | 2048 byte | 5376 byte |
| 三层命中、较大 OCM | 1010 | 1326 | 4096 byte | 3072 byte |

全错的 1536 byte 均为 wasted，其余命中场景的 speculative transferred 均为 useful。
本矩阵已 admission 的取消 payload 为 0；未 admission 的计划取消另行记录，
不得据此声称取消功能没有用或已覆盖所有并发取消情况。
主模型与 RTL 的全部 160 个 GU/DOWN 服务时长仍一致。
总周期仍存在 316–811 cycle 差，四项规则统一没有消除这些差异。
因此本次验证支持的是明确的执行规则一致性，不是性能误差已收敛。

Python 回归：288 项通过（其中新增 34 项测试实例）。
RTL/golden 回归：`rtl-test lint golden` 返回成功，共 77 个 PASS。
lint 仍有既有 warning；本轮没有修改 RTL 功能或 XDC，没有重新综合/PnR，
不宣称时序、CDC、形式等价或板级性能验证完成。

主要修改文件：`execution_rules.py`、`advanced_schema.py`、`unified_schema.py`、
`unified_sequence_simulator.py`、`hardware/capacity_ocm.py`、
`hardware/multi_channel_memory.py`、`advanced_simulator.py`（拒绝不支持的规则）、
共同 YAML、`compare_unified_alignment.py`、`tb_u200_moe.sv`（仅观测）、
两组 Python 测试与相关 README/中文说明。
