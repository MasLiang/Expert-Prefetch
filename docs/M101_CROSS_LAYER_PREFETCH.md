# M10.1：资源感知、无前层权重屏障的跨层预取

## 已接入的数据通路

`u200_moe_top` 的多层模式现在默认开启 `ENABLE_CROSS_LAYER`，支持 1–8 个 MoE 层、
单 token、同一综合配置下的顺序计算与多个未来层预测 context。没有修改主 Simulator
或 HardwareGreedy-v1 算法，也没有进行新的 200 MHz 综合或时序签核。

```text
外部未来层候选事务
  → 按目标层保存候选/地址/预算/epoch
  → 完整候选事务即可参与调度，不等待前层 GU/DOWN 完成
  → 共享 HardwareGreedy-v1 allocator
  → logical GU/DOWN pair admission
  → 同一个有限 OCM、descriptor table、DMA 与四路 DDR 接口
  → 目标层 Router truth：commit / cancel
  → GU 就绪后计算，DOWN payload 可在 GU 计算期间完成
```

提前产生预测、放行预取和开始计算是三个不同事件。候选可提前多个层提交；第 N 层
的 speculative admission 不等待第 N−1 层权重、truth 或计算完成，只要求合法的完整
预测 context、未来目标层和真实资源 credit。未解析的当前层也不阻止未来层预取。
当前层计算仍遵守真实 Router truth、activation、GU readiness 和共享 MAC 依赖。

权重完成历史仍保留至 sequence 结束，但只用于监测，不接入未来层 allocator 的放行条件。
未来层 DMA 若在其 Router 产生 truth 之前完成，useful/wasted 仍延迟分类。

## 新模块与接线

| 模块 | 作用 |
|---|---|
| `cross_layer_prefetch` | 保存逐层候选 context，校验候选与 epoch，完整事务参与共享 allocator，abort 时扫除所有已登记 epoch |
| `layer_weight_barrier` | 记录每层每个 Expert/tile 的真实 GU/DOWN 完成历史，输出 `layer_weights_done` |
| `multilayer_tile_demander` | 保存逐层实际 admitted pair bitmap；目标层 demand 跳过已预取 pair，当前层重复 speculation 也不重新分配 |
| `tile_execution_scheduler` | 保留跨层物理 descriptor/ready 记录，只选择当前 layer/epoch 的合法 prefix tile |
| `moe_layer_chain` | 根据真实数值输出推进，等待控制入口安全而非整个 DMA 排空，再按本层 epoch 清理 |

没有为未来层复制 MAC、DMA 或 OCM。Allocator 的预算是每个预测 context 的 byte
预算及 channel credit；实际 admission 仍检查共享 slot、descriptor 和 FIFO credit。
既有两个 demand reserve slots 保留，speculative 不可占满 OCM 阻塞 demand。
同时为 demand 保留两个 descriptor credit（`DEMAND_RESERVE_DESCRIPTORS=2`）：
speculative pair admission 要求至少四个 free entries，demand pair 只需两个。
未来层 truth 未产生时 descriptor 可能长期驻留，这个资源保留避免其占满 table 后
阻止当前层 demand、从而无法推进到未来层 truth 的循环等待。两项 table 会自然退化为
无预取，不是恢复前层权重屏障；OCM/DMA 约束仍照常生效。
DMA 保持 demand 优先、已启动传输不可抢占，受真实 DDR ready/valid 和 bandwidth
enable 控制。这里不宣称最大化板卡带宽，也没有加入板级 DDR PHY/CDC/shell。

## 外部接口协议

原 `prediction_*` 接口仍用于**当前层**开始计算。未来层候选使用独立接口：

| 接口 | 内容 |
|---|---|
| `future_begin_valid/ready` | 一个未来层预测 context 的 begin 握手 |
| `future_target_layer` | 目标层，必须大于 `current_layer` 且小于 `LAYER_COUNT` |
| `future_epoch / future_sequence` | 该层独立 epoch 与当前 sequence |
| `future_candidate_count` | 0..MAX_CANDIDATES；0 表示空预测，不提交 candidate beat |
| `future_budget_bytes / future_channel_credit_bytes` | 该 context 的预取预算与 credit |
| `future_gu_base_addr / future_down_base_addr / future_expert_stride_bytes` | 当前综合 tile geometry 对应的地址布局 |
| `future_candidate_valid/ready` | 后续 candidate beat 握手 |
| `future_candidate_expert/rank/confidence/last` | 合法 Expert、从 0 连续递增的 rank、Q12 confidence、严格匹配 count 的 last |
| `layer_weights_done` | 每层真实权重完成监测，不控制预取；每次 sequence_start 接受后清零 |

每层每个 sequence 最多一个 future begin，不支持覆盖更新。候选必须无重复，count、
rank、last 必须一致；错误 candidate 不接受，并通过 `transaction_error` 中止及清理。
非法 begin 的 ready 为低，不承诺将非法输入修复成合法事务。

未来层预测可在首层当前事务接受后提交；在 sequence 尚未开始/未建立首层 context
时不接受。**同一 sequence 各层必须使用不同 epoch**：现有 OCM/descriptor flush
以 epoch 匹配，不能让不同层共用 epoch。未来 context 的 sequence 必须匹配当前
sequence；当目标层成为当前层时，`prediction_epoch/sequence` 必须与已保存 context
匹配，且未来 candidate transaction 必须已完整接收。

提交未来候选不需要该层 activation 或 Router 权重已经可用。计算开始时仍需主机
加载当前层真实 Router 权重，并提交当前层量化/Top-K/地址配置。各接口在握手之前
保持稳定；当前层配置与未来层配置使用各自的独立输入端口。

## 层切换与 retirement

`result_valid` 表示当前层真实 combine 完成并产生 INT32 输出。层链捕获结果，经
配置的算术右移及 INT8 饱和产生下一层 activation，然后执行控制切换。

切换不再要求所有未来层 DMA/FIFO/descriptor 清空。它等待：当前 demand 和
allocator 停止、resolver 扫描结束、compute 空闲、descriptor manager 已完整发出
已接受的 GU/DOWN pair。进入 retirement 时冻结新的 future admission，避免
同拍接受 pair 后又清掉尚未发出的 descriptor。

清理只匹配旧层 epoch；其他层的 OCM 内容、请求 token、resolver 记录、execution
metadata 和 admission bitmap 保留。旧层错误 speculative 若仍在传输，以原
epoch/generation quarantine 并正常退休，不会写进已重用的 slot。

`sequence_done` 表示最终输出及层切换控制完成，**不保证所有残余 DMA 已退休**。
下一条 sequence 必须等待 `sequence_start_ready`，该条件仍检查全局资源排空。
`sequence_result_y` 保留最终输出，直到新 sequence 接受或复位。

soft-flush 或 transaction error 中止计算链，并逐个 flush 已登记 layer epoch。
已发起 DMA 继续退休。abort 后不输出成功 sequence_done；等待所有旧引用退休和
sequence_start_ready 后才可开始下一条 sequence，不用全局复位恢复。

## 自检与复现

```bash
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 multilayer-test
make -C hardware/m10_u200 mac-pipeline-test
PYTHONPATH=src pytest -q
```

`multilayer-test` 使用 Vivado 2023.2 XSim，有限运行时间和 watchdog，必须有明确
PASS、正常 finish 且无 unexpected fatal/error。输出到
`results/m10_multilayer_current/`：源文件 SHA256 清单、summary.json、各用例完整
编译/仿真日志，以及 `cross_layer_events.csv`。

CSV 记录绝对 RTL observation cycle：LAYER_START、WEIGHTS_READY、FUTURE_ADMIT、
FUTURE_DMA、LAYER_RESULT；不是主模型预测结果，也不是完整逐 burst timeline。
summary 中另保存实际 transferred/useful/wasted/cancelled/allocated bytes。
两个 byte 守恒分别为：

```text
实际 transferred speculative bytes = useful + wasted
admitted speculative bytes = transferred + 未传输的 cancelled bytes
```

测试覆盖前层 truth/权重未完成时放行、提前两层候选保存、错误 Expert 不污染完成监测、credit
耗尽、稳定 backpressure、epoch 冲突、空预测、重复 candidate、带宽空洞、真实
OCM 4/8 slots、future 与当前 speculation/demand 去重、错误预测取消、跨层 abort
后退休/重启，以及 8 层未来预测连续运行两次。数值参考不读取 DUT 的输出或 LUT。

### 2026-10-06：移除前层权重屏障后的验收

- Python：442 项测试通过。
- Icarus：72 个 RTL 运行点通过；allocation golden 与 6 个 dense numeric golden 通过。
- Vivado 2023.2 XSim：14 个自检用例通过，日志包含 PASS 和正常 finish；源文件
  SHA256 与验证清单一致。覆盖单层、3/8 层、提前两层、命中/错误、abort/restart、
  OCM backpressure，以及 2/4 项 descriptor table 压力。
- Verilator lint 通过，未出现 WIDTH、LATCH、MULTIDRIVEN、UNOPTFLAT 警告；
  仍有命名、空接口、未使用信号等既有警告，不宣称零 warning。
- `future_hit`：layer 1 在 cycle 93/94 接受物理 descriptor，而 layer 0 的全部真实
  权重到 cycle 515 才完成。这直接验证预取不再等待前层完整权重。
- `layer_weights_done` 仍输出真实完成历史，但已从 `cross_layer_prefetch` 接口移除。
- 主模型共同激励脚本和现行模型扫描配置统一关闭旧规则。主模型显式开启旧屏障的
  接口仅供历史 ablation；既有实验输出没有重跑或改写，不能冒充新 RTL 性能证据。
- 最小 table 无预取时仍能完成全部真实计算；这验证 descriptor demand 保留避免
  未来 truth 与当前 demand 之间的循环等待。

结果保存在 `results/m10_barrier_free_validation/`：`summary.json`、
`input_manifest.json`、`rtl_regression.log`、`lint.log` 和逐用例日志/CSV。

```bash
PYTHONPATH=src python hardware/m10_u200/scripts/validate_multilayer.py \
  --output results/m10_barrier_free_validation
make -C hardware/m10_u200 rtl-test golden lint
pytest -q -o addopts=''
```

本次没有运行综合/PnR，没有新的 200 MHz 时序签核，也未重新标定主模型逐 cycle 误差。

### 历史验收结果（移除屏障之前，不作为当前版本证据）

- 251 个 Python 测试通过。
- 70 个 Icarus RTL 运行点通过；allocation golden 和 6 个 dense numeric golden 通过。
- 11 个多层/跨层 XSim 用例通过，最终源文件 SHA256 与清单一致。
- 8 个完整默认配置单层 XSim 用例通过；summary、port metrics、boundary events、
  tile/stage timing 和 timeline 与冻结的 `results/m10_mac_pipeline` 完全一致。
- 单层与多层 Verilator lint 均运行，未发现 width、latch、多驱动或组合环警告；
  仍有既有命名、空端口与未使用信号等警告，不宣称零 warning。

小型合成激励下，命中跨层用例有 2304 bytes speculative 实际传输，其中 useful
1536、wasted 768。abort/restart 用例 admitted 3840、实际传输 3584、未传输取消
256 bytes，且恢复后的三层数值检查通过。8 层连续两次有 16 次真实 Router 和
16 个层输出，未来层 physical admissions 共 28 次。以上是功能验证，不是模型
精度、真实板卡带宽或主 Simulator 性能一致性的结论。

## 尚未签核的内容

- 主 Simulator 与本版本 RTL 的共同激励和逐事件性能对齐尚未运行。
- RTL 仍按 Router 顺序优先选择可执行 tile，并允许 tile 边界切换 Expert；不能
  宣称等价于主模型 `max_parallel_experts=1` 下整 Expert 锁定或 Expert-ID tie break。
- GU 计算仍要求对应 DOWN slot/descriptor 已预留，但不要求 DOWN payload 完成。
- 超过 8 层、每层不同 H/F/E、运行时 PE/DDR 通道数、更换 predictor transaction，
  Attention/residual 和模型精度校准不在本轮范围。
- 本轮证据是功能仿真与 lint，不是新设计的综合、PnR、200 MHz、CDC 或板卡部署验收。
