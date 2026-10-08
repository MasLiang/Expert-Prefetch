# M10.1：控制等待来源与临界预取窗口验证

## 验证结论

本轮只补充主模型 trace 和 verification-only 观测、脚本、定向 bench，**未修改
设计 RTL、XDC、Scheduler 算法或性能参数**。使用相同基础激励，在真实 XSim 2023.2
运行 30 组端到端矩阵、3 组 GU/truth 就绪边界和 2 组 commit 被动观测。

35 组功能自检全部通过；矩阵的容量上界检查通过，760 个 GU/DOWN stage 时长一致。
但是矩阵只有 24/30 组七项执行规则全部一致：delay=441–446 的六组 speculative
字节分类不同。这是 simulator 的临界窗口误差，**不是 RTL 数值/功能仿真失败**。
结果明确保留 `all_execution_rules_equal=false`、`can_ignore_boundary_error=false`。

## 观测与时间口径

真实 RTL 事件保留 layer、Expert、tile、matrix、slot、generation、epoch 和 cycle：
ADMIT、READY、SELECT、COMPUTE_ACCEPT、RELEASE_GU、RELEASE_DOWN、SLOT_FREE。
SLOT_FREE 读取对应 generation 的实际 OCM metadata，不以全局 occupancy 降低代替。
另记录 descriptor/FIFO pair credits 的变化、scheduler stale/combine/pending 阻塞状态。
这些信号不参与驱动，没有 force/deposit，也没有增加 fake ready。

主模型只新增 `tile_select`、`slot_release` trace。逻辑 OCM 回收仍是原来的即时回收，
不能将新增事件误解为新增 controller timing 模型。

`wait_segments.csv` 按每个 tile 划分三个互不重叠的 pre-GU 区间：

1. dependency_wait：起点到 truth 和 DOWN ownership 等非 GU-payload 依赖满足。
2. memory_wait：以上依赖满足后，到 GU payload ready。
3. eligible_to_gu：上述条件全部满足后，到 GU 开始。

首 tile 起点为 layer start；之后为实际执行顺序中前一 tile 的 DOWN_DONE。
三段严格相加等于起点→GU_START，不能再把内部 SELECT/ACCEPT 子段重复加到总等待。
这里不是全层 idle/memory-stall 的完整账本：首段包含 Router，也未展开 DOWN 等待和
非线性服务。`eligible` 只是 payload+ownership+truth 的抽象资格，不代表 physical
metadata 已 committed。commit 因此可能处于第三段。

SELECT→COMPUTE_ACCEPT、COMPUTE_ACCEPT→GU 单独保存真实 RTL 时长。主模型尚未显式
拆分握手，所以对应 main_cycles 留空，不伪造 0 或人为拆分 dispatch 参数。
release→SLOT_FREE 则标为 logical-vs-registered credit 比较，不加入 pre-GU 总和。

## 误差来源：哪些已证实

### 1. 首个 demand：有抵消，不只是一个固定控制常数

默认资源条件下，首 tile 从 layer 起点算：

| 区间 | 主模型 | RTL | RTL−模型 |
|---|---:|---:|---:|
| dependency_wait | 81 | 87 | +6 |
| memory_wait | 28 | 23 | −5 |
| eligible_to_gu | 4 | 4 | 0 |
| 总计 | 113 | 114 | +1 |

dependency 的差异包含 pair 原子逻辑 admission 与实际 DOWN descriptor 接受时间；
memory_wait 的缩短与不同起点有关，**不能直接解读为 RTL DMA 实际带宽更高**。
看总周期的 1 拍残差，会漏掉 +6/−5 两段的抵消；用单一补偿量会掩盖它们。

### 2. Expert 切换的额外两拍：combine 阻塞已直接确认

同 Expert 的相邻 tile 可在前一个释放后继续选择。最后一个 tile 完成时，
`tile_execution_scheduler` 的 `combine_valid1`、`combine_valid2` 两级流水阻止 SELECT。
默认实例中已观测到 cycle 297、298：GU/DOWN 都可选、stale=0、pending=0，
但 combine1/2 分别为 1；SELECT 在 cycle 299 才发生。

这是 gated combine 的真实控制依赖，不是 MAC 计算时间差。
可以研究把 combine 与下一 Expert 的 GU 重叠，但须证明 accumulator、weight/index
保持正确。本轮没有解除该门控，也没有未经验证地认为这两拍必须存在。

### 3. 命中路径：payload resident 不等于可计算

独立 commit wrapper 复用原 no-prefetch/hit 激励，被动观察 metadata 与 compute phase。
命中组 layer 1 的实际序列：

```text
558：COMPUTE_ACCEPT，metadata=RESIDENT_SPEC，payload_complete=1，gu_ready=0
559–560：WAIT_GU，payload 已到，但尚未 committed
560：resolver 发出 GU commit
561：metadata=COMMITTED，gu_ready=1
562：GU_START
```

所以正常请求 ACCEPT→GU 为 1 cycle，而这些命中请求为 4 cycle。后两个 layer
各观测到 2 cycle 的 resident-but-uncommitted WAIT_GU。主模型在 truth 解析时提交
命中对象，没有保留逐 descriptor commit 的可见时序，因此其 eligible→GU 为 4，
RTL 为 8。这四拍差异同时包括 scheduler truth 生效/刷新及 commit 交接，不能把
整个 4 cycle 都称为 DDR payload 等待。

在本矩阵 SELECT→ACCEPT 恒为 1 cycle；slot release 到同 generation 的 META_FREE
恒为 1 cycle，而模型逻辑回收为 0。descriptor/FIFO 回收仍以全局 snapshot 观测，
本轮没有逐 generation 重建它们的完整引用账本。

## 边界矩阵

五组资源条件分别是 `(slots_per_bank, descriptor_entries, FIFO depth)`：
`(2,2,2)`、`(2,4,2)`、`(2,8,4)`、`(4,4,2)`、`(4,8,4)`。
每组运行 no-prefetch/hit，另有 434–450 的 17 个逐拍 delay 点及错误预测、八层、
DDR gaps 三个回归点，共 30 组。所有模型性能参数保持上一轮配置。

descriptor=2 的压力实验不要求一定发生 early overlap；这通过显式的
`EXPECT_EARLY_OVERLAP=0` 记录，不删除数值、操作数量、因果、drain、OCM 和字节守恒
检查。对于有足够资源的早预测点，原 overlap 要求继续保留。

五组资源条件的预测收益分别为：

| 资源条件序号 | 主模型节省 | RTL 节省 |
|---|---:|---:|
| 0 | 70 | 64 |
| 1 | 74 | 68 |
| 2 | 82 | 76 |
| 3 | 74 | 68 |
| 4 | 82 | 76 |

这里全部为正，差值都为 6 cycle。但是只比较 no-prefetch 与固定 full-expert 计划，
没有测试所有 tile allocators 的策略排名，不能宣称“Scheduler 排名已验证”。

### 截止点：模型早丢失六拍的机会

| predictor delay | 主模型 spec payload | RTL spec payload | 是否一致 |
|---|---:|---:|---|
| 434–440 | 1536 | 1536 | 是 |
| 441–446 | 768 | 1536 | 否 |
| 447–450 | 768 | 768 | 是 |

模型在 441 开始失去 target 1 的 pair，RTL 到 447 才失去，临界位置差 6 cycle。
两个结果仍执行相同真实 tiles，因此 DMA/计算 key 顺序比较可以通过；但其中一个
pair 在模型中变成 demand，在 RTL 中仍是 speculative，分类和性能不再一致。
这也说明只比 key 顺序不足以证明请求类别和 admission 窗口一致。

delay=441–445 时，模型预取节省 41 cycle，RTL 节省 76 cycle，差 35 cycle；
446 时 RTL 节省 75 cycle。此处不能再用“总周期大致差 1%”来忽略预取收益误差。
根源是 allocator/逻辑 admission 的可见时间与 layer retirement 窗口的近似，再叠加
源层自身的时序差；本轮证明了截止点偏移，尚未把这 6 拍逐级唯一分摊。

## 同拍就绪验证

独立 `tb_boundary_readiness` 不走主端到端模型，它直接通过正常 scheduler 接口设置
GU ready 相对最后 truth 的 offset=-2/0/+2。三组 XSim 均仅接受一个请求，没有
在 truth、GU ready 或 DOWN ownership 之前执行。相对最后满足的条件，ACCEPT 都晚
3 cycle。矩阵涵盖 before/after truth；精确同拍由这三个定向实验补充，不能声称
端到端矩阵出现了 same-cycle bin。

## 能否忽略，以及下一步

- 当前非临界窗口的方向性研究：五组资源条件支持预取正收益，允许保留已知小残差。
- 精确收益、截止时刻、接近的策略优劣：不可忽略。六个临界点已经出现类别不一致。
- 下一轮模型应区分 payload-ready 与 committed-ready，表达每个 hit 的 commit 可见
  延迟；并以逻辑 pair reserve/accept 边界描述 future admission，而非只用初始
  allocator 常数和一个 layer-done 开关。不要再增加固定总周期补偿。
- RTL 可单独研究 combine 与下一 GU 的合法重叠；commit 门控是数据真实性与 ownership
  检查，不能简单删除。模型补充与 RTL 优化应分别验收，而不是互相迎合。

本轮的任务是验证和定位，没有修改以上执行语义；六点差异明确列为后续修正项。

## 文件与复现

```bash
make -C hardware/m10_u200 boundary-validation
pytest -q -o addopts=''
make -C hardware/m10_u200 rtl-test lint golden
```

第一条命令依次运行 3 个就绪测试、2 个 commit focus 和 30 个矩阵点。
已有结果的后处理可用 `ALIGNMENT_ARGS='--analyze-only'`，该模式不声称重跑 RTL。

结果在 `results/m10_boundary_validation/`：

- `boundary_summary.json`：结论、截止点、资源收益对比；功能通过与规则不一致分开。
- `gain_comparison.csv`、`late_boundary.csv`：资源条件及逐拍截止扫描。
- 每个案例的 `wait_segments.csv`、`control_events.csv`、`credit_events.csv`、
  `scheduler_blockers.csv`，以及完整 XSim 日志、input 和 hash manifest。
- `readiness_cases/`、`commit_focus/`：独立观测与各自 source manifest。

Python：318 passed。RTL 联合回归退出 0，77 条 PASS；Lint 仍有已知警告，未清零。
未进行新的综合、PnR、板级或 paper-scale DSE；这些检查不能证明 timing/CDC/signoff。
