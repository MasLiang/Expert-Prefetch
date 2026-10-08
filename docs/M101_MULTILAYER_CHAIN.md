# M10.1：顺序多层 MoE 数据通路

> 更新：默认多层模式已加入跨层预取，见 [跨层预取协议与验证](M101_CROSS_LAYER_PREFETCH.md)。
> 本文记录前一阶段的全局排空方案；当前可通过 `ENABLE_CROSS_LAYER=0` 选择该兼容模式。
> 下文“未来层预取尚未实现”是前一阶段状态，不是当前默认版本状态。

## 本轮范围

新增 `moe_layer_chain` 并接入 `u200_moe_top`，实现真实数值数据依赖，而不是由外部
`layer_done` 驱动计数器。暂不做 200 MHz 综合/时序优化；本轮验证是 RTL 功能仿真。

支持综合参数 `LAYER_COUNT=1..8`。现有 layer ID 为 3 bit，因此不支持超过 8 层。
`ENABLE_LAYER_CHAIN` 默认在 `LAYER_COUNT>1` 时开启；单层也可显式开启此参数以使用新协议。
默认 `LAYER_COUNT=1` 保持已有单层接口行为。

所有层共用一套 Router/Expert INT8 MAC、OCM 与四路 DDR core-clock 接口，保留 GU
就绪即可启动规则。H、F_TILE、expert 数、tile 数与物理资源在综合时确定，所有层相同；
地址、候选、Top-K、量化配置由主机逐层提供。本轮没有修改主 Simulator 或调度算法。

## 数据流与层切换

```text
sequence_start（捕获初始 INT8 activation）
  → 当前层候选事务 + 当前层真实 Router 权重
  → 真实 Router → Top-K softmax → tile DMA/GU/Nonlinear/DOWN/combine
  → result_valid（捕获本层 INT32 输出）
  → 层间 requant（算术右移，饱和到 INT8）
  → 等旧层资源全部退休
  → 同步 context flush
  → 下一层，或 sequence_done
```

层间数值规则为 `X[l+1] = saturate_int8(Y[l] >>> shift[l])`，不做浮点缩放、
四舍五入或 residual。`layer_output_requant_shift` 为 0..31，在当前层
`prediction_begin_valid && prediction_begin_ready` 时锁存；计算期间改动不影响本层。

排空屏障通过模块端口传递，检查 demand/promoted/speculative FIFO、DMA 状态、
AXI adapter、descriptor table 引用、descriptor manager、allocator、demander、
resolver 和 compute；不能只凭 `result_valid` 切层。错误 speculative 已发起传输仍需
完成并退休。自动 flush 使用本层 accepted epoch。

## 主机握手

1. 等待 `sequence_start_ready`，提供 `compute_activation`，与 `sequence_start` 同拍接受。
2. 观察 `current_layer`；为该层通过 reload/write 接口加载完整 Router 权重。
3. 提供同层 `prediction_layer`、epoch、sequence、GU/DOWN 地址、Top-K 与量化配置，
   完成 prediction begin 和 candidate 握手。即使没有预取也需完成合法空候选事务。
4. `result_valid/result_y` 是每层输出。下一层输入由硬件自动生成，无需主机复制。
5. 切到下一层后重装对应 Router 权重，提交下一层事务。
6. 最后一层排空与清理完成后 `sequence_done` 脉冲表示整条链完成。
   `sequence_result_y` 保留最后一层 INT32 输出，直到新 sequence 被接受或复位。

`sequence_busy` 表示层链活跃，包括等待主机事务和排空阶段。
开启层链时，旧 `sequence_active/layer_done` 不控制层推进。
错误层号或后续层不同 sequence 的 begin 不接受；未加载当前层 Router 权重时拒绝开始。
层 0 的首个接受事务确定整条链的 sequence。未来层事务被 backpressure，不缓存。

外部 soft-flush 或 transaction error 中止链，不产生成功 `sequence_done`。
事务错误后需显式 soft-flush 恢复；使用正确 epoch 清理，等待旧传输退休与
`sequence_start_ready`，再重新开始。仅看到 `sequence_busy=0` 不能立即重用资源。

## 可复现验证

从仓库根目录运行：

```bash
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 multilayer-test
make -C hardware/m10_u200 mac-pipeline-test
PYTHONPATH=src pytest -q
```

`multilayer-test` 使用 Vivado 2023.2 XSim，输出
`results/m10_multilayer_current/summary.json`、源文件 SHA256 清单和各用例日志。
有 watchdog、运行超时、PASS/错误检查，任何失败均返回非零状态。

专门用例包括 controller 的有符号边界、配置锁存、延迟排空和 abort/restart；
真实 3 层、shift=1 的真实 3 层、8 层连续两次；并检查每层 Router 重新加载、
错误层号/sequence 不接受、每层真实计算次数、最终结果保留和 DDR 排空。
端到端小配置为 H=16、F_TILE=16、E=6、K=2、每 Expert 2 tiles、4 OCM slots。
独立数值参考使用固定 SiLU Q4 样本，不读取 DUT LUT。
初始 activation=16：shift=0 的三层输出依次为 24、60、254；shift=1 为 24、12、2。
这些是合成数值激励，不是模型精度或板卡性能结论。

本轮实际验证：247 个 Python 测试通过；63 个 Icarus RTL 运行点通过；
allocation golden 与 6 个 dense numeric golden 通过；4 个多层 XSim 用例及
8 个完整默认配置单层 XSim 用例通过。单层 summary、port metrics、boundary events、
tile/stage timing 和 timeline 与冻结的 `results/m10_mac_pipeline` 逐文件完全相同。
单层与开启多层的 Verilator lint 均运行，未出现 width、latch 或多驱动警告；
仍有既有的命名、空端口与未使用信号等警告，不宣称零 warning。

## 仍未实现的多层扩展

- 提前若干层预测及预取：需要多个同时活跃的 layer context、跨层 OCM 归属与预算。
- 超过 8 层的 layer ABI 和对应 transaction/parser/tag 扩宽。
- 每层不同 H/F/E、运行时层数与资源拓扑配置。
- Attention、residual、完整 Transformer 与真实模型量化校准。
- 多层主 Simulator 逐事件对齐实验、板级 DDR shell、部署及新设计综合/时序签核。

因此本轮完成的是可测试的顺序 MoE-only 多层链，不是完整模型部署或跨层预取验收。
