# M10.1 RTL 实现与验证状态（2026-10-03）

U200 新功能顶层为 `hardware/m10_u200/rtl/u200_moe_top.sv`。
旧 `tile_prefetch_predictive_axi_top.sv` 默认保持外部 truth / 单 AXI 兼容模式。
新 Router/DDR 接口、配置与布局见 [接口说明](M101_U200_ROUTER_DDR_INTERFACE.md)。
本版本已闭合 prediction、truth、DMA payload、OCM、真实 int8 SwiGLU、tile partial
output 与 Top-K combine 的主要功能路径；板级平台绑定、PnR 和 timing sign-off 单独验收。

## 已修复的错误

1. DescriptorTable 的 allocation、promotion、cancel、DMA start/done 和 reference
   retirement 可在同周期处理不同 entry，不再用全局 else-if 丢弃独立事件。
2. speculative entry 在 truth 未知时，即使 DMA 完成也继续保留。所有 FIFO token、
   DMA ownership 和 Router 引用退休后才回收；queued promotion 的第二 FIFO token
   纳入 reference count，resident commit 不重复发 DMA。
3. parser 的 begin/entry ready 与 allocator/resolver/workload 生命周期连接。
   malformed count、重复 ID、非法 rank/last、越界 Expert、错误 epoch/layer 和非法
   Q15 gating weight 显式报告错误。错误 transaction 需 host soft-flush 恢复。
4. speculative DMA 在 burst 边界暂停后保存 remaining bytes 和 generation，恢复时
   不从 byte 0 重传。已 inflight 的 promoted token 不误当 queued tombstone。
5. OCM COMMITTED 与 payload-complete 分开记录；compute 必须同时满足 committed、
   完整 payload 和 matching epoch/generation，避免提前读取 inflight 数据。
6. pair admission 使用当前 logical GU/DOWN 的 bank，而不是上一 physical descriptor
   的 bank；默认保留两个 OCM slot 给 demand。
7. flush 同时处理 descriptor、queued token、inflight quarantine、resident payload、
   resolver 与 compute transaction；旧响应无法覆盖新 ownership。
8. AXI burst 根据地址限制在 4KB boundary 内；响应错误置 transaction_error 并抑制
   成功结果。配置在 prediction begin handshake 时锁存。
9. HardwareGreedy winner 与 descriptor 地址生成之间增加寄存级；下游 backpressure
   期间保持选择和 descriptor 稳定，每次 handshake 后用更新后的 credit/depth 重新选择。
   stop/flush 立即禁止待发送请求。该修改保留分配顺序，但引入显式选择流水延迟。
10. 实际 allocator 的 logical tile 大小相同，比较分数时约去共同的正 byte 因子；
    admission 仍使用完整实际 bytes。通用 comparator 默认继续支持不同 tile 大小。
    1,024 组确定性输入验证优化前后的 winner、tie-break、channel/credit 结果一致，
    包含最高位为 1 的 32-bit byte 值，避免以截断乘法换取时序。
11. Greedy winner 选择从串行链改为平衡 tournament，比较深度从 N-1 降为
    ceil(log2(N))；保持 full-width cross multiplication、rank/depth/Expert ID tie-break
    及完全相同 key 的首行优先。不足 2 次幂的叶子显式填充 invalid。
12. 全部 WIDTHEXPAND、WIDTHTRUNC 和 VARHIDDEN 告警已消除。索引转换保留原有范围
    检查；地址、4KB boundary、byte counters 显式零扩展；Q15 combine 明确 int32
    wraparound；SiLU product 先符号扩展再移位。metadata reset 改用 nonblocking。
13. OOC 目标改为 200 MHz（5 ns）。真实 MAC 的乘法、17/18/19/20-bit 平衡归约、
    int32 累加与结果存储分级；每个 16-lane chunk 使用 8 cycles，数值 ABI 不变。
    Q15 combine 分为 sum capture、multiply、layer writeback，完成脉冲只在写回产生。
14. 权威 allocator 启用 registered Greedy：候选打分和每层 tournament 分级，
    selection snapshot 到完成为 ceil(log2(N))+1 cycles，再由 scheduler 消费。
    一次只允许一个 selection inflight；handshake 更新 credit 后才启动下一次。
    stop/新 transaction 撤销旧 completion；默认 combinational 模式保留作独立对照。
15. ready-first 扫描到 OCM admission 之间新增稳定请求寄存级；speculative 字节
    finalizer 的 delta 先寄存，再加入 64-bit counters。delta 使用
    32+ceil(log2(ENTRIES)) bits，最终分类可见时间增加 1 cycle，不改变最终字节数量。
    上述延迟均是真实 RTL cycle latency，不能仅凭频率翻倍声称 layer latency 减半。

## 新接通的真实数据通路

```text
Prediction transaction
  -> parser -> HardwareGreedy candidate/prefix allocator
  -> GU/DOWN atomic descriptor admission -> token DMA arbitration
  -> AXI4 read -> tagged synchronous OCM payload RAM

Router truth -> parser -> token resolver -> commit/cancel
                           -> missing-prefix demand generator

physical admission + DMA completion + truth
  -> ready-first tile execution scoreboard
  -> 16-lane int8 MAC -> Q4 SiLU LUT / SwiGLU -> int8 DOWN MAC
  -> per-Expert int32 vector accumulation -> unsigned-Q15 gating combine
  -> result_valid / result_y
```

tile_compute_controller 的 timer 模式仅供旧 ABI 回归；权威顶层默认
USE_REAL_COMPUTE=1、AUTO_COMPUTE=1，实际 done 由数值 backend 驱动。
partial_output_accumulator 的旧 scalar 模块保留为兼容模块；实际向量累加在
tile_execution_scheduler 中完成。

USE_SEQUENCE_CONTROL=1 接入 six-layer controller；默认 0 由外部 host 控制 sequence。
controller 的 compute/bandwidth transition gate 独立计数，sequence 完成后可再次 start。
新顶层 Router 与 Expert 共享 int8 MAC，内部完成 Top-K selected-logit softmax，
量化 shift 由配置提供；predictor 仍由外部 transaction 接口供给，不包含 predictor NN。

## 数值 ABI

- H 与 F_TILE 是 16 的正整数倍；默认 H=16、F_TILE=16。
- GU payload 是 output-major G[F_TILE,H] 后接 U[F_TILE,H]，
  DOWN 是 output-major W[H,F_TILE]。
- GU bytes=2*H*F_TILE，DOWN bytes=H*F_TILE。128-bit beat 的最低 byte 对应 lane 0。
- host 提供 activation 和 GU requant shift；G/U 饱和为 Q4 int8。256-entry SiLU LUT
  返回 Q4 int8，SiLU(G)*U 算术右移 4 后饱和为 int8。DOWN 输出 int32。
- tile partial output 与 Expert sum 为 int32 二补码 wraparound；先按 Expert 完成全部
  tile，再乘 unsigned Q15 gating weight、右移 15，加入 layer output。
- 地址 stride、容量和量化格式由 host 明确提供，不假定训练模型已按此 ABI 量化。

## 可复现检查

```bash
make -C hardware/m10_u200 rtl-test
make -C hardware/m10_u200 golden
make -C hardware/m10_u200 lint
make -C hardware/m10_u200 synth-ooc
pytest -q
```

RTL 回归包含原有模块测试及真实数据路径矩阵：late/early Router、命中、全错预测、
零预算、第二 Expert 命中时 ready-first、四个 flush offset、AXI error 和 sequence restart。
数值端到端测试检查完整输出向量、四次 compute completion、demand bytes、实际
speculative bytes 与 useful/wasted/cancelled conservation。
当前 rtl-test 包含 41 个测试运行点；完整 Python 回归为 207 个测试。
新增 Router GEMV、selected softmax、四路 DDR/宽 OCM 和五组真实 Router 端到端验证。
新增 allocator 测试直接检查 backpressure 稳定性、地址/prefix/credit 更新和 stop。
平衡树另对照独立串行参考：combinational N=1/3/5/16，registered N=1/5/16，
包含单 channel、padding、同分和大 byte 值，共 7,168 点。新增 combine stage-wise
flush 与三条 UINT32_MAX 已完成记录同时分类测试，验证 64-bit 总和与 exactly-once。
验证脚本将位宽/变量遮蔽告警纳入失败条件，不依赖关闭告警。

golden 真正编译并运行 RTL：5 组 allocation trace；6 组 seeded dense numerical
reference，覆盖 16x16、32x16、16x32。它不再只是 Python 对照静态 JSON。

## 当前真实 Router / 四 DDR 顶层综合结果

Vivado 2023.2 对 `u200_moe_top`、`xcu200-fsgd2104-2-e` 执行 OOC synthesis，
配置为 4 candidates、64 Experts、4 tiles/Expert、16 descriptor entries、2 OCM banks ×
4 slots、512 byte/slot、H=16、F_TILE=16、一个 allocator aggregate credit class。
真实 Router、共享 MAC 和四物理 AXI banks 均开启。

- 0 errors、0 critical warnings；26,271 LUT、21,708 FF、2 RAMB36、64 RAMB18、64 DSP。
- 5 ns 目标：WNS=+0.371 ns、TNS=0、0 setup failing endpoints；
  WHS=+0.046 ns，hold/pulse-width failing endpoints 为 0。
- 41 个 RTL 运行点、207 个 Python 测试、5 allocation trace、6 dense numeric golden 全通过。
  lint 检查新旧顶层，共 208 个告警；132 UNUSEDSIGNAL、8 UNUSEDPARAM、
  49 PINCONNECTEMPTY、4 DECLFILENAME、14 BLKSEQ，另 1 SYNCASYNCNET 来自 striped
  adapter 的仿真断言同步读取异步 reset。没有位宽、变量遮蔽、锁存、多驱动或组合环告警。
- 26 个 RTL 文件的 SHA256 与 synthesis manifest、functional validation 均一致。
  这是未布局布线的缩小配置检查，不代表默认规模或板级时序闭合。

## 历史单 AXI 兼容顶层综合结果

以下为接入真实 Router/四路 DDR 前的结果，**不可用作新顶层时序结论**。
当前综合脚本改用 u200_moe_top；最新指标以 reports/synthesis_*.rpt 为准。

Vivado 2023.2 对已安装器件 `xcu200-fsgd2104-2-e` 的缩小配置执行 OOC synthesis：
4 candidates、4 tiles/Expert、16 descriptor entries、2 banks × 4 slots、512 byte/slot。
真实 compute 与 automatic scheduling 开启，H=16、F_TILE=16。

- 当前综合完成，0 errors；使用 24,027 LUT、16,245 FF、16 RAMB18、64 DSP。
- **5 ns（200 MHz）目标**下 WNS=+0.369 ns、TNS=0 ns，0 个 setup failing endpoints；
  WHS=+0.065 ns，hold 与 pulse-width failing endpoints 均为 0。
- 历史 10 ns 优化链：WNS=-10.755 -> -6.548 -> -2.492 -> +1.350 ns。
  本次 5 ns 优化链：原 RTL -3.618 ns / 1,116 个 failing endpoints；MAC/combine
  分级后 -3.399 ns / 100 个；Greedy 和 ready-first 分级后 +0.041 ns / 0 个；
  字节统计分级后 +0.369 ns / 0 个。最终最差路径位于相邻 Greedy tournament 级之间。
- 这些是未布局布线的综合估计；缩小配置的 200 MHz OOC 目标已满足，
  **不等于默认规模或板级 PnR 时序闭合**，不据此宣称实测上板频率。
- 源码 SHA256 同时写入 functional validation 与 synthesis manifest，避免旧报告
  被误用于新 RTL。报告位于 `hardware/m10_u200/reports/`。

## 尚未达到的完整规格验收

- FullExpert atomic whole-expert 和 BudgetProportional policy 尚未接入此数值顶层；
  当前实现 HardwareGreedy，并以 budget=0 运行 NoPrefetch。
- 四物理 bank AXI 已接入，新顶层 CHANNEL_COUNT=1 表示 aggregate allocator credit；
  board DDR PHY/shell/CDC、更多 outstanding 和有效带宽测量仍未完成。
- lint 同时检查新旧顶层，当前总计 208 个告警；无位宽/变量遮蔽/锁存/多驱动/组合环告警。
  剩余 BLKSEQ 包含临时数值运算与 flush 优先级处理，不未经并发验证强行改写。
  此前“一 always 一变量”的风格要求尚未全部落实。
- 全参数、全并发 deterministic matrix 与乱序 exactly-once 模式仍未完整覆盖。
- 数值路径保持每个 Expert 的 prefix tile 顺序，支持不同 Expert 的 ready-first。
- 缩小配置的器件级 OOC 综合不等于默认规模、U200 XPFM、板级 IO/CDC 或 PnR sign-off；
  新旧缩小配置均通过各自 OOC timing 检查，但不是完整规模或板级 sign-off。

该版本可以作为已运行、已验证主要路径的真实 RTL 基线，尚未通过完整 v1.0 freeze gate。
