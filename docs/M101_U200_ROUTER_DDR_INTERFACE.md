# U200 真实 Router 与四路 DDR 接口

## 适用范围

新功能入口是 `hardware/m10_u200/rtl/u200_moe_top.sv`。它强制启用真实 Router、
真实 Expert MAC、自动执行和四 bank stripe 数据通路；不暴露外部 truth 输入。
外部 predictor 仍通过 prediction/candidate 事务供给，不能改变真实路由。
旧 `tile_prefetch_predictive_axi_top` 默认保持外部 truth / 单 AXI 兼容模式，方便原有回归。

这是 core-clock RTL，不是 U200 shell、DDR PHY、XRT kernel 或已完成 PnR 的板级工程。
U200 的四个 DDR bank 不等于四个 Expert 并行计算引擎；此设计只有一条共享 MAC 数据通路。

## Router 与量化配置

1. 空闲时写 Router int8 权重：`router_weight_word_index` 从 0 顺序递增至
   `H*NUM_EXPERTS/16-1`，128-bit word 的最低 byte 对应第一列。
   最后一个 word 才置 `router_weight_last`；错序或 last 不合法置 sticky error。
2. 权重布局为 output-major `W[NUM_EXPERTS,H]`，常驻独立同步 RAM。
   `router_weight_reload_valid/ready` 设置 layer tag 并清除 loaded/error；必须重写全部权重。
   默认 layer tag 为 0。运行中不能 reload/write。先加载，再提交 prediction begin。
3. prediction begin 同拍锁存 activation、GU requant shift、`router_top_k_config` 和
   `router_logit_shift`。K 范围是 `1..min(NUM_EXPERTS,8)`。未加载、层 tag 不符或非法 K
   拒绝开始并置 `config_error`。
4. Router 执行 signed int8 GEMV，int32 二补码 wrap 累加，复用 Expert 的 16-lane
   product/reduction/accumulator。Router 完成前不执行 Expert；DMA 可以并行运行。
5. Top-K 按 signed logit 降序，完全同分时 Expert ID 小者优先。
6. 对选中 logits 做稳定 softmax：`d=(max_logit-logit)>>router_logit_shift` 是 Q4 距离；
   `exp_q16=round(65536*exp(-d/16))`，d=0..128 使用固定 LUT，d>128 截为 0。
   差值用 33-bit，避免 signed int32 极值差溢出。最大项指数恒为 65536。
7. 前 K-1 个 gate 用整数 `floor(exp_q16*32768/sum_exp)`，最后一项为
   `32768-sum(previous_gates)`，所以精确归一到 32768。串行 restoring divider
   每次除法 32 cycle；没有组合除法。结果通过原有 truth parser / resolver 接入。

`router_logit_shift` 决定整数 logit 与 softmax 实数尺度的对应，必须来自模型量化校准。
它不是时序参数。GU shift 也独立提供。本规则是用户确认的初版，不宣称与 FLAME 或
任意训练模型 bit-exact；需另行提供该模型的量化 ABI 和数值参考。
示例 `hardware/m10_u200/configs/u200_core_example.yaml` 是 synthetic fixture；
当前配置通过 RTL 参数及 host 端口设置，**RTL 不直接解析 YAML**。

## 四路物理 AXI 与权重布局

packed `m_ddr_*` 接口的 bank b 位于 `b*FIELD_WIDTH +: FIELD_WIDTH`；每 bank 为
128-bit AXI4 read，ARSIZE=4、INCR、最多一个 outstanding burst。没有 AXI ID 乱序。
四 bank 可同时发 AR、独立停顿和接收 R；一个逻辑 burst 在四路全部返回后才完成。

host 将虚拟连续权重按 16-byte stripe 散布到四 bank：

```text
bank          = (virtual_address / 16) % 4
local_address = (virtual_address / 64) * 16 + virtual_address % 16

virtual bytes 0..15   -> bank0 local bytes 0..15
virtual bytes 16..31  -> bank1 local bytes 0..15
virtual bytes 32..47  -> bank2 local bytes 0..15
virtual bytes 48..63  -> bank3 local bytes 0..15
virtual bytes 64..79  -> bank0 local bytes 16..31
```

这里的 local address 是各 AXI bank 的相对地址；平台若采用不同 DDR base，应在
board wrapper 中加对应 base。不能把旧单-bank 连续布局直接送入本模式。
GU/DOWN base、stride、tile bytes 必须 64-byte 对齐；MAX_BURST_BYTES 是
64 的倍数、范围 64..4096。虚拟 4KB boundary 限制保留，因此不会跨物理 bank 4KB。
每 bank burst bytes 是虚拟 burst bytes/4；ARLEN=`burst_bytes/64-1`。

四个弹性 beat buffer 合并成一个 512-bit OCM write，bank0 放最低 128-bit。
OCM 支持 512-bit 带 byte strobe 写和 128-bit 同步读，供共享 MAC 使用。
RRESP/RLAST 错误累计并使 transaction 失败；soft-flush 让已发 AXI burst drain，
epoch/generation 不符的返回不能覆盖新 ownership。平台不能把 soft-flush 当 AXI reset。

## Credit 和带宽边界

allocator 使用 `CHANNEL_COUNT=1` 聚合 credit，而不是四个 Expert-to-bank credit。
每个 tile 平均分布到四 bank，因此 host 的 aggregate credit 不应超过
`4*min(per_bank_credit)`，并且仍受 OCM、全局 speculative budget、descriptor credit 限制。
四路不能独立调度不同 descriptor；当前每 bank 只有一个 outstanding burst。

200 MHz 下接口理想上限为 `4*16=64 byte/cycle`，即 12.8 GB/s。
这不是实测 DDR 吞吐，更不是 U200 内存系统最大化利用的声明。AR 周期、返回停顿、
最慢 bank、burst 串行启动和 ownership 控制都会降低有效带宽。
增加原生接口宽度、多个 outstanding、board interconnect/CDC 与实际吞吐测试仍需后续完成。

## 与性能模拟器对齐

没有修改 M7–M9.3 模拟器语义或历史实验。事件级模型可以继续使用有效 ops/cycle 和
bytes/cycle，无需复制 RTL 的每个流水级；但不能仅忽略反复出现的 service cost。
共享 MAC 当前每 16-element chunk 需要 8 cycles，即有效 4 operation/cycle
（乘加计 2 operation），另有启动、非线性和控制开销。初始化 Router 权重与单次
layer 常驻假设分开记录；Router truth 由真实 GEMV + Top-K + softmax 生成，
不再使用人为 route_release_cycle 作为 GEMV latency。
DMA 有效吞吐应来自仿真/上板统计，不能将 64 byte/cycle 理想上限当作测量值。

## 已运行验证

```bash
python hardware/m10_u200/scripts/validate_design.py
make -C hardware/m10_u200 synth-ooc
```

新增测试包含 12 组 signed Router GEMV（48 个独立 dot product）、Top-K 同分和
signed 极值、不同 shift、Q15 精确和、背压、非法 K 与 flush；四路独立 AR/R
停顿、payload 背压、384-byte conservation 和 OCM 读回；完整 Router + 四 DDR +
MoE 命中/错误预测/零预算/flush/响应错误恢复五个运行点。旧数值 golden 与回归继续运行。
OOC 综合入口已改为新顶层；默认完整模型规模、板级 IO timing、PnR 和上板需要独立验收。
