# M10.1：Switch 两矩阵 ReLU Expert 开关

## 选择方式

顶层 `u200_moe_top` 增加**综合/展开时**参数 `SWITCH_RELU`，默认 `1'b0`。
这不是 host 在 transaction 中切换的寄存器；修改该参数需要重新综合/生成 bitstream。
整个配置中的所有层使用相同 Expert 类型，暂不支持同一 bitstream 内混用两种类型。

```systemverilog
u200_moe_top #(
  .SWITCH_RELU(1'b1)
) dut (
  // 原顶层接口保持不变；按工程要求每行连接一个端口。
);
```

| 参数 | 数值路径 | 第一矩阵 payload | nonlinear |
|---|---|---|---|
| SWITCH_RELU=0 | fused Gate-Up SwiGLU | Gate[F,H] 后接 Up[F,H] | Q4 SiLU(G)×U，右移4，int8饱和 |
| SWITCH_RELU=1 | Switch/ReLU FFN | Up[F,H] | ReLU(int8 requantized Up) |

Switch 模式为 `Y = ReLU(X·W_up)·W_down`，不含 pruning、bias 或 CEPR 实现。
Router 仍使用原 int8 GEMV、Top-K softmax Q15，与 Expert 共享同一 MAC；predictor 仍为外部输入。

## 数值契约

两种模式都使用 signed int8 权重/输入、int32 wrap MAC。Switch Up 累加结果
算术右移 `gu_requant_shift`，饱和到 [-128,127]，再执行 ReLU；Down 接收 [0,127] 的
int8 中间量，输出 int32。Layer requant 和 Q15 gating/combine 保持原规则。
没有复现预训练模型的完整量化校准，也未增加独立 Down requant/bias。

第一矩阵只执行 F_TILE 行，不读取第二份 Gate/Up 数据。为保持源文件列表和层级兼容，
数值模块仍叫 `int8_swiglu_tile_backend`，但内部由参数选择真实 SwiGLU 或 ReLU 路径。
Router 运算不受此开关影响。MAC 流水、GU 提前启动、延迟 Down、flush 和 slot 释放机制共享。

## 布局、预算与容量

旧接口名 `gu_base_addr`/`MATRIX_GU` 保留，Switch 模式中代表 `W_up`。
`physical_tile_bytes` 仍为单矩阵 H×F_TILE 个 int8 权重 bytes。
`GU_BYTE_RATIO` 默认由开关推导：Switch=1，SwiGLU=2。若手动设置该参数，必须匹配类型；
真实计算模式在 transaction 配置校验时拒绝不一致的比例。

- GU/Up 地址：`gu_base + expert*stride + tile*GU_BYTE_RATIO*physical_tile_bytes`。
- Down 地址：`down_base + expert*stride + tile*physical_tile_bytes`。
- pair logical bytes：`(GU_BYTE_RATIO+1)*physical_tile_bytes`。
- 同一比例贯穿当前 demand、当前 predictor allocator 和跨层 prefetch allocator。
- OCM 仍为固定 slot，Up/Down 各占一个 slot；Switch 减少 payload，不自动合并 slot。

H=16、F_TILE=16 时，Switch Up=256 bytes、Down=256 bytes、pair=512 bytes；
SwiGLU GU=512 bytes、Down=256 bytes、pair=768 bytes。Host 必须按选中模式提供预算和布局。
`WAIT_FULL_EXPERT` 独立开关仍可使用，整 Expert baseline 和 tile streaming 都适用。

## 验证与复现

从仓库根目录运行：

```bash
make -C hardware/m10_u200 switch-test
make -C hardware/m10_u200 switch-four-cases
make -C hardware/m10_u200 rtl-test golden lint
pytest -o addopts='' -q
```

`switch-test` 验证 dense signed int8、正/负/零输入、饱和、shift=0/3/31、准确 payload 数量，
两种矩阵尺寸的 II=1 与 33 个流水 flush/restart，以及单层命中/错误预测、AXI错误、
soft-flush、3/8 层级联和 activation 依赖。原 SwiGLU 默认回归保留。

`switch-four-cases` 使用 `configs/hardware/u200_switch_four_case_validation.yaml`，运行主
UnifiedSequenceSimulator 的 `expert_architecture=relu_ffn` 与相同约束的真实 RTL。
配置为 H16/F32/E6/Top-1、3层、2个 F tiles、四 bank 行为 AXI、有限 OCM，
完整 Expert 预算1024 bytes、tile pair预算512 bytes；3带宽×2预测模式×4case。
结果与源码哈希保存在 `results/m10_switch_four_cases/`，不覆盖历史 SwiGLU 实验。
性能验收检查真实 speculative admission、两矩阵 bytes、数值结果和 stage 唯一性，
不能只凭数值正确就宣称预取有效。

这只是小型 Switch 结构数值夹具，不是完整 Switch-base 模型、DDR PHY 或 U200 板上性能。
综合结果属于 OOC 可综合性检查，不代表 200 MHz PnR、CDC 或板级验收。

## 本轮验收结果

- Python：462 个测试通过。
- Icarus 辅助回归：89 个 RTL 运行点和 7 个 Python/RTL golden 检查通过，包含原 SwiGLU 回归。
- Vivado 2023.2 XSim：3 个 Switch backend 检查、4 个 promotion 边界检查和 24 个四 case
  性能/数值检查通过；新增 admission 检查同时覆盖当前层与 future 权重布局。
- 默认、Switch、Switch+whole 模式 lint 无 WIDTH/LATCH/MULTIDRIVEN/UNOPTFLAT 问题；
  仍有未使用信号及遗留 reset/赋值风格等告警，不声称 warning-free。
- 最新源文件在 xcu200-fsgd2104-2-e 上完成 Switch 3层小型 OOC 综合：0 error、
  0 critical warning，仍有普通综合告警。DCP、utilization/timing/DRC、源码哈希在
  `results/m10_switch_four_cases/synthesis/`。未进行 PnR 或 post-synthesis 数值仿真。

中等带宽、全部命中的代表性结果（加速比相对 A）：

| Case | 主模型 cycles | RTL cycles | 主模型加速比 | RTL 加速比 |
|---|---:|---:|---:|---:|
| A 整 Expert、无预取 | 921 | 898 | 1.0000× | 1.0000× |
| B streaming、无预取 | 723 | 718 | 1.2739× | 1.2507× |
| C 整 Expert 预取 | 688 | 672 | 1.3387× | 1.3363× |
| D streaming、tile 预取 | 620 | 610 | 1.4855× | 1.4721× |

D 相对 B：主模型 1.1661×、RTL 1.1770×。全部命中时三个供数点均为 D、C、B、A
从快到慢。严格 >1 的正收益/非正收益判断一致，但零收益与微小负收益不代表相同总周期；
原始数据保留在 `comparison.csv`。这些值仍只适用于本小型夹具。
全矩阵相对主模型的最大总周期差为 7.89%，不是逐 cycle 完全对齐。

实现期间发现并修正跨层 allocator 的遗留硬编码 `GU_BYTE_RATIO(2)`。
仅修改当前层 demand/backend 会让 Switch 的 512-byte 预算无法发出任何 future pair，
因此本次验收新增“必须存在 speculative admission”和逐 descriptor byte 检查。
