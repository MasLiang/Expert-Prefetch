# 完整 Switch 的 MIG/RDIMM 数字时序对照

## 测量目的与范围

上一轮完整 Switch 使用 AXI 行为响应器。新验证保留同一调度器、MAC、OCM、
H=768、F=3072、E=8、Top-1、六层、16-lane 和四个 case，只替换 DDR 后端。
`tb_switch_full_size.sv` 的 `M10_SWITCH_DDR4` 宏选择四路
`ddr4_bank_backend`，默认不定义该宏仍使用原行为响应器。

每路路径为 `RTL AXI → async AXI CDC → DDR4 controller/PHY BFM → RDIMM model`。
不是简单给行为响应器加入固定 delay。DDR 控制器排队、命令调度、刷新、
不同地址访问与 CDC 会改变首拍延迟和实际传输间隔，也可能改变 speculative
完成/取消时刻，进而改变总读取量；不能假定只是原周期加一个常量。

本验证使用 Vivado 2023.2、既有 DDR4 IP 配置：RDIMM
`MTA18ASF2G72PZ-2G3`、tCK=833ps、72-bit 物理接口、ECC、每路128-bit AXI。
PHY 的 simulation mode 为 BFM，是**数字时序模型，不是板级实测或模拟电气时序**。
128-bit AXI 和当前 DMA 并不代表 U200 带宽已最大化。

## 初始化与验证合同

每个候选 Expert 的 UP/DOWN 权重都通过 AXI 写入，并逐字读回检查。
六层八 Expert 的预载为 216MiB，四路各54MiB；这是模型 fixture 的初始化，
不是 decode 所需实际读取量。保留 dense payload 和全部MAC计算，不跳零。
预载、读回和校准均排除在四 case 的性能计时区间之外。
第一层 begin handshake 之前，必须观察四路校准、初始化完成。

除原 full-size 的 Router、逐行输出、144 GU/DOWN、自检和 byte/OCM 守恒外，
新增检查四路真实 DDR read command、逐 burst AR/首 R/末 R 的因果顺序、
128-bit bank 带宽守恒、CSV 物理bytes与 DUT telemetry 一致。
首拍延迟输出 min/max/mean，不能用平均延迟冒充每个 burst 固定时序。
厂商模型不受 XSim 支持的嵌套 assertions 数量另列，不能声称厂商断言全覆盖。

原 DDR example 只读打开后保存为新目录独立副本；不会修改原项目、RTL/XDC。
输出 summary 保存被测源码的运行前后hash。准备失败、运行超时、数值错误、
缺少任一 DDR 证据均不标作 PASS。`--mini` 是接入预检查，不能作为完整模型性能。

## 运行

```bash
# 小尺寸接入检查；不是完整尺寸结果
python hardware/m10_u200/scripts/run_switch_ddr4.py \
  --mini --layers 2 --jobs 4 --timeout 1200 \
  --output results/m10_switch_ddr4_mini_new

# 完整尺寸四策略；每个 case 最长四小时 wall-time
python hardware/m10_u200/scripts/run_switch_ddr4.py \
  --jobs 4 --timeout 14400 \
  --output results/m10_switch_ddr4_full_new
```

必须使用不存在的新输出目录，不能覆盖此前或正在运行的结果。
默认 `--example-project` 使用 `results/m10_ddr4_controller/example/ddr4_0_ex/ddr4_0_ex.xpr`；
该项目必须包含既有 DDR4 和 CDC IP。新建 IP 不属于这个脚本的工作。
相同尺寸的四组 DDR 结果相互比较；此前行为响应器的 1.654×/1.508×/1.502×
不是 DDR 结果，不能直接搬过来。

## 输出与判断

根目录 `summary.json` 和 `comparison.csv`；每 case 下保存编译/elaboration/
simulation 全日志、timeline.csv、ddr_bank_0..3.csv、summary.json。
小尺寸和完整尺寸必须分别检查 summary 中的 `full_switch_dimensions`。
只有终止 `status=PASS` 且请求的全部 case 都通过才构成该次验收。
正在运行的结果不能给出完整性能结论。

## 已运行结果与未运行项

接入预检查目录为 `results/m10_switch_ddr4_preflight_ready`，
**H16/F32/E8/Top-1、两层**，四组全部 PASS。同尺寸行为后端回归目录为
`results/m10_switch_behavioral_backend_regression`，均不是完整模型。

| Case | 行为响应器 cycles | MIG/RDIMM cycles |
|---|---:|---:|
| A：整 Expert 无预取 | 737 | 1,004 |
| B：streaming 无预取 | 659 | 780 |
| C：完整 Expert 预取 | 551 | 716 |
| D：tile 预取＋streaming | 512 | 571 |

每路 bank 的 AXI AR 接收到第一拍 R handshake 为 38–41 个 **200MHz core cycles**，
即 190–205ns；包含 CDC、控制器/模型与本测试的接口条件，不是 DRAM 的裸 CAS latency。
本预检查每个 bank burst 为 4 beats/64B，不能据此宣称完整模型大流量的有效带宽。
DDR 模型在 workload 区间确实观察到 READ/ACT/PRECHARGE；REFRESH count 为0，
因此本短测试不证明长时间刷新影响。每 case 披露8条厂商不支持的嵌套 assertion 警告。
独立 payload、backpressure、数值和容量检查保留；没有降低错误 severity。
Python 回归 **512 tests passed**；原行为后端四组全部 PASS。
运行后补充并单测了超时进程组清理，防止长运行超时留下后台 XSim；
已保存结果中的 source hash 保留实际运行时版本，不改写成后续脚本版本。

MIG 四 case 单组编译/elaboration/运行 wall-time 约321–353s，实际仿真结束时间
约35–38us。内存约1.2GiB/运行进程，四组彼此隔离。完整尺寸 payload 及持续
时间大很多，线性粗估可能达数十小时或天级；这只是资源计划估计，不是已测耗时。
默认四小时 wall-time 上限可能不足，超时会标为 FAIL，不会提供一个伪造速度比。

完整 H768/F3072/E8/六层原导出目录为 `results/m10_switch_ddr4_full_prepared`。
用户随后授权后台四组长跑，目录为 `results/m10_switch_ddr4_full_background_20261006`。
2026-10-07 用户要求停止并删除该组实验：已核验并停止驱动及全部四组 XSim，
将上述两个完整尺寸目录和三个启动日志移入回收站；这些路径已不在工作区中。
终止时仍处于权重预载阶段，没有产生完整 MIG 性能结果，也不能计作功能 FAIL。
小尺寸 DDR 证据与共享验证代码保留；当前切换到四通道 AXI 行为模型。
新的六层四策略重跑目录为 `results/m10_switch_axi_six_layers_20261007`，
必须以其终止 summary 为准，不能把 AXI 结果标为完整 MIG/RDIMM 实跑。
