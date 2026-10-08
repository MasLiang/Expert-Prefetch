# M10.1 第四轮优化：共享 MAC 的 tagged chunk 流水

## 范围与数据通路

本轮修改 `int8_swiglu_tile_backend.sv`，Router、GU 与 DOWN 继续共享同一 16-lane INT8 MAC。
没有增加第二个计算引擎，没有修改 Router truth、SwiGLU 数值规则、allocator、DMA 优先级
或主 Simulator/M10 适配层语义。demand producer 原本已经持续申请后续 tile；本轮未将
简单多发请求冒充新的 deadline-aware DMA lookahead 策略。

原 backend 将每个 chunk 的所有阶段串行执行，新数据通路为：

```text
同步 RAM 读 → INT8×INT8 → pair → quad → octet → total → INT32 累加 → row 写回
                下一 chunk 可以在上一 chunk 归约时进入
```

读取期间每周期发出一个 16-byte chunk，各级仍各自注册，不把四级加法合并到单周期。
乘法输入显式为 signed 8 bit，输出为 signed 16 bit；归约依次使用 17/18/19/20 bit，
累加保持原有 INT32 wrap。row/first/last 标签与数据一起传播，first chunk 重置行累加，
last chunk 才写回行结果。GU、Router、DOWN 都在最后一个结果实际写回后产生 done。

中间数组由本次 GU 完整覆盖后才使用。clear 无条件清空流水有效标签和控制 context，
数据寄存器无需整体 reset；残留数据没有 valid 就不能写回或产生 done。
GU_DONE 后仍采用上一轮独立 slot 回收；此时最后一次 GU RAM 读取和 GU 写回均已结束。
DOWN 权重未到时仍等待，不用流水优化绕过 payload readiness 或 generation 检查。

backend 重构为每个 always_ff 只更新一个字段；新 testbench 的逐周期计数/状态也分块。
模块例化继续使用每行一个端口。没有对不相关模块做全工程风格重写。

## 解析服务契约

同步 RAM 为一周期响应、无内部读背压。在现有事件观测边界下：

```text
Router cycles = E × H / 16 + 8
GU cycles     = 2 × F_TILE × H / 16 + 8
DOWN cycles   = H × F_TILE / 16 + 8
```

8 cycles 包含启动观测边界、同步读/流水 drain 与 done 观测，不是拟合整层差值的常数。
非线性仍逐元素计算；Router Top-K/softmax、调度、DMA 与 combine 时间不包含在以上公式。
H、F_TILE 仍必须为 16 的正整数倍，并未新增任意 tile 尺寸或 PE 数量支持。

完整默认 H=16、F=256、E=64、Top-2、F_TILE=16、每 Expert 16 tiles 时：
Router 为 72 cycles，GU/DOWN 每 tile 为 40/24 cycles；旧串行 GU/DOWN 为 256/128 cycles。
这是工程默认的小型 synthetic 数值模型，不是真实大模型的 H/F，也不代表完整大型 MoE 吞吐。
有限阶段的平均有效算力不能直接等同于持续 II=1 时的理论吞吐，更不能当作 U200 板级测量。

## 完整默认同激励结果

冻结比较基线为 `results/m10_gu_stage_release/`；四个物理 DDR bank、128 OCM slots、
64 descriptor entries 与 16 深 DMA FIFO 均不变。

| 用例 | 旧 cycles | 新 cycles | 减少 cycles |
|---|---:|---:|---:|
| demand_fast / wrong_fast | 13815 | 3072 | 10743 |
| demand_gaps / wrong_gaps | 13822 | 3083 | 10739 |
| hit_fast | 13717 | 2988 | 10729 |
| second_hit_fast | 13717 | 2987 | 10730 |
| hit_gaps / second_hit_gaps | 13717 | 2998 | 10719 |

这是相同测试硬件配置下的 RTL 仿真 cycle 对比，不是实测频率/板级 speedup。
Router 也更快完成，因此预测窗口变短；总体预取收益与控制等待不能照搬旧 MAC 的结论。
本轮未增加或更改实际传输字节，总体数值向量、truth、allocation 与字节分类仍与旧基线一致。

## 验证与证据

- 243 项 Python 测试通过，新增 backend 单字段 always_ff 检查及解析服务检查器反例。
- 59 个 RTL 回归运行点通过。新增 `tb_mac_chunk_pipeline` 的 48×48 非二次幂规模与
  16×32 非对称规模，检查连续 RAM 地址和 II=1、跨 chunk/row 累加、INT8 极值、
  busy 时无效 GU 启动不能覆盖 Router context，以及握手后 activation 保持。
- 每种规模运行 3 个 operation × 11 个 flush 时刻，共 33 个 flush/restart 组合，
  覆盖初始流水年龄 0–7、末次读取、drain 和完成边界；clear 后不能出现迟到 done/read。
  每次重启都重新检查 Router 和 GU→非线性→DOWN 数值。
- 独立 dense SwiGLU Python/RTL golden 的 16×16、32×16、16×32 三种规模与两个固定 seed
  全部通过；原有 12 个 dense Router GEMV 的独立数学参考继续通过。
- 八个完整默认 XSim 2023.2 用例均正常退出并 TEST_PASS。解析公式严格检查三类 MAC 服务，
  不使用“允许变化”跳过时长验证；同时检查全部 32 tiles、每 stage 释放一次、
  DOWN payload 因果、数值/路由/字节守恒、MAC/DMA 互斥和 OCM/FIFO/descriptor 容量。
- lint 通过，新乘法输入的位宽警告已通过显式 signed 8-bit 信号消除；仍存在既有
  unused/空输出连接/reset/文件名等警告，不宣称零 warning 或正式 lint signoff。

结果目录 `results/m10_mac_pipeline/` 保存最终 RTL/TB SHA256、激励、编译展开日志、
逐例日志、CSV 与 comparison JSON。新旧服务模式通过显式参数区分：旧控制优化入口
要求 GU/DOWN 时长不变，当前 `mac-pipeline-test` 要求新的解析流水公式，其他守恒条件不变。
旧 `ops_per_cycle=4` 是历史模型参数，此轮不运行该旧适配层，不能用它描述新 MAC 服务。

## 综合与时序边界

本轮使用现有非项目脚本检查安装的 U200 part，并对完整默认参数、5 ns 时钟进行 OOC 综合。
脚本支持独立输出路径与 `--full-default`，不覆盖旧项目或 XDC，不加入 false path 来隐藏问题。
完整综合结果另存于本轮 synthesis 目录。综合前/后源哈希必须一致，旧 DCP 不能代表新 RTL。

Vivado 2023.2 完整默认 OOC 综合已实际通过，目标 part 为 `xcu200-fsgd2104-2-e`，
所有 26 个 RTL 与 XDC 的最终 SHA256 均匹配。结果如下：

| 指标 | 实测综合报告 |
|---|---:|
| 全设计 LUT | 87187 |
| FF | 53585 |
| RAMB36 | 130 |
| DSP | 148 |
| 5 ns setup WNS | −0.866 ns |
| setup TNS | −641.457 ns |
| setup failing endpoints | 1955 |
| hold WHS | 0.058 ns |
| 无时钟寄存器 / 未约束内部 endpoint | 0 / 0 |
| 未设置 input / output delay 的端口 | 1219 / 2106 |

**200 MHz 尚未闭合。** 最差 setup 路径由 resolver 的 `scan_index_reg[3]` 出发，
经 descriptor 查询及 OCM commit/cancel 控制，进入 slot 32 的 state 寄存器 CE，
22 层逻辑、估计 data path delay 5.762 ns。不是新 MAC 的归约树关键路径。
后续应优先分段注册 resolver→descriptor/OCM 控制链，必须同步保护命令 generation、
request 引用退休与 flush 优先级；不能仅延迟输出并提前宣告 route resolution 完成。

网表保留共享 backend（4796 LUT / 3760 FF）、payload RAM（128 RAMB36）与
Router weight RAM（2 RAMB36）。backend 的 8×8 乘法由 Vivado 映射到 fabric，
该层级 DSP 为 0，不表示 MAC 被删除；全设计 148 DSP 用于其他乘法/控制数据通路。
这只是网表存在/资源证据，不等同于已经做过新网表数值仿真或等价证明。

DRC 没有 Error/Critical Warning，仍有 331 个 DSP pipelining warning：
DPIP-2 51、DPOP-3 132、DPOP-4 148，主要需按各控制算术通路继续评估；没有创建 waiver。
常量端口与位宽 trimming warning 也保留，禁用的 sequence 功能不能当成已实现多层硬件。
本次未执行 `report_methodology` 的自动修复，也未做 place/route 或新的网表仿真。

综合后的时序估计与布线后 STA 不同。OOC XDC 只有 core 时钟；外部 AXI/host IO 延迟仍由
板级 wrapper 定义，没有约束的 IO 路径必须明确记录，不能以此宣称板级 200 MHz 闭合。
本机为 Vivado 2023.2，`vivado-timing-methodology-checks` skill 要求 2026.1+，
因此未运行其新版本自动修复/JSON流程，采用现有兼容报告入口。

## 复现及剩余工作

```bash
PYTHONPATH=src pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 mac-pipeline-test
make -C hardware/m10_u200 synth-default
```

重跑使用 `results/m10_mac_pipeline_current/`，保护冻结的本轮与前几轮证据。
旧控制-only target 不适用于新计算服务；不要删除其断言强行通过。

仍需推进：有限容量/慢带宽下的 DMA urgency/lookahead、主 Simulator 新服务配置的独立校准与
正式同激励实验、真实多 layer context、可调 PE/独立 memory request 并发、
布线后时序与板级接口验证。这些不在本轮功能流水完成声明内。
