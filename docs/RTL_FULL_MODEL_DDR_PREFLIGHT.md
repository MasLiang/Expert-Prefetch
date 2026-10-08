# 完整模型 RTL 仿真：DDR 后端与当前验收边界

## 核验结论

2026-10-06 检查 Vivado 2023.2、XSim、当前 RTL、数值 testbench 和独立 DDR
验证工程。完整 Switch 尺寸的端到端 RTL 仿真**尚未完成**，不能拿小尺寸
三层自检的周期或宽候选 allocator 自检的 PASS 代替完整模型性能。

## 当前 DDR 并非所有用例都启用时序后端

`tb_u200_moe.sv` 在没有 `M10_DDR4_MODEL` 宏时使用 behavioral AXI read
responder，可配置 beat gap。它不是 DDR4 控制器，也不是 DDR vendor VIP。

启用该宏时，每个 DDR bank 接入 `ddr4_bank_backend`：

```
RTL 四 bank AXI read
  → 异步 AXI clock converter
  → DDR4 MIG controller（128-bit AXI）
  → PHY simulation BFM
  → RDIMM/RCD/SDRAM 数字时序模型
```

独立工程实际 readback：part `xcu200-fsgd2104-2-e`，`Simulation_Mode=BFM`，
四 bank，各 bank 单 rank，72-bit RDIMM 含 ECC。BFM 不等于 AXI VIP；此处
保留 DDR 命令与数字时序，但不包含真实 PCB/PHY 裕量和板上 shell 的所有行为。
vendor FIFO behavioral 模型还警告不建模全部同步延迟，因此这些周期也不能
直接称为板上实测周期。

## 完整尺寸尚缺的验证工作

目标应先定义为此前研究的 Switch-base MoE 子图：H=768、F=3072、8/16
Experts、Top-1、两矩阵 ReLU、24 个 F=128 logical tiles、六个 MoE 层。
这不是完整 Transformer 的 attention、norm、residual 等端到端推理。

顶层 `u200_moe_top` 已有 H、F_TILE、Expert/tile/candidate 数和 OCM 参数；
不能简单据此声称真实尺寸已经验证，也不能反过来断言 DUT 只能支持 H=16。

现有 `tb_u200_moe` 固定或假定：

- H=16、F=32、E=6、每 Expert 两个 tiles；本次 DDR recheck 是 Top-2 SwiGLU。
- 最大两个候选、512-byte OCM slot、固定小型 descriptor byte checker。
- 固定的 Router 权重、激活、数值参考结果、二维 tile admission 统计数组。
- 权重地址中 Expert stride=4096，以及两 tiles 的初始化范围。
- DDR fixture 只给固定的每层/Expert GU/DOWN 范围写入、逐字读回。
- 1ms watchdog；真实尺寸计算和通过 AXI 初始化大权重需要重新估算上限。

完整尺寸所需工作是新增可配置验证激励与独立数值参考、无重叠的 DDR 地址
布局和初始化范围、匹配 slot/OCM 的资源配置，以及合理的有界运行时间。
必须真正跑 DMA、RAM 和 MAC，不能只增大 allocator 的 byte credit。

按目前 Switch F=128 tile：Up 与 Down 各 98,304 byte，logical tile
196,608 byte。固定 slot 应至少容纳一个 physical fragment；如果选 128KiB
slot，则容量按 128KiB 而非 payload 98,304 byte 占用，主模拟器也须同样配置。
不同 slot 几何与此前 byte-OCM 实验不能直接比较速度比。

目前 layer ID 为 3 bits，链控制器明确支持 1..8 层，六层 Switch MoE 子图
可落在此范围内。更长真实模型的所有层不能直接声称已支持，需要扩展设计。
Uniform/浮点 Ranked 也没有接入 RTL：端到端比较应先采用原生整数
HardwareGreedy-v1 + Pair-first，或者另行授权扩展 RTL 策略。

## 本次重新运行的验证

独立输出目录：`results/m10_ddr4_recheck_20261006/cases/no_prefetch/`。
从已有 DDR example project 导出新脚本，重新编译当前 RTL，配置
`LAYERS=3 FUTURE_MODE=0 BUDGET=0 DDR_GAPS=0`，启用 `M10_DDR4_MODEL`。
没有修改 RTL/XDC，也没有保存覆盖交付工程。

验收必须包含：明确三层 PASS/$finish、四 bank 校准和初始化、数值
24/60/254、payload 9,216 byte、burst 服务样本与带宽守恒。周期定义是第一层
开始到最后一层结果，排除校准、初始化及结果后的资源清理。

重新验收命令：

```bash
python hardware/m10_u200/scripts/summarize_ddr_recheck.py \
  results/m10_ddr4_recheck_20261006/cases/no_prefetch
```

`xsim/` 保留 compile/elaborate/runtime 日志和 DDR bank CSV。
验收摘要明确记录 `full_switch_model_verified=false`。

本次实际验收 **PASS**：first begin cycle=7181，final result cycle=9249，
共 **2068 cycles**，core clock=200MHz，等效仿真时间 **10.34µs**；四 bank
共 **144 bursts / 9216 payload bytes**。三层结果为 **24、60、254**。
DDR 观测到每 bank 36 reads、18 activates、2 refreshes、13 precharges。
XSim 报告 CPU usage=156310ms，peak memory=1465776KiB（约1.40GiB）。
这些资源开销包含仿真初始化/校准，不能按模型大小简单线性预测完整运行时间。
Python 回归：**496 passed，14 条现有依赖弃用 warning**。

沙箱内 XSim 在 HDL 运行前出现 `unexpected exception when evaluating tcl
command`；环境 doctor 没有 hard failure，同一 snapshot 改为沙箱外执行
成功进入 HDL/校准，没有改动 DUT 或断言。已查阅 AMD 搜索结果 AR 70455
（旧版退出时 double free）和 AR 63984（旧版 Ubuntu DPI），都不能解释本次
2023.2 启动失败；没有应用 vendor patch。vendor 部分校准 SVA 被 XSim 标为
不支持，需保留限制，不能声称完整 assertion coverage。
