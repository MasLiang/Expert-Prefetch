# M10.1：接入 DDR4 控制器与 RDIMM 模型

## 目的与边界

本轮将现有四路 AXI 权重读口连接到四套真实生成的 AMD DDR4 controller IP 与 RDIMM 数字模型，不再由固定周期/周期取模的简单内存响应器决定读延迟。

接入位置是 **验证环境**，不是交付板级 wrapper。现有设计 RTL/XDC 和 `UnifiedSequenceSimulator` 未改动；原有快速后端继续用于大规模回归。不能据此宣称板上运行、完整 U200 shell 集成、200 MHz timing closure 或真实 DDR 带宽实测。

## 数据通路

```text
共享 Router/Expert MAC 与 OCM
            |
    四 bank striped AXI adapter
            |
    每 bank：128-bit AXI，200 MHz core
            |
    AXI Clock Converter（异步，3 级同步）
            |
    DDR4 controller（约 300.12 MHz UI）
            |
    RDIMM model（RCD + 18 个 x4 SDRAM，单 rank，含 ECC）
```

四套控制器独立接受 AR、返回 R；原设计等待四路合并，最慢 bank 因而能影响整个 logical burst。当前每 bank 仍最多一个 outstanding burst，本轮不新增多 outstanding 或更宽数据口。

## 配置来源

U200 文档列出四个单 rank DDR4 DIMM，每个为 16 GB Micron MTA18ASF2G72PZ-2G3B1，支持最高 2400 MT/s。[AMD UG1289](https://docs.amd.com/r/en-US/ug1289-u200-u250-reconfig-accel/DDR4-DIMM-Memory)

Vivado 2023.2 使用其 catalog 对应的 `MTA18ASF2G72PZ-2G3` 配置，明确读回：

| 项目 | 本轮设置 |
| --- | --- |
| FPGA part | `xcu200-fsgd2104-2-e` |
| 内存类型 / rank | RDIMM / 1 |
| DQ / DQS | 72 / 18 bits，ECC 启用 |
| SDRAM density / component width | 8 Gb / x4 |
| DRAM tCK | 833 ps，约 2400 MT/s；整数 ps 舍入，不是精确 2400.000 MT/s |
| reference input period | 3332 ps |
| AXI width / size | 128 bits / 16 bytes |
| address width / mapping | 34 bits / ROW_COLUMN_BANK |
| AXI arbitration | RD_PRI_REG |
| PHY simulation | BFM |
| core clock | 200 MHz |

`configs/ddr4_0.tcl`、`configs/ddr_axi_cdc.tcl` 为 Vivado 实际导出的 IP 重建参数；`ip_readback.txt` 对配置进行逐项验收。生成的 controller/model/firmware 保留在 results 内并带 vendor 原始许可说明，不将加密 vendor 模型复制进项目源码。

BFM 替代的是 PHY 仿真实现，不是把 controller 或 SDRAM 替换成固定延迟内存。仍保留命令调度与 SDRAM 数字时序；但没有板级 PCB 延迟、模拟信号完整性、真实 shell width conversion/互连争用，也不能验证实际校准裕量。[AMD PG150 PHY 参数](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/PHY-Only-Parameters)

## 初始化与独立数值检查

1. 等待每路 controller 的 `init_calib_complete`。
2. 在 UI 时钟域同步释放 AXI reset；随后释放 core 侧跨时钟桥 reset。
3. 通过 AXI 写所有测试层和六个 Expert 的 GU/DOWN 权重，无 force、deposit、RAM backdoor。
4. 每个区域逐字读回，检查数据、RRESP、RLAST；首个 R beat 出现后故意保持 RREADY=0 三个周期，检查 VALID/data/last 保持。
5. 四路全部完成初始化后，才启动 Router/MoE 共同激励。
6. 工作负载阶段继续逐 beat 验证 stripe 地址对应的真实读回 payload；原有 Router、Expert、OCM、跨层依赖、speculative byte 守恒及 MoE 数值检查保持启用。

三层小型夹具为 H=16、F=32、E=6、K=2、INT8、两 F tiles/Expert、有限 OCM，逐层输出必须是 24、60、254。这是功能和服务时间验证，不是大模型 benchmark。

## 周期定义与输出

运行无预取、命中预取、错误预取三组固定用例。
三组都显式设置当前层 `BUDGET=0`，只通过 `FUTURE_MODE=0/1/2` 改变跨层预测。
因此无预取组必须恰好传输 9216 byte 的真实 Expert 权重，不能沿用 bench 默认当前层预取预算来冒充无预取。

`total_cycles = 最后一层结果可见 cycle - 第一层 begin handshake cycle`。

不包括 DDR 校准、权重预加载、读回自检和最终结果后的 context cleanup。必须用同样的起止定义比较快速后端，不能把初始化开销加到单 token decode 时间中。

结果目录 `results/m10_ddr4_controller/` 包括：

- `ip_readback.txt`：所有必须匹配的 controller / CDC 参数；
- `summary.json`：三组周期、命中收益、错误预测开销、source hashes 与限制；
- `cases/<name>/ddr_service_samples.csv`：bank、地址、burst beats、AR、首/末 R、首 beat 延迟和总服务时间；
- `cases/<name>/xsim/ddr_bank_<b>.csv`：原始逐 bank 工作负载样本；
- compile / elaborate / simulate 日志、导出的 Vivado 仿真脚本与 waveform。

`DDR_DRAM_SUMMARY` 记录 workload 阶段 RD、ACT、REF、PRE 命令数量。未观察到的行为应明确标为未覆盖，而不是声称全部 DDR 行为已验证。

这些样本可以用于下一步校准主模拟器的内存服务模型；本轮**没有**用一个平均值覆盖既有 hardware profile，也没有修改历史 DSE 结论。首 beat latency 会受到 row/bank 状态、refresh 和请求相位影响，不能只用接口峰值 byte/cycle 替代。

## 本轮实际验证结果

Vivado 2023.2 的三组四通道 DDR4 仿真均完成数值自检，三层结果均为 24、60、254；重新分析时所有记录的源码 SHA-256 与当前文件一致。下表比较同一三层激励、当前层 BUDGET=0、相同计算/OCM 资源下的两种 RTL 验证后端，不是主模拟器已经完成 DDR 校准的结果。

| 激励 | 原快速内存后端 cycles | DDR4 controller/RDIMM 后端 cycles | 工作负载传输 byte | AR 到首 R 延迟范围（core cycles） |
| --- | ---: | ---: | ---: | ---: |
| 无预取 | 1398 | 2068 | 9216 | 31–115 |
| 跨层预测命中 | 1322 | 1781 | 9216 | 31–114 |
| 跨层预测错误 | 1398 | 2068 | 10496 | 31–115 |

快速后端数据来自本轮 `results/m10_ddr4_fast_regression/` 的 `p2_none`、`p2_hit`、`wrong_three`，并按相同 begin handshake / 最终结果边界取周期。

DDR4 后端中，命中相对无预取减少 287 cycles（13.88%，1.1611× speedup），而快速后端减少 76 cycles（5.44%）。这是此小型夹具中更长的真实控制器/CDC 服务时间提高了预取隐藏等待的机会，不是可外推至所有模型的性能结论。错误预测多传输 1280 byte，但最终总周期恰好不变；层内时序仍不同，不能理解为错误预取没有成本或永远免费。

所有 bank 均观察到真实 DDR 命令。每 bank 的 RD/ACT/REF/PRE 计数分别为：无预取 36/18/2/13，命中 36/15/1/12，错误 41/18/2/13。因此本次覆盖了读、激活、预充电和刷新命令出现，但没有遍历其所有时序组合。

每路校准在 core cycle 4332 完成，预加载与读回检查在 7166 完成，首层在 7181 开始；这些初始化时间不计入上表。首 beat 平均延迟分别为 38.56、36.06、37.71 cycles，明显不是单一固定延迟。差异同时包含控制器服务、异步桥、请求相位和内存状态；本轮尚未对这些贡献逐项归因。

验证汇总：

- 342 个 Python 测试通过；
- 35 项原快速后端 RTL 回归检查通过；
- 3 组 DDR4 数值/协议/背压检查通过；
- 三组均无日志 ERROR/Fatal 或 memory model timing violation；
- 每组仍有 13 条 vendor elaboration warning，其中 8 条为不执行的嵌套 concurrent assertion，不能声称完整 vendor assertion coverage。

原快速后端与主模型的回归仍为 30/30 执行规则和 FIFO occupancy 对齐，29/30 总周期完全相等；既有 `ddr_gaps` 用例仍相差 −8 cycles，descriptor peak 也仍有未一致项。本轮没有把这些既有差异包装成完全对齐。

## 复现命令

```bash
make -C hardware/m10_u200 ddr4-validation ALIGNMENT_ARGS="--jobs 3"
pytest -o addopts= -q
```

需要 Vivado 2023.2 位于 PATH；首次运行从 IP Tcl 重建独立工程和 vendor example models。默认 jobs=2，最多 3，避免无界并发。每个 XSim 用例 HDL watchdog 为 1 ms，启动/执行另有 wall-time 上限。脚本不删除已有目录；存在 example project 时复用并检查配置，重新编译当前 verification sources。

仅重新分析已有运行：

```bash
python hardware/m10_u200/scripts/run_ddr4_validation.py --analyze-only
```

## 工具限制与警告

新版 `vivado-ip-configurator` helper 使用 Tcl 8.6 的 `try`，本机 Vivado 2023.2 为 Tcl 8.5.14；因此使用兼容 Tcl、Vivado 原生 `write_ip_tcl` 配置重建和严格 readback，未修改 skill 或 vendor 库。

XSim 对 vendor calibration RTL 内嵌在 procedural block 的 concurrent assertions 发出 `XSIM 43-4398`，并不执行这些断言；已保留并统计。官方说明支持的是 procedural block 外的 concurrent assertion 形式。[AMD UG937](https://docs.amd.com/r/en-US/ug937-vivado-design-suite-simulation-tutorial/Concurrent-Assertion)

验证过程中保留了两类探索记录：`failed_backpressure_sampling/` 中是 VALID 更新 delta-cycle 立即采样 DATA 导致的 checker 竞态；修正为 negedge 稳定采样后，仍保持三周期背压检查。`initial_default_budget/` 中是 bench 默认当前层预算未关闭的探索性结果，不属于最终无预取对照实验。最终结果以三组显式 BUDGET=0 的 `summary.json` 为准。

其他 vendor elaboration 警告涉及 generic 向 glbl 传播、DEBUG_MESSAGES 标量索引、rank 与 dq_temp 索引。已检查 rank 警告来自当前 RANKS<=4 时未选中的 ternary 分支；dq_temp 警告来自加密 vendor SDRAM model，不能据此宣称所有内部警告均已证明无害。没有通过降级 severity 或忽略用户 checker 获取 PASS。独立数据、RLAST、RRESP、背压和结果检查不能替代所有 vendor 内部断言；本轮不声称完整 assertion coverage。

## 尚不代表的验证

- 板级引脚 / XDC / shell、XRT 或 PCIe 集成；
- 物理 PHY/PCB training 裕量与板上实测；
- ECC 故障注入、所有 DDR command/timing corner 或多 outstanding 压力测试；
- 完整大模型配置及所有调度策略性能排名；
- 主模型与 DDR4 backend 的完整性能对齐。
