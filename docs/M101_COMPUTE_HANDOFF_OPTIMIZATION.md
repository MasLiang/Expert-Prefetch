# 第一轮 RTL 控制交接优化

## 本轮范围

只修改 `tile_compute_controller.sv` 的就绪阶段交接，没有改主 Simulator、MAC 算法、
数值位宽、memory 配置、scheduler 注册搜索、pair admission 或 OCM 回收。
controller 同时整理为每个 always_ff 只更新一个寄存器，模块接口不变。

## 消除的等待

| 边界 | 优化前 | 优化后 |
|---|---|---|
| 请求接受且 GU 已就绪 | capture → WAIT_GU → registered GU_START | capture 与 registered GU_START 在同一边沿产生 |
| 非线性完成且 DOWN 已就绪 | NONLINEAR → WAIT_DOWN → registered DOWN_START | NONLINEAR_DONE 边沿直接产生 registered DOWN_START |
| GU/DOWN 未就绪 | 等待 committed/resident payload | 保留原等待条件 |

启动仍是注册脉冲，不是组合 bypass。backend 在下一边沿看到脉冲时，descriptor/generation
已经完成捕获，因此不需要让 OCM ready 组合路径直接驱动 backend。外部完成模式只响应
backend stage-done，不能用 timer 提前推进。clear 优先于启动、完成与释放。
GU/DOWN 两个 slot 仍到整个 tile 退休才释放，不会因缩短控制空档提前归还 ownership。

## 实测结果

Vivado XSim 2023.2、完整默认 U200 RTL 源列表，H=16、F=256、E=64、Top-2、16 tiles/Expert，
32 tiles/layer，与冻结 GU-only 版本相同配置和相同八点激励。

| 用例 | 优化前 cycles | 优化后 cycles | 减少 |
|---|---:|---:|---:|
| demand_fast / wrong_fast | 13909 | 13845 | 64 |
| demand_gaps / wrong_gaps | 13916 | 13852 | 64 |
| hit_fast / second_hit_fast | 13810 | 13747 | 63 |
| hit_gaps / second_hit_gaps | 13810 | 13747 | 63 |

收益约为这些测试总周期的 0.46%，不是 MAC 吞吐翻倍。正确预测命中首个 tile 的 OCM
提交等待仍被保留，因此该 tile 没有省掉一个 GU 交接周期，其余 ready-stage 优化生效。

demand_fast 的 32 tiles 累计 accept→GU 从 64 减到 32 cycles；hit_fast 从 66 减到 35。
每个 tile 的 GU 和 DOWN 真实执行时长分别仍为 256 和 128 cycles。
另一个收益来自 nonlinear→DOWN 的交接缩短。memory_wait、ready→accept、退休及 combine
规则不变，不把它们当作本次已优化项。

## 验证证据

- 239 项 Python 测试通过，包括计算服务时长守恒检查和 controller 单寄存器 always 检查。
- 54 个 RTL 回归运行点通过，新增 `tb_compute_handoff` 验证四种 GU/DOWN readiness 组合、
  已就绪立即启动、未就绪禁止推进、注册脉冲不重复、descriptor/generation 捕获、
  外部完成不受 timer 干扰，以及 clear 与请求/非线性完成/退休同周期碰撞。
- 8 个完整配置 XSim 自检正常结束且每点周期降低。检查结果向量、路由、字节分类、allocation、
  32 tiles 恰好执行一次、GU/DOWN 运算时长不变、OCM/FIFO/descriptor 容量及 MAC/DMA 互斥。
- lint 成功，仍有既有非结构性警告；没有所检查的 latch、多驱动、组合环或隐式位宽截断警告。
- 实际 backend 数值与 allocator golden 继续检查，不因控制优化跳过原回归。

行为仿真使用显式 non-project source list，不把既有 MCP 会话的非 U200 项目作为签核依据。
本轮没有新综合、STA、综合网表仿真或板级测试，不能据此宣称 200 MHz 时序闭合。
旧综合网表由旧源码生成，不能作为此次修改的验证证据。

## 复现和文件

```bash
PYTHONPATH=src pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 handoff-test
```

八点产物：`results/m10_compute_handoff/`，含 comparison JSON/CSV、REPORT、输入与 RTL 哈希清单、
编译/展开/运行日志、逐 tile/stage timing、事件与 timeline。
基线：`results/m10_gu_only_start/`，不覆盖。验证脚本要求输出目录不得等于基线目录。
通用 `gu-early-test` 的默认输出改为 `results/m10_rtl_current/`，避免日后重复运行覆盖此冻结基线。
本轮结果 `results/m10_compute_handoff/` 现作为下一轮冻结基线，`handoff-test` 重复运行改为输出
`results/m10_compute_handoff_current/`，保留本轮证据。

修改文件：controller、新增 TB、Makefile、RTL 比较验证脚本、Python 检查与中文文档。
没有修改主 Simulator/M10 adapter 的性能或调度语义。

## 下一阶段

注册 scheduler 搜索和请求准备还会带来 ready→accept 等待；可评估 lookahead 并在退休后快速接力，
必须保留 stale-search、slot generation 和背压稳定性保护。
GU slot 独立早释放、DMA lookahead、多 context 和 MAC 流水化尚未实现，不能将本轮收益外推至它们。
注册搜索 lookahead 已在下一轮完成，实际结果另见 `M101_SCHEDULER_LOOKAHEAD_OPTIMIZATION.md`。
