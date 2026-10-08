# 完整尺寸六层 Switch：主 simulator 与四通道 AXI RTL 对照

> 本文保留首次对照结果。D 的 FIFO 可见性错误现已修复；最新周期为
> 2,006,406（RTL 为 2,006,224），详见 `SWITCH_D_ALIGNMENT_DEBUG.md`。

## 实验边界

本次实际运行 `UnifiedSequenceSimulator`，不使用已失效的
`M10AlignmentSimulator`，不再次启动 RTL 或真实 DDR 器件模型。
RTL 对照来自 `results/m10_switch_axi_six_layers_20261007` 的四组已通过仿真。
没有修改主模拟器语义、Scheduler 算法、RTL 或 XDC。

两侧均使用 H=768、F=3072、E=8、Top-1、单 token、六个连续 MoE 层，
两矩阵 ReLU Expert、INT8、16-lane 共享 Router/Expert MAC、200 MHz。
这是完整 Switch Expert 尺寸的人工数值夹具，不是包含 attention/encoder 的完整
Transformer 推理，也不是训练权重或 CEPR 84.4% 命中率实验。

真实路由依次为 Expert 0–5。目标层 N 的预测在源层 N−1 开始后产生，
Top-1 全命中，候选顺序 `(N+rank)%8`，Q12 分数为 3104、352，其余均为 106。
第一层冷启动。C 提供两个候选，D 提供八个候选；两者预算均为 9 MiB。
两侧使用冻结的 HardwareGreedy-v1，而不是 Uniform/Ranked 策略。

## 硬件与执行配置

- 四个 128-bit AXI bank，256-byte aggregate burst，startup=0、beat-gap=0。
  行为 responder 每次接收 beat 后撤销 RVALID，主模型沿用两拍 payload 间隔，
  并沿用现有结构性 DMA 请求、仲裁、返回成本；不是按峰值带宽直接计算。
- OCM 为 4×32×128 KiB=16 MiB，每个实际 96 KiB 的 UP/DOWN fragment
  仍占一个 128 KiB slot。每 Expert 24 对 tile，总有效权重 4.5 MiB，实际占 6 MiB。
- descriptor=128、FIFO=64，GU/DOWN 成对 admission，tile ready-first，有限 credit，
  demand 优先、burst 非抢占、错误未发出请求可取消、未来请求可重试。
- A/C 使用 `wait_full_expert`，整 Expert UP/DOWN 全部 ready 后才能开始计算，
  并为 demand 保留 48 个 slot；B/D 保留两个 slot，GU tile ready 即可开始。
- 没有“上一层所有权重加载完成才放行下一层预取”的硬约束。
- ReLU 数据通路每拍处理一个元素。Router MAC startup 沿用八拍；Top-1 Router
  控制成本按 E×K+2K+3 声明，不从 RTL 总周期反推。
- 每次层间 Router reload 有 H×E/16=384 次 word 接收，另声明五拍 host/chain
  交接。首层开始前 reload 不计入 makespan，五次层间 reload 计入。
- 预测事务就绪成本为四拍交接加候选数量：C 为六拍、D 为十二拍。
  没有使用 RTL 观测总周期或残差拟合补偿。

完整声明式输入、预测时间规格和源代码 SHA256 均保存在结果目录。
主模拟器执行性能事件，不重新计算 RTL 的 tensor 数值；两侧共同的 truth/gating
来自同一夹具生成规则。

## 实测对照

周期误差定义为 `(主模型周期−RTL周期)/RTL周期`。

| Case | 主模型 cycles | RTL cycles | 周期误差 | 主模型相对 A 加速 | RTL 相对 A 加速 |
|---|---:|---:|---:|---:|---:|
| A：整 Expert 就绪、无预取 | 3,013,531 | 3,013,249 | +0.0094% | 1.0000× | 1.0000× |
| B：tile streaming、无预取 | 1,821,517 | 1,821,517 | 0% | 1.6544× | 1.6543× |
| C：整 Expert 就绪、完整预取 | 1,998,389 | 1,998,342 | +0.0024% | 1.5080× | 1.5079× |
| D：tile streaming、tile 预取 | 2,058,362 | 2,006,224 | +2.5988% | 1.4640× | 1.5020× |

两侧性能排序均为 **B、C、D、A**（由快到慢）。
D 相对 B 的主模型加速为 0.88494×，即慢 13.00%；RTL 为 0.90793×，即慢 10.14%。
因此这次主模型仍预测 streaming 下预取造成负收益，与 RTL 趋势一致。

## 读取量与剩余差异

| Case | demand bytes（两侧一致） | 主模型 speculative bytes | RTL speculative bytes |
|---|---:|---:|---:|
| A | 28,311,552 | 0 | 0 |
| B | 28,311,552 | 0 | 0 |
| C | 4,718,592 | 34,897,920 | 34,897,920 |
| D | 18,677,760 | 24,870,912 | 23,691,264 |

C useful/wasted 也完全一致：23,592,960 / 11,304,960 bytes。
D useful 完全一致：9,633,792 bytes；wasted 主模型为 15,237,120，RTL 为 14,057,472。
每组 demand+useful 均为六层必需的 28,311,552 bytes。

D 的主模型多读取 **1,179,648 bytes=1.125 MiB**，全部为无用 speculative 数据，
等价于十二个 96 KiB matrix fragment；峰值 OCM 主模型为 124 slots，RTL 为 112。
主模型计划的八个候选深度为 `[10,7,6,5,5,5,5,5]`，每个目标层相同。

这些观测证明 D 的差异不只是一个固定控制延迟：实际 speculative 发出/取消量和
占用轨迹也不同。待进一步逐请求诊断的是静态计划重试与 RTL 动态分配的
credit 可见时机、请求 admission/退休及 truth 到达后的取消窗口。
这些是根据架构和计数器提出的定位方向，**尚未证明某一个机制单独导致全部
52,138-cycle 残差**。现有 RTL timeline 没有逐 descriptor admission/cancel 日志，
不能凭此次结果宣称已定位完整根因。

A/B/C 在这一个夹具下高度接近；B 的总周期相等不等于所有内部事件逐拍一致，
也不代表其他配置已经通过精确对齐验收。D 可以支持本配置的趋势互证，不能
宣称 cycle/byte/order 完全对齐。RTL `wait_gu/wait_down` 是特定信号计数，
不能直接当作主模型所有 memory starvation 的同口径指标。

## 复现与验证

在项目根目录运行：

```bash
PYTHONPATH=src python examples/switch_full_axi_main_comparison.py
pytest -q
```

默认输出 `results/m10_switch_axi_main_comparison/`：

- `comparison.csv`：四组周期、byte、加速比及误差。
- `summary.json`：明确标记 `COMPLETED_NOT_CYCLE_EXACT_CERTIFIED`，包含源代码哈希。
- 每组的 `shared_input.json`、`routing.jsonl`、`predictor.jsonl`、`main_result.json`：
  完整配置、输入、prediction specs、事件 trace 和结果。

本轮完整回归 **520 tests 通过**，包含四个新增完整尺寸夹具测试。
不包含真实 DDR 器件行为，不是板级测量，不应作为 U200 实测性能结论。
