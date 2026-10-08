# M10.1 GU 提前启动

本文件记录首次 GU-only 优化与当时的冻结结果。当前 RTL 后续已改为 GU_DONE 独立回收，
不再保持两个 slot 到 tile 结束；最新规则与验证见 [GU 独立回收](M101_GU_STAGE_RELEASE.md)。

本轮修改 RTL，而不是让主 Simulator 迁就旧 RTL 的 GU/DOWN 同时就绪规则。
主 Simulator 和 M10 性能适配层的调度算法没有改动；Python 改动仅涉及事件分析与验证入口。

## 执行规则

所有自动执行入口统一使用 GU-only 启动。旧 `GU_EARLY_START` 参数、传递接口及
pair-ready 启动分支均已删除，通用 scheduler、predictive top 与 U200 top 不再存在两种模式。

启动需要真实 Router truth、当前 Expert 的下一个 prefix tile、已驻留的 GU 权重，
以及同一 Expert/tile 的有效 GU/DOWN descriptor 和已建立的 slot ownership。
不要求 DOWN 的 DMA 已完成。注册搜索结果仍需重验有效性，descriptor 捕获冲突不会发射。

执行顺序仍为 GU → 非线性 → 等待 DOWN 权重 → DOWN → accumulator/combine。
现有 compute controller 已有 WAIT_DOWN，因此没有新增计算引擎或绕过依赖。
GU 与 DOWN 分别在实际启动时更新 OCM compute 状态；两个 slot 直到 tile 退休才释放。
Router/Expert 继续共享单一 MAC engine，DMA 仍服从既有优先级、带宽与不可抢占限制。

pair-ready 仅保留在冻结历史数据与离线分析器中，用于解释旧结果，不是当前 RTL 的可选行为。

## 实际验证

使用 Vivado XSim 2023.2，完整默认 H=16、F=256、64 Experts、Top-2、16 tiles/Expert，
四路物理 DDR stripe，128 个 OCM slots，完全相同的激励与冻结旧 RTL 基线比较。

| 用例 | 旧周期 | 新周期 | 减少周期 |
|---|---:|---:|---:|
| demand_fast / wrong_fast | 13923 | 13909 | 14 |
| demand_gaps / wrong_gaps | 13934 | 13916 | 18 |
| hit_fast / second_hit_fast | 13810 | 13810 | 0 |
| hit_gaps / second_hit_gaps | 13810 | 13810 | 0 |

八点均正常结束并输出 TEST_PASS。检查输出数值、真实路由、传输字节分类、32 个 tile
恰好执行一次、MAC/DMA 互斥、OCM/FIFO/descriptor 容量，以及 DOWN_START 不早于 DOWN PAYLOAD_READY。
定向测试额外覆盖缺失或错误 DOWN descriptor、不就绪 DOWN 长期等待、等待中 clear，以及完成后释放。

Python 回归：237 passed。RTL 回归：53 个 PASS 运行点。Verilator lint 返回成功，仍有既有警告，
不宣称零警告。此次没有重新综合、网表仿真或 STA，因此旧综合网表和旧时序结果不代表当前 RTL。

## 指标与后续对齐边界

事件分析器增加显式 `gu_only_ready` 选项：启动等待截止于 GU ready，
仍保留 pair-ready 时间戳，并单独验证 DOWN 的 payload 依赖。
输出的启动 `memory_wait_cycles` 不包含执行过程中 WAIT_DOWN 的等待，不能把它当作整个 layer 的总内存等待。
长期 DOWN 背压已经通过定向 RTL 测试验证，不因减少启动等待而消失。

旧 M10 性能适配层仍采用 pair-ready，当前默认 RTL 下旧对齐入口会拒绝运行。
本轮只证明新旧 RTL 的行为守恒及实测周期变化，不宣称新 RTL 与主 Simulator 已完成性能对齐。
收益取决于 GU/DOWN 加载窗口与控制延迟；正确命中用例权重本来已就绪，所以没有收益。

## 复现

```bash
PYTHONPATH=src pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test lint
make -C hardware/m10_u200 gu-early-test
```

移除旧接口后的八点产物位于 `results/m10_gu_only_start/`，包括比较 JSON/CSV、输入与 RTL SHA 清单、
逐用例 XSim 日志、事件、tile/stage timing 与 timeline。基线目录
`results/m10_boundary_alignment/` 不会被覆盖。
