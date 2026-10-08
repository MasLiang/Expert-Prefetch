# 宽候选预取的 RTL 验证范围与结果

## 当前结论

本轮实际运行 Vivado XSim 2023.2 behavioral simulation，**21 个有界自检用例全部 PASS**，Python 回归 **472 passed**。

但是上一轮宽候选 Uniform/Ranked 的全序列周期与 memory-wait 消除比例 **尚未在 RTL 上验证**。当前 `candidate_prefetch_scheduler` 固定实例化 `hardware_greedy_v1`，当前和跨层接口都没有独立 Uniform/Ranked 策略选择开关；package 中的 policy enum 不代表顶层已接入这些算法。不能把 HardwareGreedy 的 PASS 改名成 Uniform/Ranked PASS。

此次只增加 testbench 与验证驱动，没有修改任何设计 RTL、XDC 或已有 Vivado 工程。新建了独立 in-memory U200 inspection project；原项目保持不变。

## 第一组：宽候选 allocator

测试模块：`tb_wide_candidate_allocation`。它直接连接现有 `candidate_prefetch_scheduler`，不绕过内部 HardwareGreedy 选择逻辑，也不使用 force/deposit。

输入的 byte 几何对应前一轮真实 Switch 尺寸：

- 8/16 个候选、24 个 logical tiles/Expert。
- 每个 Up/Down physical fragment 98,304 byte；logical tile 196,608 byte。
- global/channel credit 均为 9,437,184 byte，可分配 48 个 logical tiles。
- 一个聚合 allocator channel，与 U200 四 bank striping 的 credit 抽象一致。
- confidence 采用 0.758、0.086、均分尾部 0.156，并显式量化成 Q12。
- 两种 backpressure：无停顿，以及每七个 cycle 停两拍。

| 候选数 | backpressure | 分配 logical tiles | 覆盖 Expert | 分配 byte |
|---:|---|---:|---:|---:|
| 8 | 无 | 48 | 8 | 9,437,184 |
| 8 | 有 | 48 | 8 | 9,437,184 |
| 16 | 无 | 48 | 16 | 9,437,184 |
| 16 | 有 | 48 | 16 | 9,437,184 |

RTL 每一次 handshake 的 Expert、tile、顺序和 byte 数都与独立 Python `hardware_greedy_v1` 整数 golden 比较；另验证前缀连续性、地址、两矩阵 descriptor、credit 扣减及停顿期间的输出稳定性。每例都有 watchdog 和明确 PASS/$finish。

按候选排名得到的深度为：

- 8 候选：`[10,7,6,5,5,5,5,5]`。
- 16 候选：`[8,5,3,3,3,3,3,3,3,2,2,2,2,2,2,2]`。

16 候选与上一轮 float Ranked 的 `[9,5,3,3,3,3,3,3,2,2,2,2,2,2,2,2]` 不同。这也直接说明不能将 Q12 HardwareGreedy 的输出视为 float Ranked 的严格 RTL 等价实现。

**这组只有 allocator，不包含实际 DMA/OCM payload RAM 或 compute。9 MiB 是 credit，不是在此测试中实际写入 RAM 的驻留量；没有层延迟、带宽利用率或等待消除比例结论。**

## 第二组：真实数值与三层数据通路

复用已有 `tb_switch_relu_backend` 和 `tb_u200_moe`，验证真实 Router、int8 Up/ReLU/Down、OCM payload、Q15 combine、层间激活依赖、预测 promotion/cancel、descriptor key 唯一性及资源退休。

此处使用小型 H=16/F=32/E=6、每 Expert 两个 tiles、Top-1、三层、单候选夹具；并非上一轮 H=768/F=3072/E=8/16 的宽候选数值模型。

共运行一个 dense 数值用例（内部 12 个交易）及 16 个三层用例：四种 case × 两种 DDR responder 节拍 × hit/wrong 两种预测。AXI 数据源是自检 behavioral responder，不是 DDR4 PHY/板卡实测。

| responder | 预测 | A 整 Expert 无预取 | B streaming 无预取 | C 整 Expert 预取 | D 一 tile 预取 |
|---|---|---:|---:|---:|---:|
| fast | hit | 816 | 699 | 650 | 599 |
| fast | wrong | 816 | 699 | 826 | 699 |
| medium | hit | 898 | 718 | 672 | 610 |
| medium | wrong | 898 | 718 | 902 | 722 |

表中为实际 RTL 三层 makespan cycles。D 的原生 allocator 仍为 HardwareGreedy；单候选一 tile 不能用于比较 Uniform 与 Ranked。

命中时 D/B 为 fast **1.16694×**、medium **1.17705×**；错误预测时分别为 **1.00000×**、**0.99446×**。这些只能说明本小型 RTL 夹具的功能与性能，不能用其比例替代真实尺寸模拟器结果。

## XSim 启动诊断

最初新增 testbench 有声明顺序错误 `VRFC 10-3380`，已在验证文件中修正，没有修改 DUT。

随后沙箱内 XSim 编译/elaboration 成功，但 runtime 在进入 HDL 前报 `unexpected exception when evaluating tcl command`。环境 doctor 检查工具、kernel shared libraries、toolchain 和工作目录无硬失败。通过授权的沙箱外执行，同一源码、generic、plusarg、golden 和有界脚本成功运行，21 个用例均得到独立 PASS；验证了启动环境 workaround，未关闭任何功能断言或改变 RTL 语义。

已搜索 AMD 文档中的该签名及 Linux kernel/sandbox 启动情况。返回的 AR 63984（旧版本 DPI）、67272（旧版本链接）、65386/50705（旧版本 MIG）机制均不适用于本用例，没有应用厂商补丁，也没有据此宣称发现了 XSim 设计缺陷。启动失败原始日志保存在 `diagnostics/sandbox_launch/`。

## 产物与复现

```bash
PYTHONPATH=src python hardware/m10_u200/scripts/validate_wide_prefetch.py
pytest -o addopts='' -q --disable-warnings
```

结果目录：`results/m10_wide_prefetch_validation/`。

- `summary.json` 明确记录 `strict_uniform_ranked_validation: false`。
- `comparison.csv` 保存 16 个三层用例的 RTL/主模型 makespan 与 B 基线速度比。
- `golden_8.json`、`golden_16.json` 及 `.mem` 保存 bit-exact allocation 激励。
- 每例保留 compile/elaborate/runtime 日志、调用参数和资源统计。
- 验收检查了 21 份完整 runtime transcript 和 36 个源码/配置 SHA-256；Python 回归有 14 个既有依赖弃用警告，没有功能失败。

## 尚未验证及下一步

尚未证明：宽候选 Uniform/Ranked 的完整 DMA/OCM/compute 行为、上一轮 77%–83% 的 memory-wait 净消除比例、真实 Switch 尺寸的 RTL makespan，以及该配置的综合/PnR/板级带宽。

要完成严格对齐，需要先确认设计修改：加入明确的策略选择与 Rank/Uniform 数值契约，再扩展全通路宽候选夹具，保持与主模型相同的 allocation、DMA 顺序、OCM 容量/slot admission 和启动条件，并逐项比较等待指标。现有 allocator 单元测试不能跨越这一步。
