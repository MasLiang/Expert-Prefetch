# M10.1 第三轮优化：GU 阶段独立回收

## 当前实现

GU_DONE 表示真实 INT8 backend 已完成 GU 的最后一次 payload 读取，gate/up 中间值已进入
计算寄存器。后续非线性及 DOWN 不再读取 GU 权重，因此无需把 GU slot 保留到 tile 结束。
控制器输出注册的 `consumer_gu_release_valid`；内存顶层将其送到 OCM metadata 与
payload RAM，predictive 顶层同时送到 execution scheduler。软 flush 屏蔽此释放脉冲，
仍按既有 flush 优先级清理所有权。DOWN 的释放继续由 tile 的 DOWN 完成驱动。

这不是提前开始下一 Expert 计算：单 MAC 和整 tile busy 约束不变。
新空间只允许后续权重分配/DMA更早推进，下一计算仍等待当前 tile 合法退休。

## 所有权与生命周期

- GU 与 DOWN descriptor/generation 在当前 tile 期间保持稳定。
- OCM、payload binding 与 scheduler GU 有效位独立退休。
- scheduler 清除有效位必须同时匹配 slot、epoch 和 generation。
  最终 tile 退休不能通过旧 slot 编号清除新 generation 的 descriptor。
- capture 与退休碰撞时新 capture 优先；物理 OCM allocator 仍只从 FREE slot 分配。
- 迟到旧 generation 释放被 OCM 拒绝，不清除新 owner；payload binding 同样检查标签。
- DescriptorTable 不随 GU_DONE 直接回收：FIFO token、DMA ownership 与 completion 引用
  必须继续按原 retirement rule 全部退休。计算 slot 生命周期和请求表引用生命周期不同。
- GU/DOWN 原子 pair admission、bank 几何与 FIFO/descriptor credit 限制不变。

## 实际验证

使用 `vivado-simulate-rtl` 流程指导，运行实际 RTL 自检，不改主 Simulator/M10 性能语义。

1. 57 个 Icarus RTL 回归运行点全部通过。新增四 slot 压力测试使用真实 OCM、payload RAM、
   controller 和 scheduler，外部模拟计算完成脉冲（不将此单测当作数值 MAC 证明）。
   当前 pair 加一个无关 owner 占用三 slot，并由 bank 压力阻塞下一 pair；GU_DONE 后，
   在当前 tile 仍执行时分配下一 pair，将下一 DOWN 放入旧 GU slot 的新 generation。
   注入迟到旧 GU release、再完成旧 DOWN，两者均不能破坏新 owner；实际 RAM 读回验证 payload。
2. 八个完整默认配置 XSim 2023.2 用例全部正常结束并 TEST_PASS，使用真实 INT8 MAC、
   非线性、Router 与 DDR 模型。每点包含 32 个真实 tile，各产生一次 GU 和一次 DOWN 释放。
   自动检查 GU release 不早于 GU_DONE 且早于 DOWN_DONE，DOWN release 不早于 DOWN_DONE；
   缺失、重复或额外 tile 的 release 均使验证失败。
3. 新旧 RTL 的数值向量、truth、allocation、所有传输字节分类和 GU/DOWN 执行时长保持一致；
   同时验证 MAC/DMA 互斥、字节守恒与 OCM/FIFO/descriptor 容量。
4. 241 个 Python 测试通过，包含 release 检查器对缺失、重复、过早和延迟 GU 释放的反例。
   lint 与 allocator/dense SwiGLU golden 通过。lint 仍有未使用信号/参数、空输出连接、
   文件名、顺序临时量和混合 reset 使用警告，不宣称零 warning。

## 周期结果与限制

比较冻结的第二轮 `results/m10_scheduler_lookahead/`，八点总 cycles 均不变：

| 用例 | 第二轮 cycles | 本轮 cycles |
|---|---:|---:|
| demand_fast / wrong_fast | 13815 | 13815 |
| demand_gaps / wrong_gaps | 13822 | 13822 |
| hit_fast / second_hit_fast | 13717 | 13717 |
| hit_gaps / second_hit_gaps | 13717 | 13717 |

默认有 128 slot，OCM 容量未阻塞关键路径。小容量测试证明的是提前解除 admission 压力与
安全复用，不是证明所有模型必然提速，也不是板级 DDR 带宽结论。

本轮没有重新综合、综合网表仿真或 STA；旧 DCP 不能代表当前 RTL，不宣称 200 MHz 闭合。
与主 Simulator 仍有 atomic pair admission、slot/bank 计费、候选 tie-break、DMA 请求顺序、
有效 MAC 服务率和多 layer context 等差异。DMA lookahead 与 chunk 流水尚未实现。

## 复现与产物

```bash
PYTHONPATH=src pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 stage-release-test
```

`results/m10_gu_stage_release/` 保存源文件 SHA256、激励配置、编译/展开日志、逐例自检日志、
events/timeline/tile/stage CSV 以及 comparison JSON/CSV。`events.csv` 中新增
`GU_SLOT_RELEASE` / `DOWN_SLOT_RELEASE`，value 为持有的 slot generation；实验为单 epoch。
历史三轮基线目录保持冻结，`scheduler-test` 重跑使用单独的 current 目录。
