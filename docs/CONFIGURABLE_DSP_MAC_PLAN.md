# 可配置DSP MAC阵列：待实施方案

本文件只记录2026-10-06讨论的方案。**本轮不修改RTL，不声称已实现DSP阵列。**

## 决策

后续把int8乘法明确映射到DSP，并以综合报告确认实际映射。
先考虑16/32/64 lanes，再根据供数与时序验证决定是否扩到128/256 lanes。
`MAC_GROUPS`表示同一个GEMM Engine内部16-lane分组数量，不代表Expert并行数。
Router与Expert继续共享计算资源；保留GU-ready提前启动。

综合时参数化改变物理资源；运行时group enable仅改变活跃规模/功耗，不释放DSP。
不能只给乘法添加属性就视为映射验收通过。需要综合资源、网表结构、数值仿真和时序检查。

## 必须配套的修改

- OCM分bank/宽读口及lane数据选择，匹配每拍16/32/64 bytes的权重消费需求。
- 激活本地缓存和广播/分发；归约树与int32累加流水参数化。
- 对Router、GU、DOWN验证启动、行切换、排空、溢出及现有量化ABI。
- 记录payload有效MAC周期、等待权重、控制/非线性等待，不能只看prefetch百分比。
- DDR→OCM与OCM→MAC分开验证；DSP有余量不代表供数能达到目标。

## 模拟器侧的PE含义

本轮MAC规模研究保留一个逻辑Engine，在`ops_per_cycle_per_aie`中配置
`32 * mac_groups`，`total_aie_units=1`和`aie_units_per_expert=1`。
这是扩大单Engine宽度的研究假设，不是绕过资源检查或同时执行多个Expert。
它避免把tile_ready_first目前的单共享Engine限制误当成不能增加lane数量。
有效速率为待测参数，现有128-bit OCM读取不能自动支撑更宽MAC。
PE=1/2/4/8/16对应16/32/64/128/256个假想lanes；高档位不是当前RTL已有能力。

后续实施前，先通过`RTL_RULES_PE_SWEEP.md`的供数敏感性分析选择规模。
