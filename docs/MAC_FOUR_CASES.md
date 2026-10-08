# 增加MAC后，四种执行方式的同配置收益

## 问题与分母

本轮回答：**相同MAC规模、模型、OCM、带宽和输入下，四种执行方式各自有多少收益？**
不是将更大MAC与较小MAC混为预取收益，也不拼接上一轮finite-slot规则数据。

| Case | 权重就绪/执行方式 | 预测 |
|---|---|---|
| A | 非streaming：当前Expert全部GU/DOWN权重resident后才开始计算；当前层不做权重DMA/Expert计算重叠 | 无 |
| B | streaming：当前Expert按tile就绪执行并加载后续tile | 无 |
| C | 与A相同的整Expert启动条件 | 提前一层请求完整candidate Expert权重 |
| D | 与B相同的tile启动条件 | 提前一层Uniform GU prefix depth=2 |

主分母为**同一MAC档位的A**，速度比=A cycles / case cycles。
同时保存D相对B的纯预取增益、D相对C的策略比较和各case相对自身16 lanes的计算规模增益。
报表中的速度比为三个seed配对总时间比的均值，不是跨档位或跨配置挑选的最优值。

## 公平配置与边界

四case共同采用`UnifiedSequenceSimulator`，未修改核心或RTL。
复用`results/rtl_rules_pe_sweep/*_inputs.json`完整激励，每组4个MoE层、单token。
合成模型H=128/F=512/E=16/K=2，tile=64，OCM=0.75 MiB；
OLMoE真实结构H=2048/F=1024/E=64/K=8，tile=128，OCM=24 MiB。
二者整Expert分别192 KiB和6 MiB，因此A/C在容量上可行。
3个seed；合成每seed5个序列，实际recall90%；OLMoE每seed1个序列，实际recall91.67%。
路由和预测ranking均为合成，不是实测模型trace。

公共硬件：200 MHz、4 channels、每channel最多一个in-flight、4096-byte burst。
全部case使用相同DMA选择/startup/请求/返回及payload倍率2，相同GEMM startup、
非线性吞吐和控制成本。MAC速率32×groups ops/cycle，一个共享Engine，非Expert并行。
16/32/64/128/256 lanes都是宽度假设，只有现有16-lane实现不是扩宽假设。
高档带宽不是U200真实板级带宽。

两种预测case均从前一层Router done+20 cycles得到候选，
保留上一层权重完成条件，因此预测就绪不等于当时已能发DMA。
第一层无预测。非streaming不是禁止跨层speculative DMA与计算重叠，
而是不让当前Expert边取自己的权重边算。

主模型的非streaming路径目前不支持fixed slots、atomic pairs、finite descriptor/FIFO和
tile-ready-first。因此本轮**全部case统一byte OCM、无限请求credits、expert_locked**，
并使用共同的burst/供数时序。不将A/C用byte容量、B/D用slot容量的数据拼接。
这是四策略性能对照，不是当前RTL完整规则联合仿真；上一轮规则报告仍独立保留。

C的“完整Expert预取”指请求完整candidate权重，且**计算必须等当前Expert完整resident**。
不保证所有candidate都能prefetch完整：预算、已有占用和错误候选会截断/拒绝请求，
缺失部分随后demand补齐。C不做whole-Expert原子admission，也不是FLAME精确复现。
D只预取GU prefix，不是整Expert预取后仍streaming，也不是当前RTL HardwareGreedy。

## OLMoE结果

标称16 bytes/cycle/channel；所有速度比相对同档位A：

| groups / lanes | A 无stream/无预取 | B stream/无预取 | C 无stream/完整预取 | D stream/tile预取 | D相对B | D相对C |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 16 | 1.0000× | 1.4337× | 1.0697× | 1.4748× | 1.0287× | 1.3807× |
| 4 / 64 | 1.0000× | 1.4587× | 1.0719× | 1.4773× | 1.0127× | 1.3788× |
| 16 / 256 | 1.0000× | 1.1174× | 1.0367× | 1.1210× | 1.0033× | 1.0815× |

对应平均每序列cycle：

| lanes | A | B | C | D |
|---|---:|---:|---:|---:|
| 16 | 19,070,969 | 13,302,329 | 17,857,136.7 | 12,931,884 |
| 64 | 9,609,209 | 6,587,441 | 8,968,781 | 6,504,754.7 |
| 256 | 7,243,769 | 6,482,993 | 6,988,257.3 | 6,461,796.7 |

结论：16→64 lanes明显降低所有case总时间；64→256 lanes，B/D已接近供数瓶颈。
64 lanes的D相对A有1.4773×，但**不能把47.73%速度比提升全部算给预取**：
B已有1.4587×，D相对B仅1.27%速度比提升；相对B延迟降低约1.25%。
256 lanes同样仅约0.33%额外预取收益。

## 合成模型结果

标称16 bytes/cycle/channel，速度比相对同档位A：

| lanes | A | B | C | D | D相对B |
|---|---:|---:|---:|---:|---:|
| 16 | 1.0000× | 1.4783× | 1.1526× | 1.4891× | 1.0074× |
| 32 | 1.0000× | 1.7907× | 1.1831× | 1.7985× | 1.0044× |
| 64 | 1.0000× | 1.4699× | 1.0993× | 1.4774× | 1.0051× |
| 128 | 1.0000× | 1.2520× | 1.0644× | 1.2522× | 1.0002× |
| 256 | 1.0000× | 1.1412× | 1.0574× | 1.1391× | 0.9982× |

供数更高的标称64 bytes/cycle/channel时，128 lanes的B/C/D为
1.7496×/1.2030×/1.7570×；256 lanes的D相对B为1.0193×。
供数较低的4 bytes/cycle/channel时，256 lanes的D相对B为0.9959×。
完整扫描见CSV，不能只挑一个更高prefetch百分比宣布MAC扩宽最优。

## 解释

非streaming A近似累加加载与计算，streaming B可让两者部分重叠。
当两项接近时，重叠空间较大；扩大MAC令计算变短后，加载占主导，B相对A的收益也缩小。
C只能通过跨层提前加载缩短整Expert就绪等待；D同时具有streaming与提前加载两种机会。
但D对B的净收益还受预测窗口、有限OCM、候选排序、错误传输和burst阻塞约束，
不是命中任一tile就一定有大幅总延迟收益。

这些是固定Uniform策略的小矩阵测量，不是普遍定理或最优硬件设计结论。
总cycle收益和预取额外收益应分别报告。

## 复现与验证

```bash
# 若尚无上一轮冻结激励，先生成它们
python examples/rtl_rules_pe_sweep.py --workers 4
python examples/mac_four_cases.py --workers 4
pytest -q -o addopts=''
```

最终：72配置点、216 seed点、936条模拟结果，全部完成。
全部回归**430 passed**；新增9项四case/共享规则/确定性/分母测试。
产物检查：四case配对完整，payload计算周期相同，无计算工作量省略；
OCM容量、整Expert容量可行性、speculative byte守恒、非预测零speculation、来源哈希均通过。

`results/mac_four_cases/`包含`summary.json`、输入trace JSON、`per_sample.csv`、
`per_seed.csv`、`comparison.csv`及`four_cases.png/pdf`。
YAML、主模型、复用输入及驱动来源哈希保存在summary。
未实施的RTL有限credits非streaming路径和宽DSP阵列不会在本报告中冒充完成。
