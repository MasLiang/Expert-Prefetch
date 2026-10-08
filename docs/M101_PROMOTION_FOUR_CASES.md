# M10.1：命中 DMA 提升修正与四 case RTL 对照

## 1. 修正的含义

已开始的 speculative descriptor 一旦被 Router truth 命中，其**服务优先级**立即提升。
不得在 burst 边界重新退回 speculative FIFO，排到未来层错误预取之后。
原始 request 类型仍为 speculative，用于 useful/wasted bytes 记账，不改写成 demand。

RTL 使用带 generation 的 promotion 通知，包括 burst 完成同拍的命中；错误 generation
不能提升。已经让出 DMA ownership 的命中请求需要 promoted FIFO token 和对应引用计数。
仍拥有 DMA 的请求不重复生成 token。增加完成退休保护拍，避免旧 speculative token
在 DescriptorTable 尚未看到完成时再次启动并重复传输最后一个 burst。
主模型 `MultiChannelMemory` 同步修复 active 和已让出 burst 的两种命中路径。

## 2. 四种实际执行模式

| Case | 预取 | 第一条 GU 启动条件 |
|---|---|---|
| A_whole_demand | 无 | 当前 Expert 全部 GU/DOWN 权重到齐 |
| B_stream_demand | 无 | 第一 GU tile 就绪 |
| C_whole_prefetch | 完整预测 Expert | 当前 Expert 全部 GU/DOWN 权重到齐 |
| D_stream_tile_prefetch | 每个候选的第一 tile pair | 第一 GU tile 就绪 |

RTL 增加可综合的 `WAIT_FULL_EXPERT` 开关，默认 0，保持原 streaming 行为。
只在第一 GU tile 启动前检查完整驻留，之后允许已计算 tile 释放 OCM。
整 Expert 模式为 demand 预留完整 Expert 所需 slot，防止未来 speculative 占用容量而
阻塞当前 Expert 的启动；容量不足会明确报错。主模型声明同样的启动和容量规则。
四种模式共享单一 Router/Expert MAC，不把 Router 运算移到额外 PE。

## 3. 配置与验收边界

本实验为**小型真实数值 SwiGLU RTL 夹具**：H=16、F=32、E=6、Top-1、int8、
F_TILE=16，3 层级联，16 MAC lanes（32 operations/cycle）。每 Expert 两个 tile pair，
GU 512 bytes、DOWN 256 bytes，每 Expert payload 1536 bytes。
OCM 两 bank、每 bank 八个 512-byte slot，共 8192 bytes；32 descriptor、FIFO depth 8。
完整预取预算 1536 bytes，tile 预取预算 768 bytes；未来层预测提前在源 layer 0 发出。
没有“上一层全部权重完成”的预取屏障。

四个 128-bit AXI 返回口使用有限速率行为响应器。fast/medium/slow 的返回间隔分别为
最短两拍、四拍、八拍；主模型用相应的有效 payload 服务率和独立控制参数。
标称速率分别为 16/8/4 bytes/cycle/channel，模型 payload 倍率 2。
这不是 DDR4 PHY、板上测量、200 MHz 时序验证，也不是 Switch ReLU/FLAME 的 RTL 复现。

每个带宽点运行全部命中与全部错误两种预测，共 24 个性能 case。另运行四个
promotion 边界 case：setup、中途 burst、完成同拍、错误 generation。
RTL 验收包括数值输出、真实 Router truth、每个 GU/DOWN stage 恰好一次、完整权重门槛、
payload 记账及正常 `$finish`。任何功能失败不会输出性能 PASS。

## 4. 结果文件与解读

最终数值见 `results/m10_four_cases_promotion_fixed/comparison.csv` 和 `summary.json`。
每个 case 保存主模型输入/结果、RTL generics、编译/展开/运行日志、逐层起止周期。
`manifest.json` 保存 RTL、testbench、主模型及配置 SHA256。
总周期以首层 begin 到最后一层 result 为边界，不包括测试平台初始加载和结果后的 drain。
加速比为同一带宽/预测组中的 A 或 B 周期除以该 case 周期，不跨配置拼接。

`functional_passed` 和 `all_trends_match` 是不同指标；后者使用严格 >1 判断
“正收益/非正收益”，零收益与负收益同属非正收益，不是三值符号完全一致检查。
不设置事后容差掩盖差异。fast、错误预测时，主模型 D/B=1.0000×，
RTL D/B=828/832=0.9952×，说明模型没有完全覆盖这四拍控制损耗。
慢速点的 AXI phase、突发填充/退休和控制延迟尚非逐周期模型，不能据此宣称总周期一致。
命中组主要排序可用于检查调度方向；错误预测额外控制开销必须保留。

medium、全部命中的代表性结果如下。加速比均相对本表 A：

| Case | 主模型 cycles | RTL cycles | 主模型加速比 | RTL 加速比 |
|---|---:|---:|---:|---:|
| A 整 Expert、无预取 | 1131 | 1114 | 1.0000× | 1.0000× |
| B streaming、无预取 | 876 | 874 | 1.2911× | 1.2746× |
| C 整 Expert 预取 | 819 | 808 | 1.3810× | 1.3787× |
| D streaming、tile 预取 | 732 | 726 | 1.5451× | 1.5344× |

D 相对 B 的额外收益：主模型 1.1967×，RTL 1.2039×。这只对应本夹具的全部命中激励，
不是上面的 75.8%/84.4% Switch 预测质量。

最终 24 个 case 均通过数值/协议检查，四个 promotion 边界检查通过。
全部命中在三个带宽点均为 D 最快，其次 C、B、A；正收益/非正收益判断全部一致。
全矩阵最大总周期相对差为 8.20%（相对主模型），因此这里只验收调度趋势，
不宣称逐周期精确。RTL 错误预测 D/B 为 fast 0.9952×、medium 0.9954×、slow 0.9920×。

回归记录：452 个 Python 测试通过；78 个 RTL 运行点和 7 个数值 golden 检查通过。
Lint 检查默认、多层、完整 Expert 模式，没有 WIDTH/LATCH/MULTIDRIVEN/UNOPTFLAT
问题；仍有未使用信号、遗留顺序块赋值和 reset 使用等告警，未声称 warning-free。
日志保存在同一结果目录的 `pytest.log`、`rtl_regression.log`、`lint.log`。

## 5. 真实 Switch 的独立修正复跑

主模型另以原 `flame_head_quality.yaml` 条件复跑，保存于
`results/flame_head_quality_promotion_fixed/`；没有把小型 RTL 夹具当作真实 Switch 性能。
条件为 Switch-base-8、H=768/F=3072 两矩阵 ReLU、16 lanes、16 MiB OCM、六个 MoE 层、
100 序列，Top-1 候选质量 75.8%、Top-2 覆盖率 84.4%，关闭旧预取屏障。

| 标称 bytes/cycle/channel | 修正前 D/B | 修正后 D/B | 修正后 D cycles/sequence |
|---|---:|---:|---:|
| 8 | 0.984944× | 1.003133× | 1,827,152.46 |
| 16 | 1.007494× | 1.007494× | 1,781,814.92 |

低带宽点 B 仍为 1,832,877 cycles。修正消除了已命中关键请求被未来 speculation
排挤的问题，但净收益只有约 0.31%，不是新的大幅加速结论；错预测仍会消耗带宽。

## 6. 复现

从仓库根目录运行，需 Vivado 2023.2 的 xvlog/xelab/xsim 可用：

```bash
make -C hardware/m10_u200 four-cases
make -C hardware/m10_u200 rtl-test golden lint
pytest -o addopts='' -q
PYTHONPATH=src python examples/flame_condition_comparison.py \
  --config configs/experiments/flame_head_quality.yaml \
  --output results/flame_head_quality_promotion_fixed --workers 4
```

不调用已失效的 M10AlignmentSimulator，不根据 RTL 总周期反向拟合补偿常数。
