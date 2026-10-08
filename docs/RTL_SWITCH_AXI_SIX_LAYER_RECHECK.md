# 六层 Switch：回退四通道 AXI 行为模型重跑

## 2026-10-07 的清理与后端选择

按用户要求停止了完整 MIG/RDIMM 器件模型长跑。核验各进程工作目录与独立
进程组后，停止驱动及四组 XSim，确认进程退出；释放约68GiB内存。
两个完整尺寸 DDR 实验目录与三个启动日志移入回收站（约1.9GiB，可恢复）。
共享 RTL/XDC、DDR 验证代码、小尺寸 DDR 结果、此前 AXI 结果保留。
清理记录为新结果目录的 `cleanup_receipt.json`。

本轮只使用四通道 AXI 行为响应器，不实例化 MIG、CDC 或 RDIMM 模型。
响应器按实际地址生成测试权重，保留 AXI 握手、背压、burst 与 byte 检查。
`startup=0`、`beat_gap=0` 是可配置的测试设置，不是 U200 DDR 实測参数；
即使两个参数为零，beat 仍受响应器与 DUT 握手周期限制，不是零周期加载。
不把上轮小模型测得38–41cycles直接当成完整负载固定延迟。

## 完整尺寸与相同硬件约束

- H=768、F=3072、E=8、Top-1；单 token、六个串联 MoE 层。
- int8 两矩阵 ReLU Expert，FT=128，每 Expert 24 logical tiles。
- 每 Expert 4.5MiB；六层真实权重总计27MiB。
- 单共享16-lane Router/Expert MAC，模拟时钟200MHz。
- OCM=16MiB，128个128KiB slots；每矩阵fragment实际payload为96KiB。
- 一个逻辑 DMA 通道，striping 到四个128-bit AXI bank，max burst=256B。
- 预取payload预算为9MiB；C预测两个候选完整深度，D用八候选整数HardwareGreedy-v1。
- 默认Top-1全命中预测，错误的其他候选仍可能产生无用流量。

这是完整尺寸 MoE 子图，不是含 attention/norm/residual 的完整 Transformer。
使用确定性fixture权重而非训练checkpoint；数值稀疏但不跳零，dense payload
和全部MAC均实际执行。不宣称该16MiB OCM配置已在U200完成资源映射或时序收敛。

## 四策略及验收

| Case | 策略 |
|---|---|
| A_whole_demand | 无预取，完整真实Expert的UP/DOWN全部resident后计算 |
| B_stream_demand | 无预取，GU tile ready即可开始计算 |
| C_whole_full_prefetch | 完整候选Expert预取，真实Expert全部resident后计算 |
| D_stream_tile_prefetch | 八候选tile预算分配与预取，GU ready streaming |

每组必须完成六层的独立逐行数值检查、Router/gating检查、144次GU及144次DOWN、
OCM容量与物理/telemetry byte守恒。性能区间为第一层begin到第六层结果，
不含第一层Router权重初始化与最终drain，包含中间层间控制与Router权重reload。
时间分解使用真实事件间隔，不能把所有间隔或compute context busy称为纯MAC时间。
源文件运行前后hash校验通过后，才允许终止summary写成PASS。

## 本次输出与复现

独立输出目录：`results/m10_switch_axi_six_layers_20261007`。
新运行使用当前RTL重建两个snapshot，再将四组放入隔离runtime目录。
四组有完整simulation/console日志和timeline.csv；根目录有summary.json、
comparison.csv、timeline_partition.json。不能将历史结果复制成当前实跑结果。

```bash
python -u hardware/m10_u200/scripts/run_switch_full_size.py \
  --experts 8 --layers 6 --jobs 4 --startup 0 --beat-gap 0 \
  --output results/m10_switch_axi_six_layers_new_run
python hardware/m10_u200/scripts/analyze_switch_full_size_timelines.py \
  results/m10_switch_axi_six_layers_new_run
pytest -q
```

使用新的输出目录。结果以四组显式TEST_PASS及根目录终止status=PASS为准。

## 实际结果：四组全部 PASS

| Case | cycles | 200MHz换算ms | 实际payload读取MiB | speedup vs A | speedup vs B |
|---|---:|---:|---:|---:|---:|
| A：整Expert按需加载后计算 | 3,013,249 | 15.066245 | 27.00000 | 1.0000× | 0.6045× |
| B：streaming无预取 | 1,821,517 | 9.107585 | 27.00000 | 1.6543× | 1.0000× |
| C：完整Expert预取 | 1,998,342 | 9.991710 | 37.78125 | 1.5079× | 0.9115× |
| D：tile预取＋streaming | 2,006,224 | 10.031120 | 40.40625 | 1.5020× | 0.9079× |

四组metrics和timeline.csv均与此前 `results/m10_switch_full_size_verified` 完全一致，
不是复用旧snapshot：当前源码重新编译与展开，四组独立执行并通过自检。
全层逐行输出依次为384、576、432、624、456、672；144个GU、144个DOWN与
所有bytes/OCM检查均通过，运行前后source hash保持一致。
Python回归 **516 tests passed**，14条既有依赖弃用警告。

主结论仍然是B最快。C/D相对A均有收益，但相对B分别慢约9.7%和10.1%；
这是当前硬件/预算/预测fixture的结果，不是预取普遍无效的证明。
本轮预测Top-1全命中，但其他候选可能错误；不能将此结果标成CEPR 84.4%实测。

### D 相对 B 的周期与bytes核对

- 首个GU开始的各层累计间隔减少17,810cycles。
- GU→对应DOWN开始累计增加133,696cycles。
- DOWN→下一GU开始累计增加68,821cycles。
- 最后DOWN→结果、层间间隔差均为0。
- 合计增加184,707cycles，精确等于两个makespan之差。

这些是观测事件间隔，包含计算/控制/等待；不能全部标为memory stall。
D的DOWN等待信号累计133,692cycles，B为0。D有用speculative为9.1875MiB，
demand为17.8125MiB，总计所需27MiB；wasted speculative另增13.40625MiB。
取消的21.46875MiB未计入实际传输。峰值slot占用A/B=48、C/D=112，未超过128。

本轮四组运行wall-time分别约882.1s、502.4s、647.2s、570.2s，不含源码构建。
模拟时间ms用于比较硬件执行，wall-time仅表示软件仿真成本，两者不可混淆。
结束后确认无残留XSim进程。旧DDR长跑已取消且输出回收，不会继续占用资源。

后处理修复了backend说明中“NOT MIG”导致误判输出路径的问题，新增四个布局
识别测试；没有修改调度器、计算或DMA RTL，没有修改任何XDC。
