# 第二轮 RTL 优化：下一 prefix 注册搜索

## 范围与依赖

本轮只优化 `tile_execution_scheduler.sv` 的注册搜索，主 Simulator、compute controller、
MAC、DMA、OCM admission/释放和数值算法不变。组合输出拆成连续 assign；保持每个 always_ff
只更新一个寄存字段的代码风格。

旧 scheduler 在当前 tile 退休、next_tiles 更新后，才开始搜索下一 prefix。
新 scheduler 对活动 Expert 使用 `search_tiles = next_tiles + 1`，在当前计算期间就注册
下一 prefix 的 GU/DOWN slot 索引；其他 Expert 继续搜索自己的当前 prefix。
不新增计算 context，不提前发请求或消费 OCM ownership。

退休后 next_tiles 递增，已有快照才与当前 prefix 相符；仍保留 match_tile、match_expert、
valid_mem、GU ready 和同周期 capture 冲突重验。不把未来 tile 当成当前 tile 启动。
只有纯退休事件不再强制搜索刷新：capture、DMA_DONE、route_accept 和 clear 仍触发 stale
保护。若执行太短、下一 tile 未驻留或发生 slot 复用，不满足重验条件就仍需等待。
稳定请求寄存级保留，背压期间 descriptor 与 generation 不变。

Expert 最后一个 tile 的 lookahead 指向范围外 sentinel，不能错误执行 tile 16；
combine 三段与最终 layer 输出保持原流程。

## 实测周期

使用完整默认 32 tiles/layer 的八点 XSim 2023.2 自检，比较第一轮冻结基线
`results/m10_compute_handoff/`。本轮每点再减少 30 cycles。

| 用例 | 第一轮 cycles | 第二轮 cycles | 再减少 |
|---|---:|---:|---:|
| demand_fast / wrong_fast | 13845 | 13815 | 30 |
| demand_gaps / wrong_gaps | 13852 | 13822 | 30 |
| hit_fast / second_hit_fast | 13747 | 13717 | 30 |
| hit_gaps / second_hit_gaps | 13747 | 13717 | 30 |

两个 Expert 各有 15 次内部相邻 tile 交接；在这些 successor 已就绪的点，每次节省一轮
搜索刷新。累计 ready→accept 从 63 降为 33 cycles，accept→GU 则保持第一轮的
demand 32 / hit 35 cycles。首次启动、GU 提交等待、Expert combine 没有被绕过。
这不是宣称所有模型都固定节省 30 cycles：慢 DMA、低容量或并发 metadata 变化时收益会不同。

相对最初 GU-only 版本，两轮累计节省 demand/wrong 94 cycles，hit 93 cycles。
GU/DOWN 执行时长仍为每 tile 256 / 128 cycles，结果向量、路由与实际传输字节分类守恒。

## 实际验证

- 240 项 Python 测试通过；新增 scheduler 单寄存器 always 风格检查。
- 56 个 RTL 回归运行点通过，含新增 lookahead TB 的 8-slot 与 128-slot 两种树规模。
- 新 TB 覆盖预就绪 successor 在退休后直接准备稳定请求、退休同周期 DMA_DONE、
  高优先级 Expert 变就绪、下一 tile slot 被替换为错误 Expert 后以新 generation 恢复、
  clear 与退休碰撞、背压期间无关 DMA completion，以及活动 descriptor 不被 lookahead 覆盖。
- 八个完整配置 XSim 用例正常结束并 TEST_PASS，每点必须比冻结基线更快。
  检查 32 tiles 的 prefix/单次执行、stage payload 因果、GU/DOWN 运算时长、数值/字节/
  allocation 守恒、OCM/FIFO/descriptor 容量和 MAC/DMA 互斥。
- allocator 与 dense SwiGLU 数值 golden 通过；lint 成功，保留既有非结构性警告。

使用显式 non-project RTL 源列表，未修改既有 Vivado 项目；旧 MCP 项目 PART 并非 U200，
不作为本轮综合证明。没有新综合、网表仿真、STA 或板级测试，不宣称 200 MHz 已闭合。

## 复现与产物

```bash
PYTHONPATH=src pytest -o addopts='' -q
make -C hardware/m10_u200 rtl-test lint golden
make -C hardware/m10_u200 scheduler-test
```

结果目录 `results/m10_scheduler_lookahead/` 包含输入/RTL 哈希、编译展开日志、逐例 TEST_PASS
日志、comparison JSON/CSV、tile/stage timing、事件及 timeline。
上一轮目录 `results/m10_compute_handoff/` 和 GU-only 目录 `results/m10_gu_only_start/` 保持冻结。
通用验证、第一轮和第二轮重跑默认使用另外的 current 输出目录，避免覆盖冻结基线。
当前 `scheduler-test` 重跑输出为 `results/m10_scheduler_lookahead_current/`。

## 未实现项

GU_DONE 后独立回收 GU slot、DMA 请求 lookahead、多 Expert context、MAC 流水吞吐以及
主模型多层联合对齐仍未完成。本轮 lookahead 是计算候选搜索，不是新增 DMA prefetch 策略。
