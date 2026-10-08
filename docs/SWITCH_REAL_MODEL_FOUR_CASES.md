# 主模拟器：真实 Switch 模型尺寸四 case 实验

## 范围与来源

本轮使用主 `UnifiedSequenceSimulator`，不是 M10AlignmentSimulator 或 H16/F32 RTL 夹具。
模型尺寸来自 Google 官方配置：

- [switch-base-8](https://huggingface.co/google/switch-base-8/blob/main/config.json)。
- [switch-base-16](https://huggingface.co/google/switch-base-16/blob/main/config.json)。

两者 H=768、F=3072、Top-1、非 gated ReLU FFN；Expert 数分别为 8 和 16。
官方配置为 encoder-decoder，decoder 12 层，其中 6 个 sparse 层。
本轮只模拟 decoder 的六个 MoE 层 `[1,3,5,7,9,11]`，不模拟 encoder、attention、
dense FFN、layer norm、tokenization 或输出 head，因此结果不是完整模型端到端推理延迟。

官方 checkpoint 配置标注 bfloat16、Router float32。本轮 **INT8 权重/activation 和配置的
Router 有效性能是部署假设**，不是原 checkpoint 数值格式；未加载真实权重运行精度评估。
路由与 predictor 为受控 synthetic trace，不把论文统计准确率当成真实 CEPR trace。

## 工作量与共同配置

每个未剪枝 ReLU Expert 两矩阵权重：`2×768×3072=4,718,592 bytes`，即 INT8 下 4.5 MiB。
每个 token 的 Expert 工作量为 `4×768×3072=9,437,184 operations`。
第一矩阵在 simulator ABI 中仍称 `gate_up`，此模式实际只有 `W_up`，不是 SwiGLU。
F_TILE=128，每 Expert 24 个 F tiles；每个矩阵 tile 为 98,304 bytes（96 KiB）。

全部 A/B/C/D 使用相同约束：

- 一个共享 Router/Expert Engine，16 MAC lanes，配置有效性能 32 operations/cycle。
- 配置时钟 200 MHz；四通道、标称 8 或 16 bytes/cycle/channel；payload cycle 倍率为 2。
  因此 payload 有效服务率分别为每通道 4 或 8 bytes/cycle，另计 DMA/control 成本。
- shared byte OCM=16 MiB，可容纳两个完整 INT8 Expert；无 N:M pruning。
- 配置 nonlinear 吞吐率 16 elements/cycle、MAC task startup=8 cycles；不是当前 RTL
  逐元素 nonlinear 控制流水的逐周期复现。
- DMA startup=8 cycles、burst=4096 bytes、burst 边界 demand 优先；命中 promotion 使用已修正语义。
- 100 个独立 single-token 六 MoE 层序列，seed=20261005；每个供数点四 case 复用同一 truth/candidates。
  每个序列独立冷启动，不包含跨 token 的 Expert cache 复用。
- 下一 MoE 层在源 Router done 后 20 cycles 获得预测，提前一个 MoE 层；首层冷启动。
- 每层两个候选，第一名覆盖率 379/500=75.8%，候选集合覆盖率 422/500=84.4%。
  scores=0.6/0.4 是受控排序值，不是校准概率；两个论文最大值只作为独立敏感性假设。
- 主比较关闭“前层全部权重完成”屏障；保留屏障开启的 C/D 消融，不拼接为主收益。

这组主模型实验沿用研究侧 `byte OCM / unlimited request credits / expert_locked` 配置，
不是 RTL 固定 slot、有限 descriptor/FIFO 和 atomic pair 的逐周期复现。
Top-1 单 token 不涉及多个真实 Expert 的 compute 交错，但 admission 和控制容量仍不同。
配置的 MAC/BW 不代表已测得的 U200/DDR 吞吐率。

## 四 case 定义

| Case | 权重需求与预取 | 计算启动 |
|---|---|---|
| A_whole_demand | 无预测，完整 Expert demand | 全部 Up/Down 到齐 |
| B_stream_demand | 无预测，tile streaming demand | 首 Up tile 就绪 |
| C_whole_prefetch | 预测完整候选 Expert，受 OCM/budget 约束 | 真实 Expert 全部 Up/Down 到齐 |
| D_stream_tile_prefetch | 每候选 Up 前2个 tiles，Down 按需求 | 首 Up tile 就绪 |

D 使用已有主模型 `uniform_prefix`，不是小型 RTL 实验中的第一 GU/Down pair。
C 请求完整候选，但多层共享容量可能限制实际驻留；“candidate hit”不等于
“真实完整 Expert 在 Router 时已驻留”。输出记录两者，不以静态命中率冒充 ready recall。

## 输出与复现

```bash
MPLCONFIGDIR=/tmp/m10-switch-real-matplotlib \
XDG_CACHE_HOME=/tmp/m10-switch-real-cache \
PYTHONPATH=src python examples/flame_condition_comparison.py \
  --config configs/experiments/switch_real_model_four_cases.yaml \
  --output results/switch_real_model_four_cases --workers 4
pytest -o addopts='' -q
```

结果目录包含 `comparison.csv`、`per_sample.csv`、`per_seed.csv`、`per_prediction.csv`、
两模型的保存输入 JSON、`summary.json` 与两组 PNG/PDF 图：
`speedup_vs_A`、`full_true_ready_recall`。图中的 barrier=true 为额外消融。
summary 保存主模型/驱动/配置源码哈希、真实模型声明和实验边界。
主结果必须按同模型、带宽、容量、seed 的 A/B 周期计算加速比。

## 已完成结果

完成 24 个配置点、2,400 次模拟；其中 16 个为关闭屏障的 A/B/C/D 主点，
8 个为 C/D 屏障开启消融。全部源码哈希已核对；464 个 Python 测试通过。
每模型每 case/BW 为相同 seed 的100个序列均值，不是100-token batch。
本轮所有点均只使用一个 seed，不包含三 seed reference 或额外容量 probe。

平均 cycles/六 MoE 层序列（主比较均 barrier=false）：

| 模型 | 标称 B/cycle/channel | A 整 Expert无预取 | B streaming无预取 | C 完整Expert预取 | D tile预取 |
|---|---:|---:|---:|---:|---:|
| Switch-base-8 | 8 | 3,569,613 | 1,832,877 | 2,449,911.67 | 1,827,152.46 |
| Switch-base-8 | 16 | 2,684,877 | 1,795,167 | 2,049,030.28 | 1,781,814.92 |
| Switch-base-16 | 8 | 3,571,965 | 1,835,229 | 2,449,834.61 | 1,828,145.34 |
| Switch-base-16 | 16 | 2,687,229 | 1,797,519 | 2,050,992.00 | 1,784,166.92 |

相对同配置 A 的加速比与相对 B 的额外收益：

| 模型 | 标称 B/cycle/channel | B/A加速比 | C/A加速比 | D/A加速比 | D/B加速比 |
|---|---:|---:|---:|---:|---:|
| Switch-base-8 | 8 | 1.9475× | 1.4570× | 1.9536× | 1.0031× |
| Switch-base-8 | 16 | 1.4956× | 1.3103× | 1.5068× | 1.0075× |
| Switch-base-16 | 8 | 1.9463× | 1.4580× | 1.9539× | 1.0039× |
| Switch-base-16 | 16 | 1.4950× | 1.3102× | 1.5062× | 1.0075× |

这里“D/A加速比”指 `cycles(A)/cycles(D)`，不是 D 的延迟除以 A。

## 解释与结论边界

四点都为 D最快、B其次、C再次、A最慢。不能因 C相对A有1.31–1.46×收益，
就认为 C优于B：C/B只有约0.75–0.88×。

真实尺寸下，每个 Expert 的有用计算量较大，B 已通过 tile streaming 隐藏了大部分 DMA。
例如 Switch-base-8 的有用 MAC payload 为1,771,776 cycles，B低带宽总周期为1,832,877；
即使把所有剩余开销消除，受该有用计算下界限制，B最多约有1.0345×提升空间。
这只是固定吞吐配置下的上界估算，不是可实现的完美调度。
因此 D相对B的实际额外收益约0.31%–0.75%，不能推广小型、全部命中 RTL 夹具中约17.7%的收益。
完全预测错误仍可能使预取变慢；本轮是75.8%/84.4%的受控候选统计，不是零hit/全hit边界。

本轮没有修改 simulator 或 Scheduler 语义，只增加实验配置、真实尺寸/候选统计验证测试，
并修正驱动中已过时的“RTL仍有weights_done屏障”说明。没有下载预训练权重或验证INT8精度。
