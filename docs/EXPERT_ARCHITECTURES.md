# 可配置 Expert 架构

`ModelConfig.expert_architecture` 与模型尺寸、位宽独立：

| 配置 | 计算结构 | 输入投影 ops | Expert 权重 byte |
|---|---|---|---|
| `swiglu`（默认） | fused Gate-Up → SiLU×Up → Down | `4*n*H*F` | `ceil(2HF*b/8)+ceil(HF*b/8)` |
| `relu_ffn` | Up → ReLU → Down | `2*n*H*F` | `2*ceil(HF*b/8)` |

两种架构的 Down 都是 `2*n*F*H` operations；非线性 workload 都是
`n*F` elements，但算子不同。`hardware.nonlinear.elements_per_cycle` 或主模型的
`AdvancedExecutionConfig.nonlinear_elements_per_cycle` 必须由实验针对相应算子配置；
模拟器不会假定 ReLU 与 SwiGLU 有相同的真实性能，也不会自动套用理论峰值。

## 兼容性与扩展

现有配置不增加字段时保持 SwiGLU。`fuse_gate_up=false` 对 SwiGLU 仍不支持；
对 ReLU 可以设为 false，表达不存在 fused Gate-Up。输入阶段内部键仍叫
`gate_up`（包括 timeline、DMA、性能字段），在 ReLU 模型中表示单矩阵 Up，
不是又多算了一张 Gate 矩阵。具体算子可通过 `model.nonlinear_operation` 查询。

Stage、旧 weight tiling、F-axis tiling 和主 UnifiedSequenceSimulator 共享架构工作量。
调度算法、DMA、OCM 和依赖图不需要针对 Switch 写另一套。
新增 GELU、其他 gated FFN 或 shared Expert 时，应显式扩展架构与测试；
未知架构直接报错，不按最接近的架构静默运行。

## Switch 示例

`configs/models/switch_base_8.yaml` 和 `switch_base_16.yaml` 提供官方模型维度，
INT8 是显式部署假设，不是 FLAME 的实现精度。配置仅列 decoder 的六个 MoE 层；
不包含 encoder、attention 或其他非 MoE 计算。路由和 predictor trace 仍需提供。
现有 `real_moe_catalogue.yaml` adapter 仍只接受 Mixtral/OLMoE，Switch 请用
`load_model_config` 加载独立 YAML；旧真实模型实验结果没有被改写。

该版本是未剪枝、无 bias 的两矩阵 FFN 性能负载，不执行真实 tensor 数值计算。
尚未加入 FLAME N:M 压缩、索引开销或真实 CEPR，因此不能称为 FLAME 精确复现。

验证命令：`pytest -q tests/test_expert_architecture.py`；完整回归：`pytest -q`。

主模型示例：`python examples/switch_workload.py --model configs/models/switch_base_8.yaml`。
更换为 `switch_base_16.yaml` 可切换模型。此脚本硬件为显式 synthetic placeholder，
没有预取，只验证六层数据通路和工作量，不用于推导真实性能收益。
