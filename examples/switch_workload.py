"""使用可选模型 YAML 运行主模拟器；所有硬件参数均为 synthetic placeholder。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.config import load_model_config
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.unified_schema import UnifiedLayerSequence
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator
from edge_moe_sim.work_model import expert_weight_bytes


def main() -> None:
    """输出未剪枝、无预取、合成路由的 MoE-only 示例，不代表 FLAME 性能。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "configs/models/switch_base_8.yaml")
    args = parser.parse_args()
    model = load_model_config(args.model)
    # 显式 synthetic placeholder，非 U200 测量值或 FLAME 硬件配置。
    hardware = HardwareProfile("synthetic_placeholder", 1000, 2, 8000, 1, 1, 1,
                               32, True, 4, 4, 1, 1, 16 * 1024 * 1024)
    layers = model.moe_layer_indices or (0,)
    sequence = UnifiedLayerSequence(tuple(
        BatchRoutingEvent(1, 0, layer, (tuple(range(model.top_k)),)) for layer in layers
    ))
    result = UnifiedSequenceSimulator(model, hardware).simulate(
        sequence, AdvancedExecutionConfig(128, aie_units_per_expert=1,
                                         max_parallel_experts=1, candidate_count=model.top_k)
    )
    print(json.dumps({
        "model": model.name, "expert_architecture": model.expert_architecture,
        "nonlinear_operation": model.nonlinear_operation,
        "hardware": "SYNTHETIC_PLACEHOLDER_NOT_FLAME_OR_U200_MEASUREMENT",
        "synthetic_routing": True, "prefetch": False, "moe_layers": len(layers),
        "expert_weight_bytes": expert_weight_bytes(model),
        "total_cycles": result.inference_makespan_cycles,
        "demand_bytes": sum(layer.demand_bytes for layer in result.layer_results.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
