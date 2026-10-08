"""M9.1 realistic-scale workload and hardware profile validation."""
from __future__ import annotations

from pathlib import Path

import yaml

from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.workload_profile import load_workload_profile

ROOT = Path(__file__).resolve().parents[1]


def test_workload_scaling_and_tile_counts() -> None:
    """The three profiles scale H/F/E while retaining exact 16-tile geometry."""
    profiles = [
        load_workload_profile(ROOT / "configs/workloads" / name)
        for name in ("small_edge_moe.yaml", "medium_moe.yaml", "large_moe.yaml")
    ]
    assert [(p.model.hidden_size, p.model.intermediate_size, p.model.num_experts) for p in profiles] == [
        (512, 1024, 64), (1024, 4096, 128), (2048, 8192, 256),
    ]
    assert [p.candidate_count for p in profiles] == [16, 32, 64]
    assert all(p.tiles_per_expert == 16 and p.model.top_k == 2 for p in profiles)
    assert [p.num_moe_layers for p in profiles] == [8, 16, 24]
    assert all(p.provenance == "representative synthetic derived from MoE scaling" for p in profiles)


def test_workload_tile_bytes_and_operations_conserve_expert_totals() -> None:
    """Per-tile bytes/operations scale exactly to full Expert work."""
    for name in ("small_edge_moe.yaml", "medium_moe.yaml", "large_moe.yaml"):
        profile = load_workload_profile(ROOT / "configs/workloads" / name)
        assert profile.bytes_per_logical_tile * profile.tiles_per_expert == profile.expert_bytes
        assert profile.gate_up_bytes_per_tile == 2 * profile.down_bytes_per_tile
        assert profile.gate_up_ops_per_tile == 2 * profile.down_ops_per_tile
    large = load_workload_profile(ROOT / "configs/workloads/large_moe.yaml")
    assert large.bytes_per_logical_tile == 3_145_728
    assert large.gate_up_ops_per_tile + large.down_ops_per_tile == 6_291_456


def test_realistic_hardware_profiles_validate_and_are_labeled() -> None:
    """All characterization hardware files load while preserving provenance labels."""
    expected = {
        "vek280_theoretical.yaml": (3, "THEORETICAL_NOT_MEASURED"),
        "v80_like_theoretical.yaml": (8, "THEORETICAL_NOT_MEASURED"),
        "generic_fpga_streaming.yaml": (4, "SYNTHETIC_PLACEHOLDER"),
    }
    for name, (channels, tag) in expected.items():
        raw = yaml.safe_load((ROOT / "configs/hardware" / name).read_text(encoding="utf-8"))
        profile = HardwareProfile(**raw["hardware_profile"])
        assert profile.num_memory_channels == channels
        assert profile.hardware_tag == tag
        assert raw["provenance"]["notes"]
