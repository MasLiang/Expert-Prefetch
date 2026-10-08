"""Tests for the independent M10 HardwareGreedy-v1 golden model."""

from edge_moe_sim.hardware_greedy_v1 import (
    HardwareGreedyV1Candidate,
    HardwareGreedyV1Config,
    hardware_greedy_v1,
)


def test_hardware_greedy_v1_prefix_channel_and_tie_rules() -> None:
    """The v1 model obeys prefix closure and rank tie-breaking."""
    result = hardware_greedy_v1(
        (
            HardwareGreedyV1Candidate(0, 0, 4095, (100, 100)),
            HardwareGreedyV1Candidate(1, 1, 4095, (100, 100)),
        ),
        HardwareGreedyV1Config(2, 300, (200, 200)),
    )
    assert [(item.expert_id, item.tile_id, item.memory_channel) for item in result] == [
        (0, 0, 0), (1, 0, 1), (0, 1, 0),
    ]


def test_hardware_greedy_v1_clamps_confidence_and_channel_credit() -> None:
    """A high input confidence cannot exceed Q12 and channel limits are hard."""
    result = hardware_greedy_v1(
        (HardwareGreedyV1Candidate(1, 0, 9999, (64, 64)),),
        HardwareGreedyV1Config(2, 128, (0, 64)),
    )
    assert [(item.expert_id, item.tile_id) for item in result] == [(1, 0)]
