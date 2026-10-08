"""Shared small, hand-calculable fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_moe_sim.schema import (
    AIEConfig, DMAConfig, HardwareConfig, ModelConfig, NonlinearConfig, PLControlConfig,
)
from edge_moe_sim.synthetic import make_decode_event


@pytest.fixture
def small_model() -> ModelConfig:
    """Return H=4, F=8, E=4, K=2, 8-bit weights."""
    return ModelConfig("tiny", 4, 8, 4, 2, 8, 8)


@pytest.fixture
def small_hardware() -> HardwareConfig:
    """Return simple integer rates suitable for manual timing."""
    return HardwareConfig(
        1_000_000_000,
        AIEConfig(16, 16, 16, 2, 1),
        DMAConfig(16, 1),
        NonlinearConfig(8, 1),
        PLControlConfig(1, 1, 0, 0, 2),
    )


@pytest.fixture
def two_expert_event():
    """Return one top-2 decode event."""
    return make_decode_event((0, 1), gating_weights=(0.6, 0.4))
