"""Bit-exact integer golden model for the frozen M10 HardwareGreedy-v1 policy.

This module intentionally does not alter the historical M7--M9.3 scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


CONFIDENCE_MAX_Q12 = (1 << 12) - 1


@dataclass(frozen=True)
class HardwareGreedyV1Candidate:
    """One candidate Expert's exact logical-tile geometry and confidence."""

    expert_id: int
    candidate_rank: int
    confidence_q12: int
    logical_tile_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.expert_id < 0 or self.candidate_rank < 0:
            raise ValueError("expert_id and candidate_rank must be non-negative")
        if not self.logical_tile_bytes or any(value <= 0 for value in self.logical_tile_bytes):
            raise ValueError("logical_tile_bytes must be non-empty and positive")


@dataclass(frozen=True)
class HardwareGreedyV1Config:
    """Integer resource limits visible to one v1 allocation transaction."""

    num_memory_channels: int
    global_credit_bytes: int
    channel_credit_bytes: tuple[int, ...]
    priority_shift_per_depth: int = 1

    def __post_init__(self) -> None:
        if self.num_memory_channels <= 0 or self.global_credit_bytes < 0:
            raise ValueError("invalid HardwareGreedy-v1 global configuration")
        if len(self.channel_credit_bytes) != self.num_memory_channels:
            raise ValueError("channel_credit_bytes length must equal num_memory_channels")
        if any(value < 0 for value in self.channel_credit_bytes):
            raise ValueError("channel credits must be non-negative")
        if self.priority_shift_per_depth < 0:
            raise ValueError("priority_shift_per_depth must be non-negative")


@dataclass(frozen=True)
class HardwareGreedyV1Allocation:
    """One accepted prefix allocation in deterministic issue order."""

    expert_id: int
    tile_id: int
    memory_channel: int
    logical_bytes: int


def clamp_confidence_q12(value: int) -> int:
    """Clamp an external confidence to the frozen non-zero Q12 range."""
    return min(CONFIDENCE_MAX_Q12, max(1, value))


def hardware_greedy_v1(
    candidates: Sequence[HardwareGreedyV1Candidate],
    config: HardwareGreedyV1Config,
) -> tuple[HardwareGreedyV1Allocation, ...]:
    """Return the exact integer k-way prefix merge defined by M10 spec v0.9."""
    if len({candidate.expert_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate Expert IDs must be unique")
    if sorted(candidate.candidate_rank for candidate in candidates) != list(range(len(candidates))):
        raise ValueError("candidate ranks must be contiguous from zero")

    depths = [0] * len(candidates)
    loads = [0] * config.num_memory_channels
    remaining_global = config.global_credit_bytes
    remaining_channel = list(config.channel_credit_bytes)
    result: list[HardwareGreedyV1Allocation] = []

    while True:
        best_index: int | None = None
        best_numerator = 0
        best_denominator = 1
        for index, candidate in enumerate(candidates):
            depth = depths[index]
            if depth >= len(candidate.logical_tile_bytes):
                continue
            byte_count = candidate.logical_tile_bytes[depth]
            channel = candidate.expert_id % config.num_memory_channels
            if byte_count > remaining_global or byte_count > remaining_channel[channel]:
                continue
            priority = max(1, CONFIDENCE_MAX_Q12 >> (depth * config.priority_shift_per_depth))
            numerator = clamp_confidence_q12(candidate.confidence_q12) * priority
            denominator = byte_count * (1 + loads[channel])
            if best_index is None:
                best_index, best_numerator, best_denominator = index, numerator, denominator
                continue
            current = candidates[best_index]
            if (
                numerator * best_denominator > best_numerator * denominator
                or (
                    numerator * best_denominator == best_numerator * denominator
                    and (candidate.candidate_rank, depth, candidate.expert_id)
                    < (current.candidate_rank, depths[best_index], current.expert_id)
                )
            ):
                best_index, best_numerator, best_denominator = index, numerator, denominator
        if best_index is None:
            break
        candidate = candidates[best_index]
        depth = depths[best_index]
        byte_count = candidate.logical_tile_bytes[depth]
        channel = candidate.expert_id % config.num_memory_channels
        result.append(HardwareGreedyV1Allocation(candidate.expert_id, depth, channel, byte_count))
        depths[best_index] += 1
        loads[channel] += 1
        remaining_global -= byte_count
        remaining_channel[channel] -= byte_count
    return tuple(result)
