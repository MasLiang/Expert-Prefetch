"""Physically meaningful intermediate-dimension tile groups."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ModelConfig, ValidationError
from .work_model import ceil_div, down_weight_bytes, gate_up_weight_bytes


@dataclass(frozen=True)
class FTileGroup:
    """One F-axis block with exact weight, compute, and nonlinear work."""

    expert_id: int
    tile_id: int
    num_tiles: int
    f_offset: int
    f_size: int
    token_count: int
    gate_up_bytes: int
    gate_up_ops: int
    nonlinear_elements: int
    down_bytes: int
    down_ops: int


def build_f_tile_groups(
    model: ModelConfig, expert_id: int, token_count: int, f_tile_size: int,
) -> tuple[FTileGroup, ...]:
    """Partition F exactly; weight bytes are independent of ``token_count``."""
    if token_count <= 0 or f_tile_size <= 0:
        raise ValidationError("token_count and f_tile_size must be positive")
    num_tiles = ceil_div(model.intermediate_size, f_tile_size)
    groups = []
    previous_gu_bytes = 0
    previous_down_bytes = 0
    for tile_id in range(num_tiles):
        offset = tile_id * f_tile_size
        size = min(f_tile_size, model.intermediate_size - offset)
        end_f = offset + size
        cumulative_gu = ceil_div(model.input_projection_count * model.hidden_size * end_f * model.weight_bits, 8)
        cumulative_down = ceil_div(model.hidden_size * end_f * model.weight_bits, 8)
        gu_bytes = cumulative_gu - previous_gu_bytes
        down_bytes = cumulative_down - previous_down_bytes
        previous_gu_bytes, previous_down_bytes = cumulative_gu, cumulative_down
        groups.append(FTileGroup(
            expert_id, tile_id, num_tiles, offset, size, token_count,
            gu_bytes, 2 * model.input_projection_count * token_count * model.hidden_size * size,
            token_count * size,
            down_bytes, 2 * token_count * size * model.hidden_size,
        ))
    if sum(group.f_size for group in groups) != model.intermediate_size:
        raise AssertionError("F tile partition changed intermediate size")
    if sum(group.gate_up_bytes for group in groups) != gate_up_weight_bytes(model):
        raise AssertionError("F tile partition changed Gate-Up weight bytes")
    if sum(group.down_bytes for group in groups) != down_weight_bytes(model):
        raise AssertionError("F tile partition changed Down weight bytes")
    return tuple(groups)
