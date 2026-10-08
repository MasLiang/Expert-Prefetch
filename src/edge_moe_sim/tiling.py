"""Exact-total weight tiling policies for streaming simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schema import ExpertTask, ModelConfig, TileConfig, ValidationError, WeightTile
from .work_model import down_ops, down_weight_bytes, gate_up_ops, gate_up_weight_bytes


@dataclass(frozen=True)
class ExpertTileSet:
    """All Gate-Up and Down tiles belonging to one routed expert."""

    expert_task: ExpertTask
    gate_up_tiles: tuple[WeightTile, ...]
    down_tiles: tuple[WeightTile, ...]


class WeightTilingPolicy(Protocol):
    """Interface that partitions both matrices without changing totals."""

    def build_tiles(
        self,
        expert_task: ExpertTask,
        model_config: ModelConfig,
        tile_config: TileConfig,
    ) -> tuple[list[WeightTile], list[WeightTile]]:
        """Return Gate-Up tiles followed by Down tiles."""
        ...


def _partition_sizes(total_bytes: int, requested_tiles: int) -> list[int]:
    if requested_tiles <= 0:
        raise ValidationError("requested tile count must be positive")
    if requested_tiles > total_bytes:
        raise ValidationError("cannot create more positive-byte tiles than matrix bytes")
    base, remainder = divmod(total_bytes, requested_tiles)
    return [base + (1 if tile_id < remainder else 0) for tile_id in range(requested_tiles)]


def _tiles_from_sizes(
    task: ExpertTask,
    matrix_type: str,
    sizes: list[int],
    total_ops: int,
) -> list[WeightTile]:
    total_bytes = sum(sizes)
    allocated_ops = 0
    tiles: list[WeightTile] = []
    for tile_id, size_bytes in enumerate(sizes):
        compute_ops = (
            total_ops - allocated_ops
            if tile_id == len(sizes) - 1
            else total_ops * size_bytes // total_bytes
        )
        allocated_ops += compute_ops
        tiles.append(WeightTile(
            task.layer_id, task.expert_id, matrix_type, tile_id,
            len(sizes), size_bytes, compute_ops,
        ))
    if sum(tile.size_bytes for tile in tiles) != total_bytes:
        raise AssertionError("tile byte partition changed total bytes")
    if sum(tile.compute_ops for tile in tiles) != total_ops:
        raise AssertionError("tile operation partition changed total operations")
    return tiles


class FixedByteTilingPolicy:
    """Split by a maximum byte size while respecting a minimum tile count."""

    def build_tiles(
        self, expert_task: ExpertTask, model_config: ModelConfig, tile_config: TileConfig
    ) -> tuple[list[WeightTile], list[WeightTile]]:
        """Build exact-total partitions for both Expert matrices."""
        gate_bytes = gate_up_weight_bytes(model_config)
        down_bytes = down_weight_bytes(model_config)

        def sizes(total_bytes: int) -> list[int]:
            natural_count = (total_bytes + tile_config.tile_size_bytes - 1) // tile_config.tile_size_bytes
            count = max(natural_count, tile_config.min_tiles_per_matrix)
            if count == natural_count:
                return [
                    min(tile_config.tile_size_bytes, total_bytes - offset)
                    for offset in range(0, total_bytes, tile_config.tile_size_bytes)
                ]
            return _partition_sizes(total_bytes, count)

        return (
            _tiles_from_sizes(expert_task, "gate_up", sizes(gate_bytes), gate_up_ops(model_config, expert_task.token_count)),
            _tiles_from_sizes(expert_task, "down", sizes(down_bytes), down_ops(model_config, expert_task.token_count)),
        )


class FixedTileCountPolicy:
    """Partition matrices into configured counts for analytical tests."""

    def __init__(self, gate_up_tiles: int, down_tiles: int) -> None:
        if gate_up_tiles <= 0 or down_tiles <= 0:
            raise ValidationError("fixed tile counts must be positive")
        self.gate_up_tiles = gate_up_tiles
        self.down_tiles = down_tiles

    def build_tiles(
        self, expert_task: ExpertTask, model_config: ModelConfig, tile_config: TileConfig
    ) -> tuple[list[WeightTile], list[WeightTile]]:
        """Build exact-total count-based partitions."""
        gate_sizes = _partition_sizes(gate_up_weight_bytes(model_config), self.gate_up_tiles)
        down_sizes = _partition_sizes(down_weight_bytes(model_config), self.down_tiles)
        return (
            _tiles_from_sizes(expert_task, "gate_up", gate_sizes, gate_up_ops(model_config, expert_task.token_count)),
            _tiles_from_sizes(expert_task, "down", down_sizes, down_ops(model_config, expert_task.token_count)),
        )
