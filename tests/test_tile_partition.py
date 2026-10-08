"""Exact-total tile partition tests."""

from edge_moe_sim.schema import ExpertTask, TileConfig
from edge_moe_sim.tiling import FixedByteTilingPolicy, FixedTileCountPolicy
from edge_moe_sim.work_model import down_ops, down_weight_bytes, gate_up_ops, gate_up_weight_bytes


def test_tile_partition_exact_totals_and_remainder(small_model) -> None:
    """Byte and operation totals are preserved and the final tile gets remainders."""
    task = ExpertTask(0, 1, 1, 0.7)
    gate, down = FixedByteTilingPolicy().build_tiles(task, small_model, TileConfig(20))
    assert [tile.size_bytes for tile in gate] == [20, 20, 20, 4]
    assert [tile.size_bytes for tile in down] == [20, 12]
    assert sum(tile.size_bytes for tile in gate) == gate_up_weight_bytes(small_model)
    assert sum(tile.size_bytes for tile in down) == down_weight_bytes(small_model)
    assert sum(tile.compute_ops for tile in gate) == gate_up_ops(small_model, 1)
    assert sum(tile.compute_ops for tile in down) == down_ops(small_model, 1)
    assert gate[-1].compute_ops == gate_up_ops(small_model, 1) - sum(
        tile.compute_ops for tile in gate[:-1]
    )


def test_fixed_tile_count_preserves_totals(small_model) -> None:
    """Count-based test partitions never duplicate rounded work."""
    task = ExpertTask(0, 0, 1, 1.0)
    gate, down = FixedTileCountPolicy(3, 2).build_tiles(task, small_model, TileConfig(1))
    assert len(gate) == 3
    assert len(down) == 2
    assert sum(tile.size_bytes for tile in gate) == 64
    assert sum(tile.compute_ops for tile in gate) == 128
    assert sum(tile.size_bytes for tile in down) == 32
    assert sum(tile.compute_ops for tile in down) == 64

