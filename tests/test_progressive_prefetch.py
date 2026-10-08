"""Progressive depth and byte-budget allocation tests."""

from edge_moe_sim.prefetch_policy import FixedByteBudgetPolicy, ProgressiveRankPolicy
from edge_moe_sim.schema import CandidatePrefetch, PredictorOutput
from edge_moe_sim.tiling import ExpertTileSet, FixedTileCountPolicy
from edge_moe_sim.schema import ExpertTask, TileConfig


def _tile_sets(model):
    policy = FixedTileCountPolicy(4, 2)
    result = {}
    for expert_id in (0, 1):
        task = ExpertTask(0, expert_id, 1, 1.0 - expert_id * 0.1)
        gate, down = policy.build_tiles(task, model, TileConfig(1024))
        result[expert_id] = ExpertTileSet(task, tuple(gate), tuple(down))
    return result


def test_progressive_prefetch_depth(small_model) -> None:
    """Rank zero receives three Gate-Up tiles and rank one receives one."""
    output = PredictorOutput(
        0, 0, 0, ready_offset_cycles=0,
        candidates=(CandidatePrefetch(0, 0.9), CandidatePrefetch(1, 0.8)),
    )
    decisions = ProgressiveRankPolicy((3, 1), (0, 0)).allocate(
        output, _tile_sets(small_model), None
    )
    assert [(decision.expert_id, decision.tile_ids) for decision in decisions] == [
        (0, (0, 1, 2)), (1, (0,)),
    ]


def test_fixed_byte_budget_whole_tiles_and_deterministic_order(small_model) -> None:
    """Allocation never splits a tile or exceeds a 48-byte budget."""
    output = PredictorOutput(
        0, 0, 0, ready_offset_cycles=0,
        candidates=(CandidatePrefetch(0, 0.9), CandidatePrefetch(1, 0.8)),
    )
    tiles = _tile_sets(small_model)
    decisions = FixedByteBudgetPolicy(False).allocate(output, tiles, 48)
    allocated = [
        (decision.expert_id, tile_id)
        for decision in decisions for tile_id in decision.tile_ids
    ]
    used = sum(tiles[expert_id].gate_up_tiles[tile_id].size_bytes for expert_id, tile_id in allocated)
    assert allocated == [(0, 0), (0, 1), (1, 0)]
    assert used == 48

