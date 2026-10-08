"""Tile-level unlimited OCM state and prefix queries."""

from edge_moe_sim.hardware.ocm import OnChipMemory, WeightState


def test_tile_ocm_state_and_contiguous_prefix() -> None:
    """A hole terminates the prefix even when a later tile is resident."""
    ocm = OnChipMemory()
    matrix = (0, 2, "gate_up")
    ocm.register_tiled_matrix(matrix, 4)
    for tile_id in (0, 1, 3):
        key = (*matrix, tile_id)
        ocm.mark_tile_queued(key, "demand")
        ocm.mark_tile_in_flight(key, "demand", 10 + tile_id)
        ocm.mark_tile_resident(key, "demand", 10 + tile_id)
    assert ocm.get_tile_state((*matrix, 0)) is WeightState.RESIDENT_DEMAND
    assert ocm.resident_tile_count(*matrix) == 3
    assert ocm.resident_fraction(*matrix) == 0.75
    assert ocm.first_tile_ready_cycle(*matrix) == 10
    assert ocm.contiguous_ready_prefix(*matrix) == 2
