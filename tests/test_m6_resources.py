"""Multi-channel memory, AIE pool, and finite OCM capacity tests."""

from __future__ import annotations

from edge_moe_sim.hardware.aie_pool import AIEPool, AIEPoolTask
from edge_moe_sim.hardware.capacity_ocm import CapacityOnChipMemory
from edge_moe_sim.hardware.multi_channel_memory import MultiChannelMemory
from edge_moe_sim.hardware_profile import HardwareProfile


def _profile(channels: int, units: int = 8) -> HardwareProfile:
    return HardwareProfile(
        "resources", 1000, channels, 10_000, 1.0, 1, 1, 64,
        channels > 1, units, 4, 1.0, 0, 4096,
    )


def _drain(memory: MultiChannelMemory) -> int:
    cycle = 0
    while memory.pending and not all(request.canceled for request in memory.pending):
        started = memory.start_ready(cycle)
        if not started:
            active = {id(channel.active_request): channel.active_request for channel in memory.channels if channel.active_request}
            cycle = min(request.end_cycle for request in active.values())
            for request in list(active.values()):
                if request.end_cycle == cycle:
                    memory.complete(request, cycle)
        else:
            cycle = min(request.end_cycle for request in started)
            for request in started:
                if request.end_cycle == cycle:
                    memory.complete(request, cycle)
        if all(request.started for request in memory.pending) and not any(channel.active_request for channel in memory.channels):
            break
    return max((request.end_cycle or 0 for request in memory.pending), default=0)


def test_multi_channel_independent_requests_overlap() -> None:
    parallel = MultiChannelMemory(_profile(3), "round_robin_expert")
    serial = MultiChannelMemory(_profile(3), "single_channel")
    for system in (parallel, serial):
        for expert in range(3):
            system.enqueue(system.new_request(
                layer_id=0, expert_id=expert, matrix_type="gate_up", tile_ids=(0,),
                object_keys=((expert, 0),), transfer_bytes=30,
                request_type="demand", issue_cycle=0,
            ))
    assert _drain(parallel) < _drain(serial)
    assert all(channel.transferred_bytes == 30 for channel in parallel.channels)


def test_aie_pool_capacity_controls_concurrency() -> None:
    concurrent = AIEPool(_profile(1, 8))
    first = AIEPoolTask("gate_up", 160, 4, expert_id=0)
    second = AIEPoolTask("gate_up", 160, 4, expert_id=1)
    assert concurrent.reserve(first, 0)[0] == 0
    assert concurrent.reserve(second, 0)[0] == 0
    serial = AIEPool(_profile(1, 4))
    first_end = serial.reserve(first, 0)[1]
    assert serial.reserve(second, 0)[0] == first_end


def test_finite_ocm_evicts_false_then_low_priority_speculation() -> None:
    ocm = CapacityOnChipMemory(100)
    assert ocm.admit("high", 40, "speculative", 0.9, 0).admitted
    assert ocm.admit("false", 40, "speculative", 0.8, 0).admitted
    ocm.mark_false("false")
    admission = ocm.admit("demand", 50, "demand", 1.0, 1)
    assert admission.admitted
    assert admission.evicted_keys == ("false",)
    assert ocm.peak_ocm_bytes <= ocm.capacity_bytes


def test_adjacent_memory_tiles_coalesce_within_burst_limit() -> None:
    memory = MultiChannelMemory(_profile(1), "single_channel")
    first = memory.new_request(
        layer_id=0, expert_id=0, matrix_type="gate_up", tile_ids=(0,),
        object_keys=((0, 0),), transfer_bytes=16, request_type="demand", issue_cycle=0,
    )
    second = memory.new_request(
        layer_id=0, expert_id=0, matrix_type="gate_up", tile_ids=(1,),
        object_keys=((0, 1),), transfer_bytes=16, request_type="demand", issue_cycle=0,
    )
    assert memory.enqueue(first, coalesce_adjacent=True) is first
    assert memory.enqueue(second, coalesce_adjacent=True) is first
    assert first.tile_ids == (0, 1)
    assert first.transfer_bytes == 32
