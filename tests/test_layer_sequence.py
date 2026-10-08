"""Milestone 5 continuous timeline and future-prefetch regressions."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from edge_moe_sim.prefetch_policy import FixedDepthPolicy
from edge_moe_sim.schema import (
    AIEConfig, DMAConfig, HardwareConfig, InterLayerConfig, LayerSequence, ModelConfig,
    NonlinearConfig, PLControlConfig, RoutingEvent, SimulationConfig, TileConfig,
    ValidationError,
)
from edge_moe_sim.sequence_predictor import (
    SyntheticCrossLayerPredictor, SyntheticPredictorConfig,
)
from edge_moe_sim.sequence_simulator import SequenceSimulator


def _model() -> ModelConfig:
    return ModelConfig("sequence", 4, 8, 4, 2, 8, 8, moe_layer_indices=(3, 7, 11))


def _hardware(bytes_per_cycle: int = 16) -> HardwareConfig:
    return HardwareConfig(
        1_000_000_000,
        AIEConfig(16, 16, 16, 2, 1), DMAConfig(bytes_per_cycle, 1),
        NonlinearConfig(8, 1), PLControlConfig(1, 1, 0, 0, 2),
    )


def _event(layer: int, experts: tuple[int, int]) -> RoutingEvent:
    return RoutingEvent(1, 9, layer, "decode", experts, (0.6, 0.4), {expert: 1 for expert in experts})


def _sequence(three: bool = False) -> LayerSequence:
    events = (_event(3, (0, 1)), _event(7, (1, 2)))
    if three:
        events += (_event(11, (2, 3)),)
    return LayerSequence(1, 9, events)


def _predictor(model: ModelConfig, hits: int, **changes: object) -> SyntheticCrossLayerPredictor:
    values = dict(
        candidate_count=2, hit_distribution={hits: 1.0},
        predictor_latency_cycles=0, ranking_mode="true_first", seed=7,
    )
    values.update(changes)
    return SyntheticCrossLayerPredictor(SyntheticPredictorConfig(**values), model)


def _run(hits: int = 2, **kwargs: object):
    model = _model()
    return SequenceSimulator(model, _hardware()).simulate_layer_sequence(
        _sequence(), "synthetic_cross_layer", simulation=SimulationConfig("tile"),
        tile_config=TileConfig(32), predictor=_predictor(model, hits),
        prefetch_policy=FixedDepthPolicy(1, 0), seed=7, **kwargs,
    )


def test_layer_sequence_global_time() -> None:
    output = _run()
    source, target = output.result.layer_timings
    assert target.layer_start_cycle >= source.layer_done_cycle
    assert target.router_start_cycle > 0
    assert output.result.total_cycles == target.layer_done_cycle


def test_ocm_persists_across_layers() -> None:
    metric = _run().result.prediction_metrics[0]
    assert metric.ready_at_source_layer_end == 1.0
    assert metric.ready_at_target_layer_start == 1.0
    assert metric.ready_at_target_resolve == 1.0


def test_prediction_trigger_source_router_done() -> None:
    output = _run()
    prediction = output.result.prediction_events[0]
    source = output.result.layer_timings[0]
    assert prediction.prediction_ready_cycle == source.router_done_cycle


def test_prediction_trigger_source_expert_done() -> None:
    model = _model()
    predictor = _predictor(model, 2, source_point="source_expert_done", predictor_latency_cycles=3)
    output = SequenceSimulator(model, _hardware()).simulate_layer_sequence(
        _sequence(), "synthetic_cross_layer", tile_config=TileConfig(32),
        predictor=predictor, prefetch_policy=FixedDepthPolicy(1, 0), seed=7,
    )
    prediction = output.result.prediction_events[0]
    assert prediction.prediction_ready_cycle == (
        output.result.layer_timings[0].expert_phase_done_cycle + 3
    )


def test_prediction_trigger_source_gate_up_done() -> None:
    model = _model()
    predictor = _predictor(model, 2, source_point="source_gate_up_done", predictor_latency_cycles=2)
    output = SequenceSimulator(model, _hardware()).simulate_layer_sequence(
        _sequence(), "synthetic_cross_layer", tile_config=TileConfig(32),
        predictor=predictor, prefetch_policy=FixedDepthPolicy(1, 0), seed=7,
    )
    assert output.result.prediction_events[0].prediction_ready_cycle == (
        output.result.layer_timings[0].gate_up_done_cycle + 2
    )


def test_target_commit_only_at_target_resolve() -> None:
    output = _run()
    source, target = output.result.layer_timings
    commits = [entry for entry in output.timeline if entry.operation.startswith("commit_l7")]
    assert commits
    assert all(entry.start_cycle == target.resolve_cycle for entry in commits)
    assert all(entry.start_cycle != source.resolve_cycle for entry in commits)


def test_future_active_burst_blocks_source_demand() -> None:
    model = _model()
    output = SequenceSimulator(model, _hardware(bytes_per_cycle=1)).simulate_layer_sequence(
        _sequence(), "synthetic_cross_layer", tile_config=TileConfig(64),
        predictor=_predictor(model, 2), prefetch_policy=FixedDepthPolicy(1, 0), seed=7,
    )
    assert output.result.total_demand_blocked_by_future_speculative_cycles > 0
    future = next(entry for entry in output.timeline if entry.operation.startswith("future_spec"))
    source_demand = next(
        entry for entry in output.timeline
        if entry.resource == "DMA" and entry.layer_id == 3 and entry.request_type == "demand"
    )
    assert source_demand.start_cycle >= future.end_cycle


def test_queued_future_speculative_does_not_starve_source_demand() -> None:
    model = _model()
    predictor = SyntheticCrossLayerPredictor(SyntheticPredictorConfig(
        candidate_count=4, hit_distribution={2: 1.0},
        predictor_latency_cycles=0, ranking_mode="true_first",
    ), model)
    output = SequenceSimulator(model, _hardware(bytes_per_cycle=1)).simulate_layer_sequence(
        _sequence(), "synthetic_cross_layer", tile_config=TileConfig(32),
        predictor=predictor, prefetch_policy=FixedDepthPolicy(1, 0), seed=7,
    )
    futures = sorted(
        (entry for entry in output.timeline if entry.operation.startswith("future_spec")),
        key=lambda entry: entry.start_cycle,
    )
    demands = sorted(
        (entry for entry in output.timeline if entry.resource == "DMA"
         and entry.layer_id == 3 and entry.request_type == "demand"),
        key=lambda entry: entry.start_cycle,
    )
    assert len(futures) >= 2 and demands
    assert demands[0].start_cycle <= futures[1].start_cycle


@pytest.mark.parametrize("hits,recall", [(0, 0.0), (1, 0.5), (2, 1.0)])
def test_cross_layer_hit_quality(hits: int, recall: float) -> None:
    output = _run(hits)
    metric = output.result.prediction_metrics[0]
    assert metric.static_recall == recall
    target = output.result.layer_results[-1]
    if hits < 2:
        assert target.demand_tiles > 0
    if hits == 0:
        assert output.result.useful_future_speculative_bytes == 0
        assert (
            output.result.wasted_future_speculative_bytes
            + output.result.canceled_future_speculative_bytes > 0
        )


def test_sequence_determinism() -> None:
    first, second = _run(), _run()
    assert asdict(first.result) == asdict(second.result)
    assert [asdict(entry) for entry in first.timeline] == [asdict(entry) for entry in second.timeline]


def test_moe_layer_mapping() -> None:
    model = _model()
    assert model.next_moe_layer(3) == 7
    assert model.next_moe_layer(7) == 11
    assert model.next_moe_layer(11) is None


def test_candidate_count_validation() -> None:
    model = _model()
    with pytest.raises(ValidationError, match="false"):
        SyntheticCrossLayerPredictor(
            SyntheticPredictorConfig(candidate_count=4, hit_distribution={0: 1.0}), model
        )


def test_inter_layer_prefetch_window() -> None:
    baseline = _run(inter_layer_config=InterLayerConfig(default_aie_cycles=0))
    longer = _run(inter_layer_config=InterLayerConfig(default_aie_cycles=100))
    assert longer.result.prediction_metrics[0].prediction_lead_to_resolve >= (
        baseline.result.prediction_metrics[0].prediction_lead_to_resolve
    )
    assert longer.result.layer_results[-1].aie_memory_stall_cycles <= (
        baseline.result.layer_results[-1].aie_memory_stall_cycles
    )


def test_inter_layer_background_dma_competition() -> None:
    no_background = _run(inter_layer_config=InterLayerConfig(default_aie_cycles=10))
    background = _run(inter_layer_config=InterLayerConfig(
        default_aie_cycles=10, default_background_dma_bytes=128
    ))
    assert background.result.layer_timings[-1].layer_start_cycle >= (
        no_background.result.layer_timings[-1].layer_start_cycle
    )


def test_sequence_bandwidth_and_aie_mutex() -> None:
    output = _run()
    hardware = _hardware()
    for entry in output.timeline:
        if entry.resource == "DMA" and entry.tile_size_bytes is not None:
            payload_cycles = entry.end_cycle - entry.start_cycle - hardware.dma.startup_cycles
            assert entry.tile_size_bytes <= hardware.dma.bytes_per_cycle * payload_cycles
    aie = sorted(
        (entry.start_cycle, entry.end_cycle) for entry in output.timeline
        if entry.resource == "AIE" and entry.end_cycle > entry.start_cycle
    )
    assert all(left[1] <= right[0] for left, right in zip(aie, aie[1:]))
