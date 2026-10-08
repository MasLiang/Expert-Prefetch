"""Continuous multi-layer event simulator with future-target prefetch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import Random

from .event_queue import EventQueue
from .events import Event, EventType
from .hardware.aie import AIETask
from .hardware.dma import DMARequest
from .hardware.nonlinear import NonlinearTask
from .hardware.ocm import WeightState
from .hardware.resource_manager import ResourceManager
from .prefetch_policy import FixedDepthPolicy, PrefetchPolicy
from .schema import (
    ExpertExecutionRecord, ExpertTask, HardwareConfig, InterLayerConfig, LayerSequence,
    LayerTimingRecord, ModelConfig, PredictionEvent, PredictionMetrics, PredictorOutput, RoutingEvent,
    SequenceSimulationResult, SimulationConfig, SimulationResult, TileConfig,
    TileExecutionRecord, ValidationError, WeightTile,
)
from .scheduler import ReadyFirstScheduler, SimulatorState
from .sequence_predictor import SequencePredictor
from .timeline import TimelineEntry
from .tiling import ExpertTileSet, FixedByteTilingPolicy, FixedTileCountPolicy, WeightTilingPolicy
from .work_model import ceil_div, nonlinear_elements, router_ops


class SequenceScenario(str, Enum):
    """Supported continuous sequence scenarios."""

    EXACT = "sequence_exact"
    SYNTHETIC = "synthetic_cross_layer"
    ORACLE_SET = "cross_layer_oracle_set"
    ORACLE_OPTIMAL = "cross_layer_oracle_optimal"
    IDEAL = "sequence_ideal"


@dataclass
class SequenceSimulationOutput:
    """Continuous result paired with its one global timeline."""

    result: SequenceSimulationResult
    timeline: list[TimelineEntry]


@dataclass
class _LayerState:
    event: RoutingEvent
    timing: LayerTimingRecord
    tasks: list[ExpertTask]
    tile_sets: dict[int, ExpertTileSet]
    tile_by_key: dict[tuple[int, int, str, int], WeightTile]
    pending: list[ExpertTask]
    current_task: ExpertTask | None = None
    current_record: ExpertExecutionRecord | None = None
    current_phase: str | None = None
    current_tile_id: int = 0
    expert_records: list[ExpertExecutionRecord] = field(default_factory=list)
    memory_wait_start_cycle: int | None = None
    memory_stall_cycles: int = 0
    first_tile_stall_cycles: int = 0
    inter_tile_stall_cycles: int = 0
    gate_up_tile_stall_cycles: int = 0
    down_tile_stall_cycles: int = 0
    router_payload_cycles: int = 0
    expert_payload_cycles: int = 0
    aie_control_cycles: int = 0
    resolved: bool = False
    completed: bool = False


@dataclass
class _InterLayerRuntime:
    source_layer_id: int
    target_layer_id: int
    fixed_done_cycle: int
    background_request: DMARequest | None
    background_done: bool
    target_started: bool = False


@dataclass
class _SequenceState:
    sequence: LayerSequence
    scenario: SequenceScenario
    resources: ResourceManager
    queue: EventQueue
    layers: dict[int, _LayerState]
    layer_order: tuple[int, ...]
    predictor: SequencePredictor | None
    prefetch_policy: PrefetchPolicy
    rng: Random
    inter_layer: InterLayerConfig
    simulation: SimulationConfig
    current_cycle: int = 0
    active_layer_id: int | None = None
    aie_active: bool = False
    nonlinear_active: bool = False
    requests: dict[tuple[object, ...], DMARequest] = field(default_factory=dict)
    dma_timeline_entries: dict[int, TimelineEntry] = field(default_factory=dict)
    timeline: list[TimelineEntry] = field(default_factory=list)
    prediction_events: list[PredictionEvent] = field(default_factory=list)
    triggered_predictions: set[tuple[int, int]] = field(default_factory=set)
    inter_runtime: dict[tuple[int, int], _InterLayerRuntime] = field(default_factory=dict)
    blocked_active_future_ids: set[int] = field(default_factory=set)
    demand_blocked_by_future_cycles: int = 0
    finish_cycle: int = 0


class SequenceSimulator:
    """Run a same-token sequence on one persistent abstract machine."""

    def __init__(self, model: ModelConfig, hardware: HardwareConfig) -> None:
        self.model = model
        self.hardware = hardware

    def simulate_layer_sequence(
        self,
        layer_sequence: LayerSequence,
        scenario: SequenceScenario | str,
        *,
        simulation: SimulationConfig | None = None,
        tile_config: TileConfig | None = None,
        inter_layer_config: InterLayerConfig | None = None,
        predictor: SequencePredictor | None = None,
        prefetch_policy: PrefetchPolicy | None = None,
        tiling_policy: WeightTilingPolicy | None = None,
        seed: int = 0,
    ) -> SequenceSimulationOutput:
        """Simulate all supplied layers using one queue and one resource manager."""
        layer_sequence.validate(self.model)
        selected = SequenceScenario(scenario)
        if selected in {SequenceScenario.SYNTHETIC, SequenceScenario.ORACLE_SET, SequenceScenario.ORACLE_OPTIMAL} and predictor is None:
            raise ValidationError(f"scenario {selected.value} requires a sequence predictor")
        simulation_config = simulation or SimulationConfig("tile")
        resolved_tile = tile_config or TileConfig(65_536)
        policy = tiling_policy or (
            FixedByteTilingPolicy()
            if simulation_config.granularity == "tile"
            else FixedTileCountPolicy(1, 1)
        )
        layers = {
            event.layer_id: self._build_layer(event, policy, resolved_tile)
            for event in layer_sequence.routing_events
        }
        resources = ResourceManager(self.hardware)
        for layer in layers.values():
            for tile_set in layer.tile_sets.values():
                resources.ocm.register_tiled_matrix(
                    (layer.event.layer_id, tile_set.expert_task.expert_id, "gate_up"),
                    len(tile_set.gate_up_tiles),
                )
                resources.ocm.register_tiled_matrix(
                    (layer.event.layer_id, tile_set.expert_task.expert_id, "down"),
                    len(tile_set.down_tiles),
                )
        state = _SequenceState(
            layer_sequence, selected, resources, EventQueue(), layers,
            tuple(event.layer_id for event in layer_sequence.routing_events),
            predictor if selected not in {SequenceScenario.EXACT, SequenceScenario.IDEAL} else None,
            prefetch_policy or FixedDepthPolicy(1, 0), Random(seed),
            inter_layer_config or InterLayerConfig(), simulation_config,
        )
        state.queue.push(0, EventType.LAYER_START, {"layer_id": state.layer_order[0]})
        while state.queue:
            event = state.queue.pop()
            if event.cycle < state.current_cycle:
                raise RuntimeError("sequence event time moved backwards")
            state.current_cycle = event.cycle
            self._handle(state, event)
        if not all(layer.completed for layer in state.layers.values()):
            raise RuntimeError("sequence terminated before every layer completed")
        return self._build_output(state, resolved_tile)

    def _build_layer(
        self, event: RoutingEvent, policy: WeightTilingPolicy, tile_config: TileConfig
    ) -> _LayerState:
        tasks = [
            ExpertTask(event.layer_id, expert_id, event.expert_token_counts[expert_id], weight)
            for expert_id, weight in zip(event.true_expert_ids, event.true_gating_weights)
        ]
        task_by_id = {task.expert_id: task for task in tasks}
        tile_sets: dict[int, ExpertTileSet] = {}
        tile_by_key: dict[tuple[int, int, str, int], WeightTile] = {}
        for expert_id in range(self.model.num_experts):
            task = task_by_id.get(expert_id, ExpertTask(event.layer_id, expert_id, 1, 0.0))
            gate, down = policy.build_tiles(task, self.model, tile_config)
            tile_sets[expert_id] = ExpertTileSet(task, tuple(gate), tuple(down))
            for tile in gate + down:
                tile_by_key[(tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)] = tile
        timing = LayerTimingRecord(event.request_id, event.token_position, event.layer_id, 0)
        return _LayerState(event, timing, tasks, tile_sets, tile_by_key, list(tasks))

    def _handle(self, state: _SequenceState, event: Event) -> None:
        handlers = {
            EventType.LAYER_START: self._layer_start,
            EventType.ROUTER_START: self._router_start,
            EventType.AIE_OPERATION_START: self._aie_start,
            EventType.AIE_OPERATION_DONE: self._aie_done,
            EventType.ROUTER_DONE: self._router_done,
            EventType.TOPK_DONE: self._topk_done,
            EventType.DISPATCH_DONE: self._resolve,
            EventType.PREDICTION_TRIGGER: self._prediction_trigger,
            EventType.PREDICTOR_READY: self._prediction_ready,
            EventType.DMA_REQUEST: self._dma_request,
            EventType.DMA_START: self._dma_start,
            EventType.DMA_DONE: self._dma_done,
            EventType.SCHEDULER_INVOKE: self._scheduler,
            EventType.NONLINEAR_START: self._nonlinear_start,
            EventType.NONLINEAR_DONE: self._nonlinear_done,
            EventType.EXPERT_DONE: self._expert_done,
            EventType.LAYER_DONE: self._layer_done,
            EventType.INTER_LAYER_START: self._inter_layer_start,
            EventType.INTER_LAYER_FIXED_DONE: self._inter_layer_fixed_done,
        }
        try:
            handler = handlers[event.event_type]
        except KeyError as exc:
            raise RuntimeError(f"unsupported sequence event {event.event_type}") from exc
        handler(state, event)

    def _layer_start(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        state.active_layer_id = layer.event.layer_id
        layer.timing.layer_start_cycle = event.cycle
        self._milestone(state, layer, "source_layer_start", event.cycle)
        if state.predictor is not None and state.predictor.source_point == "custom_offset":
            config = getattr(state.predictor, "config", None)
            offset = int(getattr(config, "custom_offset_cycles_from_source_start", 0) or 0)
            state.queue.push(event.cycle + offset, EventType.PREDICTION_TRIGGER, {
                "layer_id": layer.event.layer_id, "feature_type": "custom_offset",
            })
        state.queue.push(event.cycle, EventType.ROUTER_START, {"layer_id": layer.event.layer_id})

    def _router_start(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        routed_tokens = ceil_div(
            sum(layer.event.expert_token_counts[expert] for expert in layer.event.true_expert_ids),
            self.model.top_k,
        )
        state.queue.push(event.cycle, EventType.AIE_OPERATION_START, {
            "layer_id": layer.event.layer_id, "kind": "router",
            "operation_count": router_ops(self.model, routed_tokens),
        })

    def _aie_start(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        kind = event.payload["kind"]
        tile: WeightTile | None = event.payload.get("tile")
        operation = "router" if kind == "router" else tile.matrix_type
        task = AIETask(
            operation, event.payload["operation_count"],
            tile_mode=state.simulation.granularity == "tile" and tile is not None,
            matrix_start=tile is None or tile.tile_id == 0,
        )
        start = state.resources.aie.earliest_start(task, event.cycle)
        if state.aie_active or start != event.cycle:
            state.queue.push(max(start, event.cycle + (1 if state.aie_active else 0)), EventType.AIE_OPERATION_START, event.payload)
            return
        before = len(state.resources.aie.intervals)
        end = state.resources.aie.reserve(task, start)
        interval = state.resources.aie.intervals[before]
        layer.aie_control_cycles += interval.control_cycles
        state.aie_active = True
        if kind == "router":
            layer.router_payload_cycles += interval.payload_cycles
            layer.timing.router_start_cycle = start
            name = f"router_l{layer.event.layer_id}"
            expert_id = None
        else:
            layer.expert_payload_cycles += interval.payload_cycles
            expert_id = tile.expert_id
            prefix = "gu" if tile.matrix_type == "gate_up" else "down"
            name = f"{prefix}_l{tile.layer_id}_e{tile.expert_id}_t{tile.tile_id}"
            record = layer.current_record
            if record is None:
                raise RuntimeError("tile compute has no expert record")
            if layer.timing.first_expert_start_cycle is None:
                layer.timing.first_expert_start_cycle = start
                self._milestone(state, layer, "source_expert_start", start)
            ready = state.resources.ocm.get_tile_ready_cycle(tile.key if hasattr(tile, "key") else (
                tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id
            ))
            records = record.gate_up_tile_records if tile.matrix_type == "gate_up" else record.down_tile_records
            records.append(TileExecutionRecord(
                tile.matrix_type, tile.tile_id, int(ready if ready is not None else start),
                event.payload.get("eligible_cycle", start), start, end,
                event.payload.get("starvation_cycles", 0),
                self._request_type(state, tile),
                state.resources.ocm.speculative_useful((tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)),
            ))
            if tile.matrix_type == "gate_up":
                record.gate_up_start_cycle = record.gate_up_start_cycle or start
                record.gate_up_done_cycle = end
            else:
                record.down_start_cycle = record.down_start_cycle or start
                record.down_done_cycle = end
            record.compute_cycles += interval.payload_cycles
        state.timeline.append(TimelineEntry(
            "AIE", name, start, end, layer.event.request_id, layer.event.layer_id,
            expert_id, tile_id=tile.tile_id if tile else None,
            num_tiles=tile.num_tiles if tile else None,
            tile_size_bytes=tile.size_bytes if tile else None,
            compute_ops=tile.compute_ops if tile else event.payload["operation_count"],
            starvation_cycles=event.payload.get("starvation_cycles"),
            ready_cycle=event.payload.get("ready_cycle"),
        ))
        state.queue.push(end, EventType.AIE_OPERATION_DONE, {
            "layer_id": layer.event.layer_id, "kind": kind, "tile": tile,
        })

    def _aie_done(self, state: _SequenceState, event: Event) -> None:
        state.aie_active = False
        layer = state.layers[event.payload["layer_id"]]
        if event.payload["kind"] == "router":
            state.queue.push(event.cycle, EventType.ROUTER_DONE, {"layer_id": layer.event.layer_id})
            return
        tile: WeightTile = event.payload["tile"]
        if tile.tile_id + 1 < tile.num_tiles:
            layer.current_tile_id = tile.tile_id + 1
            layer.current_phase = "waiting_gate" if tile.matrix_type == "gate_up" else "waiting_down"
            self._try_progress(state, layer, event.cycle)
        elif tile.matrix_type == "gate_up":
            layer.current_tile_id = 0
            layer.current_phase = "waiting_nonlinear"
            if layer.timing.gate_up_done_cycle is None:
                layer.timing.gate_up_done_cycle = event.cycle
                self._milestone(state, layer, "source_gate_up_done", event.cycle)
            state.queue.push(event.cycle, EventType.NONLINEAR_START, {"layer_id": layer.event.layer_id})
        else:
            state.queue.push(event.cycle, EventType.EXPERT_DONE, {"layer_id": layer.event.layer_id})
        state.queue.push(event.cycle, EventType.DMA_START)

    def _router_done(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        layer.timing.router_done_cycle = event.cycle
        self._milestone(state, layer, "source_router_done", event.cycle)
        end = event.cycle + self.hardware.pl_control.topk_cycles
        if end > event.cycle:
            state.timeline.append(TimelineEntry(
                "CONTROLLER", f"topk_l{layer.event.layer_id}", event.cycle, end,
                layer.event.request_id, layer.event.layer_id,
            ))
        state.queue.push(end, EventType.TOPK_DONE, {"layer_id": layer.event.layer_id})

    def _topk_done(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        layer.timing.topk_done_cycle = event.cycle
        end = event.cycle + self.hardware.pl_control.dispatch_cycles
        if end > event.cycle:
            state.timeline.append(TimelineEntry(
                "CONTROLLER", f"dispatch_l{layer.event.layer_id}", event.cycle, end,
                layer.event.request_id, layer.event.layer_id,
            ))
        state.queue.push(end, EventType.DISPATCH_DONE, {"layer_id": layer.event.layer_id})

    def _resolve(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        layer.resolved = True
        layer.timing.resolve_cycle = event.cycle
        state.timeline.append(TimelineEntry(
            "CONTROLLER", f"resolve_l{layer.event.layer_id}", event.cycle, event.cycle,
            layer.event.request_id, layer.event.layer_id,
        ))
        self._milestone(state, layer, "source_resolve", event.cycle)
        self._reconcile_target_prediction(state, layer, event.cycle)
        if state.scenario is SequenceScenario.IDEAL:
            for task in layer.tasks:
                for tile in layer.tile_sets[task.expert_id].gate_up_tiles + layer.tile_sets[task.expert_id].down_tiles:
                    key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
                    if state.resources.ocm.state(key) is WeightState.ABSENT:
                        state.resources.ocm.mark_tile_resident(key, "demand", event.cycle)
        else:
            for task in layer.tasks:
                for tile in layer.tile_sets[task.expert_id].gate_up_tiles + layer.tile_sets[task.expert_id].down_tiles:
                    key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
                    if state.resources.ocm.state(key) is WeightState.ABSENT:
                        self._issue_request(state, tile, "demand", event.cycle)
        state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE, {"layer_id": layer.event.layer_id})
        state.queue.push(event.cycle, EventType.DMA_START)

    def _milestone(self, state: _SequenceState, layer: _LayerState, feature: str, cycle: int) -> None:
        if state.predictor is None or state.predictor.source_point != feature:
            return
        state.queue.push(cycle, EventType.PREDICTION_TRIGGER, {
            "layer_id": layer.event.layer_id, "feature_type": feature,
        })

    def _prediction_trigger(self, state: _SequenceState, event: Event) -> None:
        if state.predictor is None:
            return
        source_id = event.payload["layer_id"]
        try:
            source_index = state.layer_order.index(source_id)
        except ValueError:
            return
        target_model_id = self.model.next_moe_layer(source_id)
        if target_model_id is None or target_model_id not in state.layers:
            return
        target_index = state.layer_order.index(target_model_id)
        if target_index <= source_index:
            return
        key = (source_id, target_model_id)
        if key in state.triggered_predictions:
            return
        state.triggered_predictions.add(key)
        source = state.layers[source_id].event
        target = state.layers[target_model_id].event
        prediction = state.predictor.create_prediction(
            state.sequence, self.model, source, target, event.cycle, state.rng
        )
        state.prediction_events.append(prediction)
        state.timeline.append(TimelineEntry(
            "PREDICTOR", f"predict_l{source_id}_to_l{target_model_id}",
            event.cycle, prediction.prediction_ready_cycle,
            source.request_id, source_id,
            source_layer_id=source_id, target_layer_id=target_model_id,
            source_token_position=source.token_position,
            target_token_position=target.token_position,
            prediction_stage=prediction.prediction_stage,
            speculative_scope="future_layer", prediction_id=prediction.prediction_id,
        ))
        state.queue.push(prediction.prediction_ready_cycle, EventType.PREDICTOR_READY, {
            "prediction": prediction,
        })

    def _prediction_ready(self, state: _SequenceState, event: Event) -> None:
        prediction: PredictionEvent = event.payload["prediction"]
        target = state.layers[prediction.target_layer_id]
        if target.resolved:
            return
        output = PredictorOutput(
            prediction.request_id, prediction.target_token_position, prediction.target_layer_id,
            candidates=prediction.candidates,
        )
        decisions = state.prefetch_policy.allocate(output, target.tile_sets, None)
        for decision in decisions:
            tiles = (
                target.tile_sets[decision.expert_id].gate_up_tiles
                if decision.matrix_type == "gate_up"
                else target.tile_sets[decision.expert_id].down_tiles
            )
            for tile_id in decision.tile_ids:
                tile = tiles[tile_id]
                key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
                if state.resources.ocm.state(key) is not WeightState.ABSENT:
                    continue
                request = self._issue_request(
                    state, tile, "speculative", event.cycle,
                    score=decision.score, rank=decision.candidate_rank,
                )
                request.speculative_scope = "future_layer"
                request.source_target_prediction_id = prediction.prediction_id
                request.target_request_id = prediction.request_id
                request.target_token_position = prediction.target_token_position
                request.target_layer_id = prediction.target_layer_id
                request.prediction_stage = prediction.prediction_stage
                request.prediction_ready_cycle = prediction.prediction_ready_cycle
                request.source_layer_id = prediction.source_layer_id
                request.source_token_position = prediction.source_token_position
        state.queue.push(event.cycle, EventType.DMA_START)

    def _issue_request(
        self, state: _SequenceState, tile: WeightTile, request_type: str,
        cycle: int, *, score: float = 0.0, rank: int | None = None,
    ) -> DMARequest:
        key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
        existing = state.requests.get(key)
        if existing is not None:
            return existing
        request = DMARequest(
            tile.layer_id, tile.expert_id, tile.matrix_type, tile.size_bytes,
            request_type, score, cycle, tile.tile_id, rank,
        )
        state.requests[key] = request
        state.queue.push(cycle, EventType.DMA_REQUEST, {
            "request": request, "request_type": request_type,
        })
        return request

    def _dma_request(self, state: _SequenceState, event: Event) -> None:
        request: DMARequest = event.payload["request"]
        if request.request_type != "background":
            state.resources.ocm.mark_tile_queued(request.key, request.request_type)
        state.resources.dma.enqueue(request)
        active = state.resources.dma.active_request
        if (
            request.request_type == "demand" and active is not None
            and active.request_type == "speculative"
            and active.speculative_scope == "future_layer"
            and id(active) not in state.blocked_active_future_ids
            and active.end_cycle is not None
        ):
            state.blocked_active_future_ids.add(id(active))
            blocked = max(0, active.end_cycle - event.cycle)
            state.demand_blocked_by_future_cycles += blocked
            state.resources.dma.demand_blocked_by_speculative_cycles += blocked
        state.queue.push(event.cycle, EventType.DMA_START)

    def _dma_start(self, state: _SequenceState, event: Event) -> None:
        dma = state.resources.dma
        if dma.active_request is not None:
            return
        pending = dma.pending_requests()
        if not pending:
            return

        def key(request: DMARequest) -> tuple[object, ...]:
            if request.request_type == "demand" or request.committed:
                priority = 0
            elif request.request_type == "background":
                priority = 1
            else:
                priority = 2
            return (
                priority,
                request.issue_cycle if priority < 2 else -request.score,
                request.candidate_rank if request.candidate_rank is not None else 2**30,
                0 if request.matrix_type == "gate_up" else 1,
                request.tile_id if request.tile_id is not None else -1,
                request.layer_id, request.expert_id,
            )

        request = min(pending, key=key)
        start = max(event.cycle, dma.available_cycle)
        if start != event.cycle:
            state.queue.push(start, EventType.DMA_START)
            return
        end = dma.start_request(request, start)
        if request.request_type != "background":
            state.resources.ocm.mark_tile_in_flight(request.key, request.request_type, end)
        if request.request_type == "demand":
            dma.demand_queue_wait_cycles += start - request.issue_cycle
        state.queue.push(end, EventType.DMA_DONE, {"request": request})

    def _dma_done(self, state: _SequenceState, event: Event) -> None:
        request: DMARequest = event.payload["request"]
        state.resources.dma.complete_request(request)
        if request.request_type != "background":
            state.resources.ocm.mark_tile_resident(request.key, request.request_type, event.cycle)
            if request.useful is True:
                state.resources.ocm.commit(request.key)
            elif request.useful is False:
                state.resources.ocm.mark_wasted(request.key)
        prediction_id = request.source_target_prediction_id or ""
        prefix = (
            "future_spec" if request.speculative_scope == "future_layer"
            else request.request_type
        )
        if request.request_type == "background":
            operation = f"background_dma_l{request.layer_id}"
        else:
            matrix = "gu" if request.matrix_type == "gate_up" else "down"
            operation = f"{prefix}_{matrix}_l{request.layer_id}_e{request.expert_id}_t{request.tile_id}"
        timeline_entry = TimelineEntry(
            "DMA", operation,
            request.start_cycle if request.start_cycle is not None else event.cycle,
            event.cycle,
            state.sequence.request_id, request.layer_id, request.expert_id,
            request.request_type, request.useful, request.committed, request.candidate_rank,
            request.tile_id, tile_size_bytes=request.size_bytes, ready_cycle=event.cycle,
            target_layer_id=request.target_layer_id,
            source_layer_id=request.source_layer_id,
            source_token_position=request.source_token_position,
            target_token_position=request.target_token_position,
            prediction_stage=request.prediction_stage or "",
            speculative_scope=request.speculative_scope,
            prediction_id=prediction_id,
        )
        state.timeline.append(timeline_entry)
        state.dma_timeline_entries[id(request)] = timeline_entry
        if request.request_type == "background":
            for runtime in state.inter_runtime.values():
                if runtime.background_request is request:
                    runtime.background_done = True
                    self._try_start_inter_target(state, runtime, event.cycle)
                    break
        active_id = state.active_layer_id
        if active_id is not None:
            layer = state.layers[active_id]
            self._try_progress(state, layer, event.cycle)
            state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE, {"layer_id": active_id})
        state.queue.push(event.cycle, EventType.DMA_START)

    def _reconcile_target_prediction(self, state: _SequenceState, layer: _LayerState, cycle: int) -> None:
        true_ids = set(layer.event.true_expert_ids)
        for request in sorted(
            [request for request in state.requests.values()
             if request.layer_id == layer.event.layer_id and request.request_type == "speculative"],
            key=lambda request: (request.expert_id, request.matrix_type, request.tile_id or 0),
        ):
            if request.expert_id in true_ids:
                if not request.committed:
                    request.committed = True
                    request.useful = True
                    state.resources.ocm.commit(request.key)
                    state.resources.dma.speculative_requests_committed += 1
                    if request.end_cycle is not None and state.resources.dma.active_request is not request:
                        state.resources.dma.useful_speculative_bytes += request.size_bytes
                action = "commit"
                useful = True
            else:
                useful = False
                if not request.started and state.resources.ocm.cancel_tile_if_queued(request.key):
                    request.canceled = True
                    request.useful = False
                    state.resources.dma.canceled_speculative_bytes += request.size_bytes
                    state.resources.dma.speculative_requests_canceled += 1
                else:
                    request.useful = False
                    state.resources.ocm.mark_wasted(request.key)
                    if request.end_cycle is not None and state.resources.dma.active_request is not request:
                        state.resources.dma.wasted_speculative_bytes += request.size_bytes
                action = "cancel"
            transfer_entry = state.dma_timeline_entries.get(id(request))
            if transfer_entry is not None:
                transfer_entry.useful = useful
                transfer_entry.committed = request.committed
            matrix = "gu" if request.matrix_type == "gate_up" else "down"
            state.timeline.append(TimelineEntry(
                "CONTROLLER", f"{action}_l{layer.event.layer_id}_e{request.expert_id}_{matrix}_t{request.tile_id}",
                cycle, cycle + (
                    self.hardware.pl_control.commit_cycles if useful
                    else self.hardware.pl_control.cancel_cycles
                ), layer.event.request_id, layer.event.layer_id, request.expert_id,
                "speculative", useful, request.committed, request.candidate_rank,
                request.tile_id, source_layer_id=request.source_layer_id,
                target_layer_id=layer.event.layer_id,
                source_token_position=request.source_token_position,
                target_token_position=layer.event.token_position,
                prediction_stage=request.prediction_stage or "",
                speculative_scope=request.speculative_scope,
                prediction_id=request.source_target_prediction_id or "",
            ))

    def _scheduler(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        if not layer.resolved or layer.completed or layer.current_task is not None:
            return
        if not layer.pending:
            done = event.cycle
            layer.timing.expert_phase_done_cycle = done
            self._milestone(state, layer, "source_expert_done", done)
            finish = done + self.hardware.pl_control.combine_cycles
            if finish > done:
                state.timeline.append(TimelineEntry(
                    "CONTROLLER", f"combine_l{layer.event.layer_id}", done, finish,
                    layer.event.request_id, layer.event.layer_id,
                ))
            state.queue.push(finish, EventType.LAYER_DONE, {"layer_id": layer.event.layer_id})
            return
        ready_cycles = {
            task.expert_id: state.resources.ocm.get_tile_ready_cycle(
                (task.layer_id, task.expert_id, "gate_up", 0)
            ) for task in layer.pending
        }
        task = ReadyFirstScheduler().choose_next(
            layer.pending, SimulatorState(ready_cycles, state.resources.ocm, layer.event.layer_id),
            event.cycle,
        )
        if task is None:
            raise RuntimeError("ready-first scheduler returned no pending expert")
        layer.current_task = task
        layer.current_record = ExpertExecutionRecord(task.expert_id, token_count=task.token_count)
        layer.current_phase = "waiting_gate"
        layer.current_tile_id = 0
        layer.memory_wait_start_cycle = event.cycle
        self._try_progress(state, layer, event.cycle)

    def _try_progress(self, state: _SequenceState, layer: _LayerState, cycle: int) -> None:
        if layer.current_task is None or layer.current_record is None:
            return
        if layer.current_phase not in {"waiting_gate", "waiting_down"}:
            return
        matrix = "gate_up" if layer.current_phase == "waiting_gate" else "down"
        tiles = (
            layer.tile_sets[layer.current_task.expert_id].gate_up_tiles
            if matrix == "gate_up" else layer.tile_sets[layer.current_task.expert_id].down_tiles
        )
        tile = tiles[layer.current_tile_id]
        key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
        if not state.resources.ocm.is_resident(key, cycle):
            if layer.memory_wait_start_cycle is None:
                layer.memory_wait_start_cycle = cycle
            return
        eligible = layer.memory_wait_start_cycle if layer.memory_wait_start_cycle is not None else cycle
        stall = max(0, cycle - eligible)
        if stall:
            layer.memory_stall_cycles += stall
            layer.current_record.memory_stall_cycles += stall
            if tile.tile_id == 0:
                layer.first_tile_stall_cycles += stall
                layer.current_record.first_tile_stall_cycles += stall
            else:
                layer.inter_tile_stall_cycles += stall
                layer.current_record.inter_tile_stall_cycles += stall
            if matrix == "gate_up":
                layer.gate_up_tile_stall_cycles += stall
            else:
                layer.down_tile_stall_cycles += stall
            state.timeline.append(TimelineEntry(
                "AIE", f"memory_stall_l{layer.event.layer_id}_e{tile.expert_id}_{matrix}_t{tile.tile_id}",
                eligible, cycle, layer.event.request_id, layer.event.layer_id,
                tile.expert_id, starvation_cycles=stall,
            ))
        layer.memory_wait_start_cycle = None
        ready = state.resources.ocm.get_tile_ready_cycle(key)
        if tile.tile_id == 0:
            if matrix == "gate_up":
                layer.current_record.gate_up_weight_ready_cycle = ready
            else:
                layer.current_record.down_weight_ready_cycle = ready
        layer.current_phase = matrix
        state.queue.push(cycle, EventType.AIE_OPERATION_START, {
            "layer_id": layer.event.layer_id, "kind": "tile", "tile": tile,
            "operation_count": tile.compute_ops, "eligible_cycle": eligible,
            "starvation_cycles": stall, "ready_cycle": ready,
        })

    def _nonlinear_start(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        if layer.current_task is None or layer.current_record is None:
            raise RuntimeError("nonlinear start without expert")
        task = NonlinearTask(nonlinear_elements(self.model, layer.current_task.token_count))
        start = state.resources.nonlinear.earliest_start(task, event.cycle)
        end = state.resources.nonlinear.reserve(task, start)
        state.nonlinear_active = True
        layer.current_record.nonlinear_start_cycle = start
        layer.current_record.nonlinear_done_cycle = end
        state.timeline.append(TimelineEntry(
            "NONLINEAR", f"nonlinear_l{layer.event.layer_id}_e{layer.current_task.expert_id}",
            start, end, layer.event.request_id, layer.event.layer_id, layer.current_task.expert_id,
        ))
        state.queue.push(end, EventType.NONLINEAR_DONE, {"layer_id": layer.event.layer_id})

    def _nonlinear_done(self, state: _SequenceState, event: Event) -> None:
        state.nonlinear_active = False
        layer = state.layers[event.payload["layer_id"]]
        layer.current_phase = "waiting_down"
        layer.current_tile_id = 0
        layer.memory_wait_start_cycle = event.cycle
        self._try_progress(state, layer, event.cycle)

    def _expert_done(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        if layer.current_task is None or layer.current_record is None:
            raise RuntimeError("expert done without current expert")
        layer.pending.remove(layer.current_task)
        layer.expert_records.append(layer.current_record)
        layer.current_task = None
        layer.current_record = None
        layer.current_phase = None
        state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE, {"layer_id": layer.event.layer_id})

    def _layer_done(self, state: _SequenceState, event: Event) -> None:
        layer = state.layers[event.payload["layer_id"]]
        if layer.completed:
            return
        layer.completed = True
        layer.timing.layer_done_cycle = event.cycle
        current_index = state.layer_order.index(layer.event.layer_id)
        if current_index + 1 < len(state.layer_order):
            target_id = state.layer_order[current_index + 1]
            state.queue.push(event.cycle, EventType.INTER_LAYER_START, {
                "source_layer_id": layer.event.layer_id, "target_layer_id": target_id,
            })
        else:
            state.finish_cycle = event.cycle

    def _inter_layer_start(self, state: _SequenceState, event: Event) -> None:
        source_id = event.payload["source_layer_id"]
        target_id = event.payload["target_layer_id"]
        segment = state.inter_layer.segment(source_id, target_id)
        start = event.cycle
        aie_start = max(start, state.resources.aie.available_cycle)
        aie_end = (
            state.resources.aie.reserve_fixed(segment.aie_cycles, aie_start)
            if segment.aie_cycles else aie_start
        )
        nonlinear_start = max(start, state.resources.nonlinear.available_cycle)
        nonlinear_end = (
            state.resources.nonlinear.reserve_fixed(segment.nonlinear_cycles, nonlinear_start)
            if segment.nonlinear_cycles else nonlinear_start
        )
        controller_end = start + segment.controller_cycles
        if segment.aie_cycles:
            state.timeline.append(TimelineEntry(
                "AIE", f"inter_layer_l{source_id}_to_l{target_id}", aie_start, aie_end,
                state.sequence.request_id, source_id,
                source_layer_id=source_id, target_layer_id=target_id,
            ))
        if segment.nonlinear_cycles:
            state.timeline.append(TimelineEntry(
                "NONLINEAR", f"inter_layer_l{source_id}_to_l{target_id}",
                nonlinear_start, nonlinear_end, state.sequence.request_id, source_id,
                source_layer_id=source_id, target_layer_id=target_id,
            ))
        if segment.controller_cycles:
            state.timeline.append(TimelineEntry(
                "CONTROLLER", f"inter_layer_l{source_id}_to_l{target_id}",
                start, controller_end, state.sequence.request_id, source_id,
                source_layer_id=source_id, target_layer_id=target_id,
            ))
        background: DMARequest | None = None
        if segment.background_dma_bytes:
            background = DMARequest(
                target_id, -1, "gate_up", segment.background_dma_bytes,
                "background", 0.0, start,
            )
            state.queue.push(start, EventType.DMA_REQUEST, {
                "request": background, "request_type": "background",
            })
        fixed_done = max(aie_end, nonlinear_end, controller_end)
        runtime = _InterLayerRuntime(
            source_id, target_id, fixed_done, background,
            background is None,
        )
        state.inter_runtime[(source_id, target_id)] = runtime
        state.queue.push(fixed_done, EventType.INTER_LAYER_FIXED_DONE, {
            "source_layer_id": source_id, "target_layer_id": target_id,
        })

    def _inter_layer_fixed_done(self, state: _SequenceState, event: Event) -> None:
        runtime = state.inter_runtime[(event.payload["source_layer_id"], event.payload["target_layer_id"])]
        self._try_start_inter_target(state, runtime, event.cycle)

    def _try_start_inter_target(
        self, state: _SequenceState, runtime: _InterLayerRuntime, cycle: int
    ) -> None:
        if runtime.target_started or not runtime.background_done or cycle < runtime.fixed_done_cycle:
            return
        runtime.target_started = True
        state.queue.push(cycle, EventType.LAYER_START, {"layer_id": runtime.target_layer_id})

    def _request_type(self, state: _SequenceState, tile: WeightTile) -> str | None:
        request = state.requests.get((tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id))
        return request.request_type if request else None

    def _build_output(self, state: _SequenceState, tile_config: TileConfig) -> SequenceSimulationOutput:
        layer_results = [self._layer_result(state, state.layers[layer_id], tile_config) for layer_id in state.layer_order]
        metrics = [self._prediction_metrics(state, prediction) for prediction in state.prediction_events]
        dma = state.resources.dma
        future = [request for request in state.requests.values() if request.speculative_scope == "future_layer"]
        transferred = [request for request in future if request.end_cycle is not None]
        last = layer_results[-1]
        result = SequenceSimulationResult(
            state.scenario.value, state.sequence.request_id, state.sequence.token_position,
            0, state.finish_cycle, state.finish_cycle,
            layer_results, [state.layers[layer_id].timing for layer_id in state.layer_order],
            state.prediction_events, metrics,
            dma.demand_bytes, dma.speculative_bytes, dma.useful_speculative_bytes,
            dma.wasted_speculative_bytes, dma.canceled_speculative_bytes,
            sum(layer.memory_stall_cycles for layer in state.layers.values()),
            state.demand_blocked_by_future_cycles,
            sum(request.size_bytes for request in transferred),
            sum(request.size_bytes for request in transferred if request.useful is True),
            sum(request.size_bytes for request in transferred if request.useful is False),
            sum(request.size_bytes for request in future if request.canceled),
            last.total_cycles, last.aie_memory_stall_cycles,
            sum(metric.ready_at_source_layer_end for metric in metrics) / len(metrics) if metrics else None,
            sum(metric.ready_at_target_layer_start for metric in metrics) / len(metrics) if metrics else None,
            sum(metric.ready_at_target_resolve for metric in metrics) / len(metrics) if metrics else None,
        )
        timeline = sorted(state.timeline, key=lambda item: (
            item.start_cycle, item.end_cycle, item.resource, item.operation,
            -1 if item.expert_id is None else item.expert_id,
        ))
        return SequenceSimulationOutput(result, timeline)

    def _layer_result(
        self, state: _SequenceState, layer: _LayerState, tile_config: TileConfig
    ) -> SimulationResult:
        start, done = layer.timing.layer_start_cycle, layer.timing.layer_done_cycle
        requests = [request for request in state.requests.values() if request.layer_id == layer.event.layer_id]
        transferred = [request for request in requests if request.end_cycle is not None]
        true_tile_count = sum(
            len(layer.tile_sets[task.expert_id].gate_up_tiles) + len(layer.tile_sets[task.expert_id].down_tiles)
            for task in layer.tasks
        )
        resident_resolve = sum(
            state.resources.ocm.is_resident(
                (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id), layer.timing.resolve_cycle
            )
            for task in layer.tasks
            for tile in layer.tile_sets[task.expert_id].gate_up_tiles + layer.tile_sets[task.expert_id].down_tiles
        )
        first_ready = sum(
            state.resources.ocm.is_resident(
                (task.layer_id, task.expert_id, "gate_up", 0), layer.timing.resolve_cycle
            ) for task in layer.tasks
        )
        candidates = {
            candidate.expert_id
            for prediction in state.prediction_events if prediction.target_layer_id == layer.event.layer_id
            for candidate in prediction.candidates
        }
        predictive = bool(candidates)
        true_ids = set(layer.event.true_expert_ids)
        duration = done - start
        idle = max(0, duration - layer.router_payload_cycles - layer.expert_payload_cycles - layer.aie_control_cycles)
        return SimulationResult(
            state.scenario.value, layer.event.request_id, layer.event.token_position, layer.event.layer_id,
            start, layer.timing.router_start_cycle, layer.timing.router_done_cycle,
            layer.timing.topk_done_cycle, layer.timing.resolve_cycle, layer.timing.resolve_cycle,
            done, duration, layer.router_payload_cycles, layer.expert_payload_cycles,
            layer.memory_stall_cycles, idle,
            sum(entry.end_cycle - entry.start_cycle for entry in state.timeline
                if entry.resource == "NONLINEAR" and entry.layer_id == layer.event.layer_id
                and entry.operation.startswith("nonlinear_")),
            sum((request.end_cycle or 0) - (request.start_cycle or 0) for request in transferred),
            sum(request.size_bytes for request in transferred if request.request_type in {"demand", "background"}),
            sum(request.size_bytes for request in transferred if request.request_type == "speculative"),
            sum(request.size_bytes for request in transferred if request.request_type == "speculative" and request.useful is True),
            sum(request.size_bytes for request in transferred if request.request_type == "speculative" and request.useful is False),
            sum(request.size_bytes for request in requests if request.canceled),
            len(candidates & true_ids) / len(true_ids) if predictive else None,
            bool(candidates & true_ids) if predictive else None,
            true_ids <= candidates if predictive else None,
            first_ready / len(true_ids) if predictive else None,
            first_ready / len(true_ids), layer.expert_records,
            aie_control_transition_cycles=layer.aie_control_cycles,
            demand_queue_wait_cycles=sum(
                (request.start_cycle or request.issue_cycle) - request.issue_cycle
                for request in requests if request.request_type == "demand"
            ),
            demand_blocked_by_speculative_cycles=state.demand_blocked_by_future_cycles,
            tile_mode=state.simulation.granularity == "tile",
            tile_size_bytes=tile_config.tile_size_bytes,
            total_tiles_required=true_tile_count,
            total_tiles_transferred=len(transferred),
            demand_tiles=sum(request.request_type == "demand" for request in transferred),
            speculative_tiles=sum(request.request_type == "speculative" for request in transferred),
            useful_speculative_tiles=sum(request.request_type == "speculative" and request.useful is True for request in transferred),
            wasted_speculative_tiles=sum(request.request_type == "speculative" and request.useful is False for request in transferred),
            canceled_speculative_tiles=sum(request.canceled for request in requests),
            true_tiles_resident_at_resolve=resident_resolve,
            true_tile_resident_fraction_at_resolve=resident_resolve / true_tile_count if true_tile_count else 0.0,
            true_experts_first_tile_ready_at_resolve=first_ready,
            aie_first_tile_stall_cycles=layer.first_tile_stall_cycles,
            aie_inter_tile_stall_cycles=layer.inter_tile_stall_cycles,
            aie_gate_up_tile_stall_cycles=layer.gate_up_tile_stall_cycles,
            aie_down_tile_stall_cycles=layer.down_tile_stall_cycles,
        )

    def _prediction_metrics(self, state: _SequenceState, prediction: PredictionEvent) -> PredictionMetrics:
        source = state.layers[prediction.source_layer_id]
        target = state.layers[prediction.target_layer_id]
        true_ids = tuple(target.event.true_expert_ids)
        candidates = tuple(candidate.expert_id for candidate in prediction.candidates)
        candidate_set = set(candidates)

        def fraction(cycle: int) -> float:
            return len(state.resources.ocm.target_tiles_resident_at_cycle(
                target.event.layer_id, true_ids, cycle
            )) / len(true_ids)

        scheduled_ready = sum(
            state.resources.ocm.get_tile_ready_cycle((target.event.layer_id, record.expert_id, "gate_up", 0))
            is not None
            and int(state.resources.ocm.get_tile_ready_cycle((target.event.layer_id, record.expert_id, "gate_up", 0)) or 0)
            <= int(record.gate_up_start_cycle or 0)
            for record in target.expert_records
        ) / len(true_ids)
        ranks = tuple(candidates.index(expert) for expert in true_ids if expert in candidate_set)
        return PredictionMetrics(
            prediction.prediction_id, prediction.source_layer_id, prediction.target_layer_id,
            prediction.prediction_ready_cycle, target.timing.layer_start_cycle,
            target.timing.router_start_cycle, target.timing.resolve_cycle,
            target.timing.layer_start_cycle - prediction.prediction_ready_cycle,
            target.timing.router_start_cycle - prediction.prediction_ready_cycle,
            target.timing.resolve_cycle - prediction.prediction_ready_cycle,
            prediction.configured_candidate_count or len(candidates),
            prediction.configured_generation_mode or "unknown",
            prediction.configured_target_recall,
            prediction.sampled_hit_count if prediction.sampled_hit_count is not None else len(candidate_set & set(true_ids)),
            len(candidate_set & set(true_ids)) / len(true_ids),
            bool(candidate_set & set(true_ids)), set(true_ids) <= candidate_set, ranks,
            fraction(source.timing.layer_done_cycle), fraction(target.timing.layer_start_cycle),
            fraction(target.timing.router_start_cycle), fraction(target.timing.resolve_cycle),
            scheduled_ready,
        )


def simulate_layer_sequence(
    layer_sequence: LayerSequence,
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    simulation_config: SimulationConfig,
    inter_layer_config: InterLayerConfig,
    predictor: SequencePredictor | None,
    prefetch_policy: PrefetchPolicy,
    *,
    scenario: SequenceScenario | str = SequenceScenario.SYNTHETIC,
    tile_config: TileConfig | None = None,
    seed: int = 0,
) -> SequenceSimulationOutput:
    """Convenience functional API for continuous layer simulation."""
    return SequenceSimulator(model_config, hardware_config).simulate_layer_sequence(
        layer_sequence, scenario, simulation=simulation_config, tile_config=tile_config,
        inter_layer_config=inter_layer_config, predictor=predictor,
        prefetch_policy=prefetch_policy, seed=seed,
    )
