"""Unified event-driven simulation core for five MoE memory scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .dma_scheduler import CriticalPathAwareScheduler, DMARequestScheduler, TileCriticalPathAwareScheduler
from .event_queue import EventQueue
from .events import Event, EventType
from .hardware.aie import AIETask
from .hardware.dma import DMARequest
from .hardware.nonlinear import NonlinearTask
from .hardware.ocm import WeightState
from .hardware.resource_manager import ResourceManager
from .predictor import NoPredictor, OraclePredictor, Predictor
from .prefetch_policy import PrefetchPolicy, policy_from_config
from .scheduler import ExpertScheduler, ReadyFirstScheduler, RouterOrderScheduler, SimulatorState
from .schema import (
    ExpertExecutionRecord, ExpertTask, HardwareConfig, ModelConfig, PredictorOutput,
    PrefetchConfig, RoutingEvent, SimulationConfig, SimulationResult, TileConfig,
    TileExecutionRecord, WeightTile,
)
from .timeline import TimelineEntry
from .trace import TraceSimulationResult, aggregate_results
from .tiling import ExpertTileSet, FixedByteTilingPolicy, WeightTilingPolicy
from .work_model import (
    ceil_div, down_ops, down_weight_bytes, gate_up_ops, gate_up_weight_bytes,
    nonlinear_elements, router_ops,
)


class Scenario(str, Enum):
    """Supported decode simulation scenarios."""

    DEMAND_SERIAL = "demand_serial"
    EXACT_PIPELINE = "exact_pipeline"
    PREDICTIVE = "predictive"
    ORACLE = "oracle"
    IDEAL = "ideal"


@dataclass
class SimulationOutput:
    """A per-event result paired with its deterministic timeline."""

    result: SimulationResult
    timeline: list[TimelineEntry]


@dataclass
class _RunState:
    event: RoutingEvent
    scenario: Scenario
    resources: ResourceManager
    queue: EventQueue
    tasks: list[ExpertTask]
    expert_scheduler: ExpertScheduler
    dma_scheduler: DMARequestScheduler
    prefetch: PrefetchConfig
    predictor_output: PredictorOutput | None = None
    current_cycle: int = 0
    router_start_cycle: int = 0
    router_done_cycle: int = 0
    topk_done_cycle: int = 0
    dispatch_done_cycle: int = 0
    layer_done_cycle: int = 0
    resolved: bool = False
    aie_active: bool = False
    nonlinear_active: bool = False
    current_task: ExpertTask | None = None
    current_phase: str | None = None
    current_record: ExpertExecutionRecord | None = None
    pending_tasks: list[ExpertTask] = field(default_factory=list)
    expert_records: list[ExpertExecutionRecord] = field(default_factory=list)
    dma_requests: dict[tuple[object, ...], DMARequest] = field(default_factory=dict)
    dma_timeline_entries: dict[int, TimelineEntry] = field(default_factory=dict)
    timeline: list[TimelineEntry] = field(default_factory=list)
    memory_wait_start_cycle: int | None = None
    memory_stall_cycles: int = 0
    dependency_wait_cycles: int = 0
    resolve_control_cycles: int = 0
    expert_schedule_ready_cycle: int = 0
    ready_recall_at_router_done: float | None = None
    ready_recall_at_resolve: float | None = None
    blocked_speculative_request_id: int | None = None
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    tile_config: TileConfig | None = None
    tile_sets: dict[int, ExpertTileSet] = field(default_factory=dict)
    tile_by_key: dict[tuple[int, int, str, int], WeightTile] = field(default_factory=dict)
    current_tile_id: int = 0
    first_tile_stall_cycles: int = 0
    inter_tile_stall_cycles: int = 0
    gate_up_tile_stall_cycles: int = 0
    down_tile_stall_cycles: int = 0
    true_tiles_resident_at_resolve: int = 0
    true_first_tiles_resident_at_resolve: int = 0
    prefetch_policy: PrefetchPolicy | None = None


class Simulator:
    """Run independent, deterministic, event-driven decode simulations."""

    def __init__(self, model: ModelConfig, hardware: HardwareConfig) -> None:
        self.model = model
        self.hardware = hardware

    def simulate(
        self,
        event: RoutingEvent,
        scenario: Scenario | str,
        *,
        predictor: Predictor | None = None,
        expert_scheduler: ExpertScheduler | None = None,
        scheduler: ExpertScheduler | None = None,
        dma_scheduler: DMARequestScheduler | None = None,
        prefetch: PrefetchConfig | None = None,
        simulation: SimulationConfig | None = None,
        tile_config: TileConfig | None = None,
        tiling_policy: WeightTilingPolicy | None = None,
        prefetch_policy: PrefetchPolicy | None = None,
    ) -> SimulationOutput:
        """Simulate one routing event with isolated resources and OCM state."""
        event.validate(self.model)
        try:
            selected = Scenario(scenario)
        except ValueError as exc:
            raise ValueError(f"unsupported scenario {scenario!r}") from exc
        chosen_expert_scheduler = expert_scheduler or scheduler
        if chosen_expert_scheduler is None:
            chosen_expert_scheduler = (
                RouterOrderScheduler() if selected is Scenario.DEMAND_SERIAL else ReadyFirstScheduler()
            )
        chosen_predictor: Predictor
        if selected is Scenario.ORACLE:
            chosen_predictor = predictor or OraclePredictor()
        elif selected is Scenario.PREDICTIVE:
            chosen_predictor = predictor or NoPredictor()
        else:
            chosen_predictor = NoPredictor()
        simulation_config = simulation or SimulationConfig()
        resolved_tile_config = tile_config or TileConfig(65_536)
        tasks = self._tasks(event)
        tile_sets: dict[int, ExpertTileSet] = {}
        tile_by_key: dict[tuple[int, int, str, int], WeightTile] = {}
        if simulation_config.granularity == "tile":
            policy = tiling_policy or FixedByteTilingPolicy()
            true_tasks = {task.expert_id: task for task in tasks}
            for expert_id in range(self.model.num_experts):
                geometry_task = true_tasks.get(
                    expert_id, ExpertTask(event.layer_id, expert_id, 1, 0.0)
                )
                gate_tiles, down_tiles = policy.build_tiles(
                    geometry_task, self.model, resolved_tile_config
                )
                tile_set = ExpertTileSet(geometry_task, tuple(gate_tiles), tuple(down_tiles))
                tile_sets[expert_id] = tile_set
                for tile in gate_tiles + down_tiles:
                    tile_by_key[(tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)] = tile
        state = _RunState(
            event=event,
            scenario=selected,
            resources=ResourceManager(self.hardware),
            queue=EventQueue(),
            tasks=tasks,
            expert_scheduler=chosen_expert_scheduler,
            dma_scheduler=dma_scheduler or (
                TileCriticalPathAwareScheduler()
                if simulation_config.granularity == "tile"
                else CriticalPathAwareScheduler()
            ),
            prefetch=prefetch or PrefetchConfig(),
            simulation=simulation_config,
            tile_config=resolved_tile_config if simulation_config.granularity == "tile" else None,
            tile_sets=tile_sets,
            tile_by_key=tile_by_key,
            prefetch_policy=(prefetch_policy or policy_from_config(prefetch or PrefetchConfig()))
            if simulation_config.granularity == "tile" else None,
        )
        state.pending_tasks = list(state.tasks)
        if simulation_config.granularity == "tile":
            for tile_set in tile_sets.values():
                state.resources.ocm.register_tiled_matrix(
                    (event.layer_id, tile_set.expert_task.expert_id, "gate_up"), len(tile_set.gate_up_tiles)
                )
                state.resources.ocm.register_tiled_matrix(
                    (event.layer_id, tile_set.expert_task.expert_id, "down"), len(tile_set.down_tiles)
                )
        state.queue.push(0, EventType.LAYER_START, {"predictor": chosen_predictor})
        while state.queue:
            queued_event = state.queue.pop()
            state.current_cycle = queued_event.cycle
            self._handle_event(state, queued_event)
            if state.layer_done_cycle and not state.queue:
                break
        if state.layer_done_cycle == 0:
            raise RuntimeError("simulation terminated without LAYER_DONE")
        return self._build_output(state)

    def simulate_event(
        self, event: RoutingEvent, scenario: Scenario | str, **kwargs: Any
    ) -> SimulationOutput:
        """Backward-compatible explicit name for independent event simulation."""
        return self.simulate(event, scenario, **kwargs)

    def simulate_layer_sequence(
        self, layer_sequence: Any, scenario: Any, **kwargs: Any
    ) -> Any:
        """Run a continuous sequence on one persistent abstract machine."""
        from .sequence_simulator import SequenceSimulator

        return SequenceSimulator(self.model, self.hardware).simulate_layer_sequence(
            layer_sequence, scenario, **kwargs
        )

    def simulate_trace(
        self,
        events: Sequence[RoutingEvent],
        scenario: Scenario | str,
        *,
        predictor: Predictor | None = None,
        expert_scheduler: ExpertScheduler | None = None,
        dma_scheduler: DMARequestScheduler | None = None,
        prefetch: PrefetchConfig | None = None,
        simulation: SimulationConfig | None = None,
        tile_config: TileConfig | None = None,
        tiling_policy: WeightTilingPolicy | None = None,
        prefetch_policy: PrefetchPolicy | None = None,
    ) -> TraceSimulationResult:
        """Simulate each trace record independently and aggregate their metrics."""
        outputs = [
            self.simulate(
                event, scenario, predictor=predictor, expert_scheduler=expert_scheduler,
                dma_scheduler=dma_scheduler, prefetch=prefetch,
                simulation=simulation, tile_config=tile_config, tiling_policy=tiling_policy,
                prefetch_policy=prefetch_policy,
            ).result
            for event in events
        ]
        return aggregate_results(outputs)

    def _tasks(self, event: RoutingEvent) -> list[ExpertTask]:
        return [
            ExpertTask(event.layer_id, expert_id, event.expert_token_counts[expert_id], weight)
            for expert_id, weight in zip(event.true_expert_ids, event.true_gating_weights)
        ]

    def _handle_event(self, state: _RunState, event: Event) -> None:
        handlers = {
            EventType.LAYER_START: self._layer_start,
            EventType.ROUTER_START: self._router_start,
            EventType.ROUTER_DONE: self._router_done,
            EventType.TOPK_DONE: self._topk_done,
            EventType.DISPATCH_DONE: self._dispatch_done,
            EventType.PREDICTOR_READY: self._predictor_ready,
            EventType.DMA_REQUEST: self._dma_request,
            EventType.DMA_START: self._dma_start,
            EventType.DMA_DONE: self._dma_done,
            EventType.DMA_COMMIT: self._dma_commit,
            EventType.DMA_CANCEL: self._dma_cancel,
            EventType.AIE_OPERATION_START: self._aie_start,
            EventType.AIE_OPERATION_DONE: self._aie_done,
            EventType.NONLINEAR_START: self._nonlinear_start,
            EventType.NONLINEAR_DONE: self._nonlinear_done,
            EventType.EXPERT_DONE: self._expert_done,
            EventType.SCHEDULER_INVOKE: self._scheduler_invoke,
            EventType.LAYER_DONE: self._layer_done,
        }
        handler = handlers.get(event.event_type)
        if handler is None:
            raise RuntimeError(f"no handler for {event.event_type}")
        handler(state, event)

    def _layer_start(self, state: _RunState, event: Event) -> None:
        predictor: Predictor = event.payload["predictor"]
        state.predictor_output = predictor.predict(state.event, event.cycle)
        if state.predictor_output is not None:
            state.predictor_output.validate(self.model)
            state.queue.push(
                event.cycle + state.predictor_output.ready_offset_cycles,
                EventType.PREDICTOR_READY,
            )
        state.queue.push(event.cycle, EventType.ROUTER_START)

    def _router_start(self, state: _RunState, event: Event) -> None:
        routed_tokens = ceil_div(
            sum(state.event.expert_token_counts[expert_id] for expert_id in state.event.true_expert_ids),
            self.model.top_k,
        )
        state.queue.push(event.cycle, EventType.AIE_OPERATION_START, {
            "operation": "router",
            "operation_count": router_ops(self.model, routed_tokens),
        })

    def _aie_start(self, state: _RunState, event: Event) -> None:
        if state.aie_active:
            raise RuntimeError("attempted to start overlapping AIE operations")
        operation = event.payload["operation"]
        operation_count = event.payload["operation_count"]
        tile: WeightTile | None = event.payload.get("tile")
        task = AIETask(
            operation,
            operation_count,
            tile_mode=tile is not None,
            matrix_start=tile is None or tile.tile_id == 0,
        )
        start_cycle = state.resources.aie.earliest_start(task, event.cycle)
        if start_cycle != event.cycle:
            state.queue.push(start_cycle, EventType.AIE_OPERATION_START, event.payload)
            return
        before = len(state.resources.aie.intervals)
        end_cycle = state.resources.aie.reserve(task, start_cycle)
        interval = state.resources.aie.intervals[before]
        state.aie_active = True
        expert_id = state.current_task.expert_id if operation != "router" and state.current_task else None
        timeline_operation = operation
        if tile is not None:
            short_matrix = "gu" if tile.matrix_type == "gate_up" else "down"
            timeline_operation = f"aie_{short_matrix}_e{tile.expert_id}_t{tile.tile_id}"
        state.timeline.append(TimelineEntry(
            "aie", timeline_operation, start_cycle, end_cycle, state.event.request_id,
            state.event.layer_id, expert_id,
            tile_id=tile.tile_id if tile else None,
            num_tiles=tile.num_tiles if tile else None,
            tile_size_bytes=tile.size_bytes if tile else None,
            compute_ops=tile.compute_ops if tile else None,
            starvation_cycles=event.payload.get("starvation_cycles") if tile else None,
            ready_cycle=event.payload.get("ready_cycle") if tile else None,
        ))
        if operation == "router":
            state.router_start_cycle = start_cycle
        elif state.current_record is not None:
            if operation == "gate_up":
                if state.current_record.gate_up_start_cycle is None:
                    state.current_record.gate_up_start_cycle = start_cycle
                state.current_record.gate_up_done_cycle = end_cycle
            else:
                if state.current_record.down_start_cycle is None:
                    state.current_record.down_start_cycle = start_cycle
                state.current_record.down_done_cycle = end_cycle
            state.current_record.compute_cycles += interval.payload_cycles
            if tile is not None:
                request = state.dma_requests.get(
                    (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
                )
                tile_record = TileExecutionRecord(
                    matrix_type=tile.matrix_type,
                    tile_id=tile.tile_id,
                    weight_ready_cycle=event.payload["ready_cycle"],
                    compute_eligible_cycle=event.payload["compute_eligible_cycle"],
                    compute_start_cycle=start_cycle,
                    compute_done_cycle=end_cycle,
                    starvation_cycles=event.payload["starvation_cycles"],
                    dma_request_type=request.request_type if request else None,
                    speculative_useful=request.useful if request else None,
                )
                records = (
                    state.current_record.gate_up_tile_records
                    if operation == "gate_up"
                    else state.current_record.down_tile_records
                )
                records.append(tile_record)
        state.queue.push(end_cycle, EventType.AIE_OPERATION_DONE, {
            "operation": operation,
            "tile": tile,
        })

    def _aie_done(self, state: _RunState, event: Event) -> None:
        state.aie_active = False
        operation = event.payload["operation"]
        tile: WeightTile | None = event.payload.get("tile")
        if operation == "router":
            state.queue.push(event.cycle, EventType.ROUTER_DONE)
        elif tile is not None:
            if tile.tile_id + 1 < tile.num_tiles:
                state.current_tile_id = tile.tile_id + 1
                state.current_phase = "waiting_gate" if operation == "gate_up" else "waiting_down"
                self._try_progress_current(state, event.cycle)
                state.queue.push(event.cycle, EventType.DMA_START)
            elif operation == "gate_up":
                state.current_tile_id = 0
                state.current_phase = "waiting_nonlinear"
                state.queue.push(event.cycle, EventType.NONLINEAR_START)
                state.queue.push(event.cycle, EventType.DMA_START)
            else:
                state.queue.push(event.cycle, EventType.EXPERT_DONE)
        elif operation == "gate_up":
            state.current_phase = "waiting_nonlinear"
            state.queue.push(event.cycle, EventType.NONLINEAR_START)
            state.queue.push(event.cycle, EventType.DMA_START)
        else:
            state.queue.push(event.cycle, EventType.EXPERT_DONE)

    def _router_done(self, state: _RunState, event: Event) -> None:
        state.router_done_cycle = event.cycle
        true_ids = state.event.true_expert_ids
        ready = sum(
            state.resources.ocm.is_tile_resident((state.event.layer_id, expert_id, "gate_up", 0))
            if state.simulation.granularity == "tile"
            else state.resources.ocm.is_resident((state.event.layer_id, expert_id, "gate_up"))
            for expert_id in true_ids
        )
        state.ready_recall_at_router_done = ready / len(true_ids)
        topk_done = event.cycle + self.hardware.pl_control.topk_cycles
        if topk_done > event.cycle:
            state.timeline.append(TimelineEntry(
                "controller", "topk", event.cycle, topk_done,
                state.event.request_id, state.event.layer_id,
            ))
        state.queue.push(topk_done, EventType.TOPK_DONE)

    def _topk_done(self, state: _RunState, event: Event) -> None:
        state.topk_done_cycle = event.cycle
        dispatch_done = event.cycle + self.hardware.pl_control.dispatch_cycles
        if dispatch_done > event.cycle:
            state.timeline.append(TimelineEntry(
                "controller", "dispatch", event.cycle, dispatch_done,
                state.event.request_id, state.event.layer_id,
            ))
        state.queue.push(dispatch_done, EventType.DISPATCH_DONE)

    def _dispatch_done(self, state: _RunState, event: Event) -> None:
        state.dispatch_done_cycle = event.cycle
        state.resolved = True
        state.timeline.append(TimelineEntry(
            "controller", "router_resolve", event.cycle, event.cycle,
            state.event.request_id, state.event.layer_id,
        ))
        true_ids = set(state.event.true_expert_ids)
        if state.scenario is Scenario.IDEAL:
            if state.simulation.granularity == "tile":
                for task in state.tasks:
                    tile_set = state.tile_sets[task.expert_id]
                    for tile in tile_set.gate_up_tiles + tile_set.down_tiles:
                        state.resources.ocm.mark_tile_resident(
                            (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id),
                            "demand", event.cycle,
                        )
            else:
                for task in state.tasks:
                    for matrix_type in ("gate_up", "down"):
                        state.resources.ocm.mark_resident(
                            (task.layer_id, task.expert_id, matrix_type), "demand", event.cycle
                        )
        if state.simulation.granularity == "tile":
            ready = sum(
                state.resources.ocm.is_tile_resident((state.event.layer_id, expert_id, "gate_up", 0))
                for expert_id in true_ids
            )
            all_true_tiles = [
                tile for tile_set in state.tile_sets.values()
                for tile in tile_set.gate_up_tiles + tile_set.down_tiles
            ]
            state.true_tiles_resident_at_resolve = sum(
                state.resources.ocm.is_tile_resident(
                    (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
                )
                for tile in all_true_tiles
            )
            state.true_first_tiles_resident_at_resolve = ready
        else:
            ready = sum(
                state.resources.ocm.is_resident((state.event.layer_id, expert_id, "gate_up"))
                for expert_id in true_ids
            )
        state.ready_recall_at_resolve = ready / len(true_ids)
        for request in list(state.dma_requests.values()):
            if request.request_type != "speculative":
                continue
            event_type = EventType.DMA_COMMIT if request.expert_id in true_ids else EventType.DMA_CANCEL
            state.queue.push(event.cycle, event_type, {"request": request})
        has_commits = any(
            request.request_type == "speculative" and request.expert_id in true_ids
            for request in state.dma_requests.values()
        )
        has_cancels = any(
            request.request_type == "speculative" and request.expert_id not in true_ids
            for request in state.dma_requests.values()
        )
        state.resolve_control_cycles = max(
            self.hardware.pl_control.commit_cycles if has_commits else 0,
            self.hardware.pl_control.cancel_cycles if has_cancels else 0,
        )
        state.expert_schedule_ready_cycle = event.cycle + state.resolve_control_cycles
        if state.scenario is not Scenario.IDEAL and state.scenario is not Scenario.DEMAND_SERIAL:
            if state.simulation.granularity == "tile":
                for task in state.tasks:
                    tile_set = state.tile_sets[task.expert_id]
                    for tile in tile_set.gate_up_tiles + tile_set.down_tiles:
                        self._issue_tile_dma(state, tile, "demand", event.cycle)
            else:
                for task in state.tasks:
                    self._issue_dma(state, task, "gate_up", "demand", event.cycle)
                    self._issue_dma(state, task, "down", "demand", event.cycle)
        active = state.resources.dma.active_request
        if active is not None and active.request_type == "speculative" and active.expert_id not in true_ids:
            missing_true = any(
                request.request_type == "demand" and not request.started
                for request in state.dma_requests.values()
            )
            if missing_true and active.end_cycle is not None:
                state.resources.dma.demand_blocked_by_speculative_cycles += active.end_cycle - event.cycle
                state.blocked_speculative_request_id = id(active)
        state.queue.push(state.expert_schedule_ready_cycle, EventType.SCHEDULER_INVOKE)
        state.queue.push(event.cycle, EventType.DMA_START)

    def _predictor_ready(self, state: _RunState, event: Event) -> None:
        output = state.predictor_output
        if output is None:
            return
        state.timeline.append(TimelineEntry(
            "controller", "predictor_ready", event.cycle, event.cycle,
            state.event.request_id, state.event.layer_id,
        ))
        if state.resolved:
            return
        if state.simulation.granularity == "tile":
            if state.prefetch_policy is None:
                raise RuntimeError("tile mode has no prefetch policy")
            limited_output = PredictorOutput(
                output.request_id,
                output.token_position,
                output.layer_id,
                ready_offset_cycles=output.ready_offset_cycles,
                candidates=output.candidates[:state.prefetch.max_candidates],
            )
            decisions = state.prefetch_policy.allocate(
                limited_output, state.tile_sets, state.prefetch.byte_budget
            )
            for decision in decisions:
                tile_set = state.tile_sets[decision.expert_id]
                tiles = (
                    tile_set.gate_up_tiles
                    if decision.matrix_type == "gate_up"
                    else tile_set.down_tiles
                )
                for tile_id in decision.tile_ids:
                    self._issue_tile_dma(
                        state, tiles[tile_id], "speculative", event.cycle,
                        decision.score, decision.candidate_rank,
                    )
            state.queue.push(event.cycle, EventType.DMA_START)
            return
        candidates = list(zip(output.candidate_ids, output.candidate_scores))[:state.prefetch.max_candidates]
        for rank, (expert_id, score) in enumerate(candidates):
            pseudo_task = ExpertTask(state.event.layer_id, expert_id, 0, score)
            self._issue_dma(state, pseudo_task, "gate_up", "speculative", event.cycle, score, rank)
            if state.prefetch.scope == "full_expert":
                self._issue_dma(state, pseudo_task, "down", "speculative", event.cycle, score, rank)
        state.queue.push(event.cycle, EventType.DMA_START)

    def _issue_dma(
        self,
        state: _RunState,
        task: ExpertTask,
        matrix_type: str,
        request_type: str,
        issue_cycle: int,
        score: float = 0.0,
        candidate_rank: int | None = None,
    ) -> DMARequest | None:
        key = (task.layer_id, task.expert_id, matrix_type)
        if key in state.dma_requests or state.resources.ocm.get_state(key) is not WeightState.ABSENT:
            return None
        size_bytes = gate_up_weight_bytes(self.model) if matrix_type == "gate_up" else down_weight_bytes(self.model)
        request = DMARequest(
            layer_id=task.layer_id,
            expert_id=task.expert_id,
            matrix_type=matrix_type,
            size_bytes=size_bytes,
            request_type=request_type,
            score=score,
            issue_cycle=issue_cycle,
            candidate_rank=candidate_rank,
        )
        state.dma_requests[key] = request
        state.queue.push(issue_cycle, EventType.DMA_REQUEST, {
            "request": request,
            "request_type": request_type,
        })
        return request

    def _issue_tile_dma(
        self,
        state: _RunState,
        tile: WeightTile,
        request_type: str,
        issue_cycle: int,
        score: float = 0.0,
        candidate_rank: int | None = None,
    ) -> DMARequest | None:
        """Create one unique whole-tile request."""
        key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
        if key in state.dma_requests or state.resources.ocm.get_tile_state(key) is not WeightState.ABSENT:
            return None
        request = DMARequest(
            layer_id=tile.layer_id,
            expert_id=tile.expert_id,
            matrix_type=tile.matrix_type,
            tile_id=tile.tile_id,
            size_bytes=tile.size_bytes,
            request_type=request_type,
            score=score,
            issue_cycle=issue_cycle,
            candidate_rank=candidate_rank,
        )
        state.dma_requests[key] = request
        state.queue.push(issue_cycle, EventType.DMA_REQUEST, {
            "request": request, "request_type": request_type,
        })
        return request

    def _dma_request(self, state: _RunState, event: Event) -> None:
        request: DMARequest = event.payload["request"]
        if request.canceled:
            return
        state.resources.ocm.mark_queued(request.key, request.request_type)
        state.resources.dma.enqueue(request)
        state.queue.push(event.cycle, EventType.DMA_START)

    def _simulator_state(self, state: _RunState) -> SimulatorState:
        ready_cycles: dict[int, int | None] = {}
        for task in state.pending_tasks:
            key = (
                (task.layer_id, task.expert_id, "gate_up", 0)
                if state.simulation.granularity == "tile"
                else (task.layer_id, task.expert_id, "gate_up")
            )
            if state.resources.ocm.is_resident(key):
                ready_cycles[task.expert_id] = state.resources.ocm.get_ready_cycle(key)
            elif state.resources.ocm.get_state(key) in {
                WeightState.IN_FLIGHT_DEMAND, WeightState.IN_FLIGHT_SPECULATIVE,
            }:
                ready_cycles[task.expert_id] = state.resources.ocm.get_ready_cycle(key)
            else:
                ready_cycles[task.expert_id] = None
        next_tasks = [task for task in state.pending_tasks if task is not state.current_task]
        next_task = state.expert_scheduler.choose_next(
            next_tasks,
            SimulatorState(ready_cycles),
            state.current_cycle,
        ) if state.resolved else None
        return SimulatorState(
            ready_cycles,
            state.resources.ocm,
            state.event.layer_id,
            state.current_task.expert_id if state.current_task else None,
            state.current_phase,
            next_task.expert_id if next_task else None,
            tuple(state.tasks) if state.resolved else (),
            ({expert_id: rank for rank, expert_id in enumerate(state.event.true_expert_ids)}
             if state.resolved else None),
            state.current_tile_id if state.simulation.granularity == "tile" else None,
        )

    def _dma_start(self, state: _RunState, event: Event) -> None:
        dma = state.resources.dma
        if dma.active_request is not None:
            return
        request = state.dma_scheduler.choose_next_request(
            dma.pending_requests(), self._simulator_state(state), event.cycle
        )
        if request is None:
            return
        start_cycle = max(event.cycle, dma.available_cycle)
        if start_cycle != event.cycle:
            state.queue.push(start_cycle, EventType.DMA_START)
            return
        if request.request_type == "demand":
            dma.demand_queue_wait_cycles += start_cycle - request.issue_cycle
        end_cycle = dma.start_request(request, start_cycle)
        state.resources.ocm.mark_in_flight(request.key, request.request_type, end_cycle)
        state.queue.push(end_cycle, EventType.DMA_DONE, {"request": request})

    def _dma_done(self, state: _RunState, event: Event) -> None:
        request: DMARequest = event.payload["request"]
        state.resources.ocm.mark_resident(request.key, request.request_type, event.cycle)
        state.resources.dma.complete_request(request)
        operation = f"{request.request_type}_{request.matrix_type}_e{request.expert_id}"
        tile = state.tile_by_key.get(request.key) if request.tile_id is not None else None
        if request.tile_id is not None:
            short_matrix = "gu" if request.matrix_type == "gate_up" else "down"
            operation = f"{request.request_type}_{short_matrix}_e{request.expert_id}_t{request.tile_id}"
        entry = TimelineEntry(
            "dma", operation,
            request.start_cycle if request.start_cycle is not None else event.cycle,
            event.cycle,
            state.event.request_id, state.event.layer_id, request.expert_id,
            request.request_type, request.useful, request.committed, request.candidate_rank,
            request.tile_id,
            tile.num_tiles if tile else None,
            tile.size_bytes if tile else None,
            tile.compute_ops if tile else None,
            None,
            event.cycle,
        )
        state.timeline.append(entry)
        state.dma_timeline_entries[id(request)] = entry
        state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE)
        state.queue.push(event.cycle, EventType.DMA_START)

    def _dma_commit(self, state: _RunState, event: Event) -> None:
        request: DMARequest = event.payload["request"]
        if request.committed:
            return
        request.committed = True
        request.useful = True
        state.resources.ocm.commit(request.key)
        state.resources.dma.speculative_requests_committed += 1
        if request.started and state.resources.dma.active_request is not request:
            state.resources.dma.useful_speculative_bytes += request.size_bytes
        entry = state.dma_timeline_entries.get(id(request))
        if entry is not None:
            entry.useful = True
            entry.committed = True
        state.timeline.append(TimelineEntry(
            "controller", f"dma_commit_{request.matrix_type}_e{request.expert_id}",
            event.cycle, event.cycle + self.hardware.pl_control.commit_cycles,
            state.event.request_id, state.event.layer_id,
            request.expert_id, "speculative", True, True, request.candidate_rank,
        ))

    def _dma_cancel(self, state: _RunState, event: Event) -> None:
        request: DMARequest = event.payload["request"]
        if request.canceled or request.useful is False:
            return
        if not request.started and state.resources.ocm.cancel_if_queued(request.key):
            request.canceled = True
            request.useful = False
            state.resources.dma.canceled_speculative_bytes += request.size_bytes
            state.resources.dma.speculative_requests_canceled += 1
        else:
            request.useful = False
            state.resources.ocm.mark_wasted(request.key)
            if request.started and state.resources.dma.active_request is not request:
                state.resources.dma.wasted_speculative_bytes += request.size_bytes
        entry = state.dma_timeline_entries.get(id(request))
        if entry is not None:
            entry.useful = False
            entry.committed = False
        state.timeline.append(TimelineEntry(
            "controller", f"dma_cancel_{request.matrix_type}_e{request.expert_id}",
            event.cycle, event.cycle + self.hardware.pl_control.cancel_cycles,
            state.event.request_id, state.event.layer_id,
            request.expert_id, "speculative", False, False, request.candidate_rank,
        ))

    def _scheduler_invoke(self, state: _RunState, event: Event) -> None:
        if not state.resolved:
            return
        if event.cycle < state.expert_schedule_ready_cycle:
            state.queue.push(state.expert_schedule_ready_cycle, EventType.SCHEDULER_INVOKE)
            return
        if state.current_task is None:
            if not state.pending_tasks:
                self._schedule_layer_done(state, event.cycle)
                return
            view = self._simulator_state(state)
            task = state.expert_scheduler.choose_next(state.pending_tasks, view, event.cycle)
            if task is None:
                raise RuntimeError("expert scheduler returned None with pending experts")
            state.current_task = task
            state.current_phase = "waiting_gate"
            state.current_tile_id = 0
            state.current_record = ExpertExecutionRecord(task.expert_id, token_count=task.token_count)
            if state.scenario is Scenario.DEMAND_SERIAL:
                if state.simulation.granularity == "tile":
                    for tile in state.tile_sets[task.expert_id].gate_up_tiles:
                        self._issue_tile_dma(state, tile, "demand", event.cycle)
                else:
                    self._issue_dma(state, task, "gate_up", "demand", event.cycle)
        self._try_progress_current(state, event.cycle)

    def _try_progress_current(self, state: _RunState, cycle: int) -> None:
        task = state.current_task
        record = state.current_record
        if task is None or record is None:
            return
        if state.simulation.granularity == "tile":
            self._try_progress_tile(state, cycle)
            return
        if state.current_phase == "waiting_gate":
            key = (task.layer_id, task.expert_id, "gate_up")
            if state.resources.ocm.is_resident(key):
                record.gate_up_weight_ready_cycle = state.resources.ocm.get_ready_cycle(key)
                self._close_memory_wait(state, cycle)
                state.current_phase = "gate_up"
                state.queue.push(cycle, EventType.AIE_OPERATION_START, {
                    "operation": "gate_up",
                    "operation_count": gate_up_ops(self.model, task.token_count),
                })
            elif state.memory_wait_start_cycle is None:
                state.memory_wait_start_cycle = cycle
        elif state.current_phase == "waiting_down":
            key = (task.layer_id, task.expert_id, "down")
            if state.resources.ocm.is_resident(key):
                record.down_weight_ready_cycle = state.resources.ocm.get_ready_cycle(key)
                self._close_memory_wait(state, cycle)
                state.current_phase = "down"
                state.queue.push(cycle, EventType.AIE_OPERATION_START, {
                    "operation": "down",
                    "operation_count": down_ops(self.model, task.token_count),
                })
            elif state.memory_wait_start_cycle is None:
                state.memory_wait_start_cycle = cycle

    def _try_progress_tile(self, state: _RunState, cycle: int) -> None:
        """Start the current matrix tile as soon as dependencies and bytes permit."""
        task = state.current_task
        record = state.current_record
        if task is None or record is None:
            return
        if state.current_phase not in {"waiting_gate", "waiting_down"}:
            return
        matrix_type = "gate_up" if state.current_phase == "waiting_gate" else "down"
        tiles = (
            state.tile_sets[task.expert_id].gate_up_tiles
            if matrix_type == "gate_up"
            else state.tile_sets[task.expert_id].down_tiles
        )
        if state.current_tile_id >= len(tiles):
            raise RuntimeError("tile execution index exceeds matrix tile count")
        if state.scenario is Scenario.DEMAND_SERIAL:
            all_resident = all(
                state.resources.ocm.is_tile_resident(
                    (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
                )
                for tile in tiles
            )
            if not all_resident:
                if state.memory_wait_start_cycle is None:
                    state.memory_wait_start_cycle = cycle
                return
        tile = tiles[state.current_tile_id]
        key = (tile.layer_id, tile.expert_id, tile.matrix_type, tile.tile_id)
        if not state.resources.ocm.is_tile_resident(key):
            if state.memory_wait_start_cycle is None:
                state.memory_wait_start_cycle = cycle
            return
        eligible_cycle = state.memory_wait_start_cycle if state.memory_wait_start_cycle is not None else cycle
        starvation = cycle - eligible_cycle
        self._close_memory_wait(state, cycle, matrix_type=matrix_type, tile_id=tile.tile_id)
        ready_cycle = state.resources.ocm.get_tile_ready_cycle(key)
        if tile.tile_id == 0:
            if matrix_type == "gate_up":
                record.gate_up_weight_ready_cycle = ready_cycle
                record.gate_up_start_cycle = cycle
            else:
                record.down_weight_ready_cycle = ready_cycle
                record.down_start_cycle = cycle
        state.current_phase = matrix_type
        state.queue.push(cycle, EventType.AIE_OPERATION_START, {
            "operation": matrix_type,
            "operation_count": tile.compute_ops,
            "tile": tile,
            "compute_eligible_cycle": eligible_cycle,
            "starvation_cycles": starvation,
            "ready_cycle": ready_cycle,
        })

    def _close_memory_wait(
        self,
        state: _RunState,
        cycle: int,
        *,
        matrix_type: str | None = None,
        tile_id: int | None = None,
    ) -> None:
        if state.memory_wait_start_cycle is None:
            return
        stall = cycle - state.memory_wait_start_cycle
        if stall > 0:
            state.memory_stall_cycles += stall
            if state.current_record is not None:
                state.current_record.memory_stall_cycles += stall
                if tile_id == 0:
                    state.current_record.first_tile_stall_cycles += stall
                elif tile_id is not None:
                    state.current_record.inter_tile_stall_cycles += stall
            if tile_id == 0:
                state.first_tile_stall_cycles += stall
            elif tile_id is not None:
                state.inter_tile_stall_cycles += stall
            if matrix_type == "gate_up":
                state.gate_up_tile_stall_cycles += stall
            elif matrix_type == "down":
                state.down_tile_stall_cycles += stall
            state.timeline.append(TimelineEntry(
                "aie", "memory_stall", state.memory_wait_start_cycle, cycle,
                state.event.request_id, state.event.layer_id,
                state.current_task.expert_id if state.current_task else None,
            ))
        state.memory_wait_start_cycle = None

    def _nonlinear_start(self, state: _RunState, event: Event) -> None:
        if state.current_task is None or state.current_record is None:
            raise RuntimeError("nonlinear start without current expert")
        task = NonlinearTask(nonlinear_elements(self.model, state.current_task.token_count))
        start_cycle = state.resources.nonlinear.earliest_start(task, event.cycle)
        end_cycle = state.resources.nonlinear.reserve(task, start_cycle)
        state.nonlinear_active = True
        state.current_record.nonlinear_start_cycle = start_cycle
        state.current_record.nonlinear_done_cycle = end_cycle
        state.dependency_wait_cycles += end_cycle - event.cycle
        state.timeline.append(TimelineEntry(
            "nonlinear", f"nonlinear_e{state.current_task.expert_id}",
            start_cycle, end_cycle, state.event.request_id, state.event.layer_id,
            state.current_task.expert_id,
        ))
        state.queue.push(end_cycle, EventType.NONLINEAR_DONE)

    def _nonlinear_done(self, state: _RunState, event: Event) -> None:
        state.nonlinear_active = False
        if state.current_task is None:
            raise RuntimeError("nonlinear completion without current expert")
        state.current_phase = "waiting_down"
        if state.scenario is Scenario.DEMAND_SERIAL:
            if state.simulation.granularity == "tile":
                for tile in state.tile_sets[state.current_task.expert_id].down_tiles:
                    self._issue_tile_dma(state, tile, "demand", event.cycle)
            else:
                self._issue_dma(state, state.current_task, "down", "demand", event.cycle)
        self._try_progress_current(state, event.cycle)
        state.queue.push(event.cycle, EventType.DMA_START)

    def _expert_done(self, state: _RunState, event: Event) -> None:
        if state.current_task is None or state.current_record is None:
            raise RuntimeError("expert completion without current expert")
        state.pending_tasks.remove(state.current_task)
        state.expert_records.append(state.current_record)
        state.current_task = None
        state.current_record = None
        state.current_phase = None
        state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE)
        state.queue.push(event.cycle, EventType.DMA_START)

    def _schedule_layer_done(self, state: _RunState, experts_done_cycle: int) -> None:
        if any(event.event_type is EventType.LAYER_DONE for event in state.queue):
            return
        layer_done = experts_done_cycle + self.hardware.pl_control.combine_cycles
        if layer_done > experts_done_cycle:
            state.timeline.append(TimelineEntry(
                "controller", "combine", experts_done_cycle, layer_done,
                state.event.request_id, state.event.layer_id,
            ))
        state.queue.push(layer_done, EventType.LAYER_DONE)

    def _layer_done(self, state: _RunState, event: Event) -> None:
        state.layer_done_cycle = event.cycle

    def _build_output(self, state: _RunState) -> SimulationOutput:
        true_ids = set(state.event.true_expert_ids)
        candidates = set(state.predictor_output.candidate_ids) if state.predictor_output else set()
        predictive_scenario = state.scenario in {Scenario.PREDICTIVE, Scenario.ORACLE}
        static_recall = len(candidates & true_ids) / len(true_ids) if predictive_scenario else None
        any_hit = bool(candidates & true_ids) if predictive_scenario else None
        full_hit = true_ids <= candidates if predictive_scenario else None
        aie = state.resources.aie
        router_payload = aie.intervals[0].payload_cycles
        expert_payload = aie.busy_cycles - router_payload
        total_cycles = state.layer_done_cycle
        control_gap = (
            self.hardware.pl_control.topk_cycles
            + self.hardware.pl_control.dispatch_cycles
            + self.hardware.pl_control.combine_cycles
            + state.resolve_control_cycles
        )
        idle = total_cycles - aie.busy_cycles - aie.control_transition_cycles
        unutilized = max(0, idle - state.memory_stall_cycles - state.dependency_wait_cycles - control_gap)
        dma = state.resources.dma
        tile_mode = state.simulation.granularity == "tile"
        tile_requests = [request for request in state.dma_requests.values() if request.tile_id is not None]
        transferred_tile_requests = [request for request in tile_requests if request.end_cycle is not None]
        total_tiles_required = (
            sum(
                len(state.tile_sets[task.expert_id].gate_up_tiles)
                + len(state.tile_sets[task.expert_id].down_tiles)
                for task in state.tasks
            )
            if tile_mode else 0
        )
        useful_leads = [
            state.dispatch_done_cycle - request.issue_cycle
            for request in tile_requests
            if request.request_type == "speculative" and request.useful is True
        ]
        result = SimulationResult(
            scenario=state.scenario.value,
            request_id=state.event.request_id,
            token_position=state.event.token_position,
            layer_id=state.event.layer_id,
            layer_start_cycle=0,
            router_start_cycle=state.router_start_cycle,
            router_done_cycle=state.router_done_cycle,
            topk_done_cycle=state.topk_done_cycle,
            dispatch_done_cycle=state.dispatch_done_cycle,
            resolve_cycle=state.dispatch_done_cycle,
            layer_done_cycle=state.layer_done_cycle,
            total_cycles=total_cycles,
            aie_router_busy_cycles=router_payload,
            aie_expert_busy_cycles=expert_payload,
            aie_memory_stall_cycles=state.memory_stall_cycles,
            aie_idle_cycles=idle,
            nonlinear_busy_cycles=state.resources.nonlinear.busy_cycles,
            dma_busy_cycles=dma.busy_cycles,
            demand_bytes=dma.demand_bytes,
            speculative_bytes=dma.speculative_bytes,
            useful_speculative_bytes=dma.useful_speculative_bytes,
            wasted_speculative_bytes=dma.wasted_speculative_bytes,
            canceled_speculative_bytes=dma.canceled_speculative_bytes,
            static_recall=static_recall,
            any_hit=any_hit,
            full_hit=full_hit,
            ready_recall_at_router_done=state.ready_recall_at_router_done if predictive_scenario else None,
            ready_recall_at_resolve=(
                state.ready_recall_at_resolve if predictive_scenario or tile_mode else None
            ),
            expert_records=state.expert_records,
            aie_control_transition_cycles=aie.control_transition_cycles,
            aie_dependency_wait_cycles=state.dependency_wait_cycles,
            aie_control_gap_cycles=control_gap,
            aie_unutilized_cycles=unutilized,
            demand_queue_wait_cycles=dma.demand_queue_wait_cycles,
            demand_blocked_by_speculative_cycles=dma.demand_blocked_by_speculative_cycles,
            speculative_requests_issued=dma.speculative_requests_issued,
            speculative_requests_committed=dma.speculative_requests_committed,
            speculative_requests_canceled=dma.speculative_requests_canceled,
            tile_mode=tile_mode,
            tile_size_bytes=state.tile_config.tile_size_bytes if state.tile_config else None,
            total_tiles_required=total_tiles_required,
            total_tiles_transferred=len(transferred_tile_requests),
            demand_tiles=sum(request.request_type == "demand" for request in transferred_tile_requests),
            speculative_tiles=sum(request.request_type == "speculative" for request in transferred_tile_requests),
            useful_speculative_tiles=sum(
                request.request_type == "speculative" and request.useful is True
                for request in transferred_tile_requests
            ),
            wasted_speculative_tiles=sum(
                request.request_type == "speculative" and request.useful is False
                for request in transferred_tile_requests
            ),
            canceled_speculative_tiles=sum(request.canceled for request in tile_requests),
            true_tiles_resident_at_resolve=state.true_tiles_resident_at_resolve,
            true_tile_resident_fraction_at_resolve=(
                state.true_tiles_resident_at_resolve / total_tiles_required
                if total_tiles_required else 0.0
            ),
            true_experts_first_tile_ready_at_resolve=state.true_first_tiles_resident_at_resolve,
            aie_first_tile_stall_cycles=state.first_tile_stall_cycles,
            aie_inter_tile_stall_cycles=state.inter_tile_stall_cycles,
            aie_gate_up_tile_stall_cycles=state.gate_up_tile_stall_cycles,
            aie_down_tile_stall_cycles=state.down_tile_stall_cycles,
            mean_useful_prefetch_lead_cycles=(
                sum(useful_leads) / len(useful_leads) if useful_leads else None
            ),
        )
        timeline = sorted(
            state.timeline,
            key=lambda item: (
                item.start_cycle, item.end_cycle, item.resource, item.operation,
                -1 if item.expert_id is None else item.expert_id,
            ),
        )
        return SimulationOutput(result, timeline)
