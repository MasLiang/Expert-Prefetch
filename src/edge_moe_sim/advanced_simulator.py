"""Event-driven batch/F-tile simulator for bandwidth-to-compute transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from statistics import fmean, median

from .advanced_schema import (
    AdvancedExecutionConfig, AdvancedExpertRecord, AdvancedScenario,
    AdvancedSimulationResult,
)
from .batch_predictor import BatchPredictor, BatchPredictorOutput
from .batch_routing import BatchRoutingEvent, batch_workload_stats
from .event_queue import EventQueue
from .events import Event, EventType
from .f_tile import FTileGroup, build_f_tile_groups
from .hardware.aie_pool import AIEPool, AIEPoolTask
from .hardware.capacity_ocm import CapacityOnChipMemory
from .hardware.multi_channel_memory import MemoryRequest, MultiChannelMemory
from .hardware.nonlinear import NonlinearEngine, NonlinearTask
from .hardware_profile import HardwareProfile
from .roofline_metrics import calculate_roofline_metrics
from .schema import ModelConfig, NonlinearConfig, ValidationError
from .timeline import TimelineEntry
from .tile_scheduler import (
    CurrentTimelineState, ExpertTile, PrefetchPlan, TilePredictionResult,
    TilePrefetchScheduler, TileResidentState, TileSchedulingHardware,
)
from .work_model import expert_weight_bytes, router_ops


TileObjectKey = tuple[int, int, str, int]


@dataclass
class _ExpertState:
    expert_id: int
    token_count: int
    groups: tuple[FTileGroup, ...]
    record: AdvancedExpertRecord
    launched: bool = False
    done: bool = False
    tile_id: int = 0
    stage: str = "waiting_gu"
    compute_busy: bool = False
    weight_wait_start: int | None = None


@dataclass
class _AdvancedState:
    event: BatchRoutingEvent
    scenario: AdvancedScenario
    config: AdvancedExecutionConfig
    profile: HardwareProfile
    queue: EventQueue
    memory: MultiChannelMemory
    aie: AIEPool
    nonlinear: NonlinearEngine
    ocm: CapacityOnChipMemory
    experts: dict[int, _ExpertState]
    predictor_output: BatchPredictorOutput | None
    candidates: tuple[object, ...]
    timeline: list[TimelineEntry] = field(default_factory=list)
    requests: list[MemoryRequest] = field(default_factory=list)
    object_to_request: dict[TileObjectKey, MemoryRequest] = field(default_factory=dict)
    current_cycle: int = 0
    resolved: bool = False
    router_resolve_cycle: int = 0
    compute_finish_cycle: int = 0
    demand_serial_order: list[int] = field(default_factory=list)
    active_experts: set[int] = field(default_factory=set)
    pending_experts: set[int] = field(default_factory=set)
    candidate_depths: dict[int, tuple[int, int]] = field(default_factory=dict)
    resident_true_bytes_at_resolve: int = 0
    startable_at_resolve: set[int] = field(default_factory=set)
    full_resident_at_resolve: set[int] = field(default_factory=set)
    prefetched_true_experts: set[int] = field(default_factory=set)
    demand_blocked_cycles: int = 0
    blocked_active_request_ids: set[int] = field(default_factory=set)
    speculative_blocked_cycles: int = 0
    startup_readiness_at_resolve: dict[int, float] = field(default_factory=dict)
    startup_gap_bytes_at_resolve: dict[int, int] = field(default_factory=dict)
    tile_scheduler: TilePrefetchScheduler | None = None
    tile_prefetch_plan: PrefetchPlan | None = None
    router_scheduled_done_cycle: int = 0


class AdvancedSimulator:
    """Simulate one batch-routed layer through the shared resource event DAG."""

    def __init__(self, model: ModelConfig, profile: HardwareProfile) -> None:
        self.model = model
        self.profile = profile

    def simulate(
        self,
        event: BatchRoutingEvent,
        scenario: AdvancedScenario | str,
        config: AdvancedExecutionConfig,
        *,
        predictor: BatchPredictor | None = None,
        tile_prefetch_scheduler: TilePrefetchScheduler | None = None,
    ) -> AdvancedSimulationResult:
        """Run a causal batch/F-tile simulation and return detailed metrics."""
        from .execution_rules import ExecutionRules
        if config.demand_loading != "tile_streaming":
            raise ValidationError("demand_loading modes are supported by UnifiedSequenceSimulator only")
        if config.execution_rules != ExecutionRules():
            raise ValidationError('非默认 execution_rules 仅由 UnifiedSequenceSimulator 支持，不可静默忽略')
        event.validate(self.model)
        selected = AdvancedScenario(scenario)
        if config.aie_units_per_expert > self.profile.total_aie_units:
            raise ValidationError("aie_units_per_expert exceeds total_aie_units")
        if config.router_aie_units > self.profile.total_aie_units:
            raise ValidationError("router_aie_units exceeds total_aie_units")
        predictive = selected in {
            AdvancedScenario.PREDICTIVE_FULL_EXPERT,
            AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,
            AdvancedScenario.PREDICTIVE_RANKED_PREFIX,
            AdvancedScenario.PREDICTIVE_DSE,
            AdvancedScenario.ORACLE_OPTIMAL_DEPTH,
        }
        predictor_output = predictor.predict(event) if predictor is not None else None
        if predictive and predictor_output is None:
            raise ValidationError(f"scenario {selected.value} requires a batch predictor")
        candidates = (
            predictor_output.aggregate_candidates(config.candidate_count)
            if predictor_output is not None else ()
        )
        counts = event.expert_token_counts()
        experts: dict[int, _ExpertState] = {}
        for expert_id, token_count in counts.items():
            groups = build_f_tile_groups(self.model, expert_id, token_count, config.f_tile_size)
            record = AdvancedExpertRecord(
                expert_id, token_count, expert_weight_bytes(self.model),
                sum(group.gate_up_ops + group.down_ops for group in groups),
            )
            experts[expert_id] = _ExpertState(expert_id, token_count, groups, record)
        state = _AdvancedState(
            event, selected, config, self.profile, EventQueue(),
            MultiChannelMemory(self.profile, config.placement_policy),
            AIEPool(self.profile),
            NonlinearEngine(NonlinearConfig(
                config.nonlinear_elements_per_cycle, config.nonlinear_startup_cycles
            )),
            CapacityOnChipMemory(self.profile.ocm_bytes), experts,
            predictor_output, candidates,
            pending_experts=set(experts), demand_serial_order=sorted(experts),
            tile_scheduler=tile_prefetch_scheduler,
        )
        self._schedule_router(state)
        if predictor_output is not None:
            state.queue.push(
                predictor_output.prediction_ready_cycle, EventType.PREDICTOR_READY
            )
        while state.queue:
            queued = state.queue.pop()
            if queued.cycle < state.current_cycle:
                raise RuntimeError("advanced event cycle moved backwards")
            state.current_cycle = queued.cycle
            self._handle(state, queued)
        if not all(expert.done for expert in state.experts.values()):
            raise RuntimeError("advanced simulation ended before all experts completed")
        makespan = max(
            state.compute_finish_cycle,
            max((request.end_cycle or 0 for request in state.requests if request.started), default=0),
        )
        return self._build_result(state, makespan)

    def _schedule_router(self, state: _AdvancedState) -> None:
        task = AIEPoolTask(
            "router", router_ops(self.model, state.event.batch_size),
            state.config.router_aie_units, state.event.layer_id,
        )
        start, end = state.aie.reserve(task, 0)
        state.router_scheduled_done_cycle = end
        state.timeline.append(TimelineEntry(
            "AIE_POOL", f"batch_router_l{state.event.layer_id}", start, end,
            state.event.request_id, state.event.layer_id, compute_ops=task.operation_count,
        ))
        state.queue.push(end, EventType.AIE_OPERATION_DONE, {"kind": "router"})

    def _handle(self, state: _AdvancedState, event: Event) -> None:
        if event.event_type is EventType.PREDICTOR_READY:
            self._predictor_ready(state, event.cycle)
        elif event.event_type is EventType.AIE_OPERATION_DONE:
            self._aie_done(state, event)
        elif event.event_type is EventType.DMA_DONE:
            self._memory_done(state, event)
        elif event.event_type is EventType.NONLINEAR_DONE:
            self._nonlinear_done(state, event)
        elif event.event_type is EventType.SCHEDULER_INVOKE:
            self._try_launch_experts(state, event.cycle)
        else:
            raise RuntimeError(f"advanced simulator received unsupported event {event.event_type}")

    def _predictor_ready(self, state: _AdvancedState, cycle: int) -> None:
        if state.resolved:
            return
        if state.tile_scheduler is not None:
            self._scheduled_predictor_ready(state, cycle)
            return
        decisions = self._prefetch_decisions(state)
        used = 0
        for expert_id, matrix, depth, score, rank in decisions:
            groups = self._geometry_groups(state, expert_id)
            selected = groups[:min(depth, len(groups))]
            for group in selected:
                size = group.gate_up_bytes if matrix == "gate_up" else group.down_bytes
                if (
                    state.config.speculative_byte_budget is not None
                    and used + size > state.config.speculative_byte_budget
                ):
                    continue
                key = self._key(state, expert_id, matrix, group.tile_id)
                if self._issue_tile_request(
                    state, key, size, "speculative", cycle,
                    score=score, rank=rank,
                ):
                    used += size
                    old = state.candidate_depths.get(expert_id, (0, 0))
                    state.candidate_depths[expert_id] = (
                        max(old[0], group.tile_id + 1) if matrix == "gate_up" else old[0],
                        max(old[1], group.tile_id + 1) if matrix == "down" else old[1],
                    )
        self._kick_memory(state, cycle)

    def _scheduled_predictor_ready(self, state: _AdvancedState, cycle: int) -> None:
        """Allocate logical tiles, then issue their two matrix fragments in plan order."""
        candidates = tuple((candidate.expert_id, float(candidate.score)) for candidate in state.candidates)
        geometry: dict[int, tuple[ExpertTile, ...]] = {}
        for rank, candidate in enumerate(state.candidates):
            groups = self._geometry_groups(state, candidate.expert_id)
            geometry[candidate.expert_id] = tuple(
                ExpertTile(
                    candidate.expert_id, group.tile_id,
                    group.gate_up_bytes + group.down_bytes,
                    group.gate_up_bytes, group.down_bytes,
                    rank, float(candidate.score), cycle, TileResidentState.PLANNED,
                )
                for group in groups
            )
        hardware = TileSchedulingHardware(
            state.profile.ocm_bytes,
            state.profile.effective_bytes_per_cycle_per_channel,
            state.profile.num_memory_channels,
        )
        timeline = CurrentTimelineState(
            cycle, state.ocm.current_bytes, state.config.speculative_byte_budget,
            state.router_scheduled_done_cycle,
        )
        plan = state.tile_scheduler.schedule(
            TilePredictionResult(candidates), geometry, hardware, timeline
        )
        state.tile_prefetch_plan = plan
        # Gate-Up fragments are issued first in logical-plan order to maximize
        # startup coverage. Down fragments retain the same relative plan order.
        physical_order = 0
        for matrix in ("gate_up", "down"):
            for transfer in plan.transfers:
                tile = transfer.tile
                size = tile.gate_up_bytes if matrix == "gate_up" else tile.down_bytes
                key = self._key(state, tile.expert_id, matrix, tile.tile_id)
                if self._issue_tile_request(
                    state, key, size, "speculative", cycle,
                    score=tile.confidence, rank=tile.predicted_rank,
                    schedule_order=physical_order,
                ):
                    old = state.candidate_depths.get(tile.expert_id, (0, 0))
                    state.candidate_depths[tile.expert_id] = (
                        max(old[0], tile.tile_id + 1),
                        max(old[1], tile.tile_id + 1),
                    )
                physical_order += 1
        self._kick_memory(state, cycle)

    def _prefetch_decisions(
        self, state: _AdvancedState,
    ) -> list[tuple[int, str, int, float, int]]:
        decisions = []
        for rank, candidate in enumerate(state.candidates):
            num_tiles = len(self._geometry_groups(state, candidate.expert_id))
            if state.scenario is AdvancedScenario.PREDICTIVE_FULL_EXPERT:
                gu_depth = down_depth = num_tiles
            elif state.scenario is AdvancedScenario.PREDICTIVE_RANKED_PREFIX:
                gu_depth = (
                    state.config.rank_based_depths[rank]
                    if rank < len(state.config.rank_based_depths)
                    else state.config.uniform_gu_prefetch_depth
                )
                down_depth = state.config.down_prefetch_depth
            elif state.scenario in {
                AdvancedScenario.PREDICTIVE_DSE, AdvancedScenario.ORACLE_OPTIMAL_DEPTH,
            } and state.config.explicit_depths:
                gu_depth = (
                    state.config.explicit_depths[rank]
                    if rank < len(state.config.explicit_depths) else 0
                )
                down_depth = state.config.down_prefetch_depth
            else:
                gu_depth = state.config.uniform_gu_prefetch_depth
                down_depth = state.config.down_prefetch_depth
            decisions.append((candidate.expert_id, "gate_up", gu_depth, candidate.score, rank))
            if down_depth:
                decisions.append((candidate.expert_id, "down", down_depth, candidate.score, rank))
        return decisions

    def _geometry_groups(self, state: _AdvancedState, expert_id: int) -> tuple[FTileGroup, ...]:
        if expert_id in state.experts:
            return state.experts[expert_id].groups
        return build_f_tile_groups(self.model, expert_id, 1, state.config.f_tile_size)

    @staticmethod
    def _key(state: _AdvancedState, expert_id: int, matrix: str, tile_id: int) -> TileObjectKey:
        return (state.event.layer_id, expert_id, matrix, tile_id)

    def _issue_tile_request(
        self, state: _AdvancedState, key: TileObjectKey, size: int,
        request_type: str, cycle: int, *, score: float = 0.0,
        rank: int | None = None,
        schedule_order: int | None = None,
    ) -> bool:
        existing = state.object_to_request.get(key)
        if existing is not None and not existing.canceled:
            return False
        admission = state.ocm.admit(
            key, size, "speculative" if request_type == "speculative" else "demand",
            score, cycle,
        )
        self._handle_evictions(state, admission.evicted_keys, cycle)
        if not admission.admitted:
            return False
        request = state.memory.new_request(
            layer_id=key[0], expert_id=key[1], matrix_type=key[2],
            tile_ids=(key[3],), object_keys=(key,), transfer_bytes=size,
            request_type=request_type, issue_cycle=cycle, score=score,
            candidate_rank=rank,
            speculative_scope="future_layer" if request_type == "speculative" else "",
            schedule_order=schedule_order,
        )
        queued = state.memory.enqueue(
            request, coalesce_adjacent=state.config.coalesce_adjacent_tiles
        )
        if queued is request:
            state.requests.append(request)
        state.object_to_request[key] = queued
        if request_type == "demand":
            active_spec = {
                id(channel.active_request): channel.active_request
                for channel in state.memory.channels
                if channel.active_request is not None
                and channel.active_request.request_type == "speculative"
            }
            for active_id, active in active_spec.items():
                if active_id not in state.blocked_active_request_ids and active.end_cycle is not None:
                    state.blocked_active_request_ids.add(active_id)
                    state.demand_blocked_cycles += max(0, active.end_cycle - cycle)
        return True

    def _handle_evictions(
        self, state: _AdvancedState, keys: tuple[object, ...], cycle: int
    ) -> None:
        for raw_key in keys:
            key = raw_key
            request = state.object_to_request.get(key)
            if request is None or request.started:
                continue
            state.memory.cancel(request)
            for object_key in request.object_keys:
                if object_key in state.ocm.objects:
                    state.ocm.release(object_key, cycle)

    def _kick_memory(self, state: _AdvancedState, cycle: int) -> None:
        for request in state.memory.start_ready(cycle):
            for key in request.object_keys:
                state.ocm.set_running(key, True)
            if request.request_type == "speculative":
                state.speculative_blocked_cycles += cycle - request.issue_cycle
            state.queue.push(request.end_cycle or cycle, EventType.DMA_DONE, {"request": request})
            state.timeline.append(TimelineEntry(
                "MEMORY", self._memory_operation(request), cycle,
                request.end_cycle or cycle, state.event.request_id, request.layer_id,
                request.expert_id, request.request_type, request.useful,
                request.committed, request.candidate_rank,
                request.tile_ids[0], tile_size_bytes=request.transfer_bytes,
                speculative_scope=request.speculative_scope,
            ))

    @staticmethod
    def _memory_operation(request: MemoryRequest) -> str:
        tiles = (
            str(request.tile_ids[0]) if len(request.tile_ids) == 1
            else f"{request.tile_ids[0]}-{request.tile_ids[-1]}"
        )
        prefix = "spec" if request.request_type == "speculative" else request.request_type
        return f"{prefix}_{request.matrix_type}_e{request.expert_id}_t{tiles}_ch{'-'.join(map(str, request.channel_ids))}"

    def _aie_done(self, state: _AdvancedState, event: Event) -> None:
        if event.payload["kind"] == "router":
            self._resolve(state, event.cycle)
            return
        expert = state.experts[event.payload["expert_id"]]
        matrix = event.payload["matrix"]
        group = expert.groups[expert.tile_id]
        expert.compute_busy = False
        if matrix == "gate_up":
            state.ocm.release(self._key(state, expert.expert_id, "gate_up", group.tile_id), event.cycle)
            self._refill_demand_window(state, event.cycle)
            expert.stage = "nonlinear"
            task = NonlinearTask(group.nonlinear_elements)
            start = state.nonlinear.earliest_start(task, event.cycle)
            end = state.nonlinear.reserve(task, start)
            state.timeline.append(TimelineEntry(
                "NONLINEAR", f"nl_e{expert.expert_id}_f{group.tile_id}",
                start, end, state.event.request_id, state.event.layer_id,
                expert.expert_id, tile_id=group.tile_id,
            ))
            state.queue.push(end, EventType.NONLINEAR_DONE, {"expert_id": expert.expert_id})
        else:
            state.ocm.release(self._key(state, expert.expert_id, "down", group.tile_id), event.cycle)
            self._refill_demand_window(state, event.cycle)
            expert.tile_id += 1
            if expert.tile_id == len(expert.groups):
                expert.done = True
                expert.record.finish_time = event.cycle
                state.active_experts.discard(expert.expert_id)
                state.compute_finish_cycle = max(state.compute_finish_cycle, event.cycle)
                if state.scenario is AdvancedScenario.DEMAND_SERIAL:
                    self._issue_next_serial_expert(state, event.cycle)
                elif state.scenario in {
                    AdvancedScenario.EXACT_FULL_EXPERT_LOAD,
                    AdvancedScenario.PREDICTIVE_FULL_EXPERT,
                }:
                    self._issue_full_load_batch(state, event.cycle)
                state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE)
            else:
                expert.stage = "waiting_gu"
                expert.weight_wait_start = event.cycle
                self._try_progress_expert(state, expert, event.cycle)

    def _resolve(self, state: _AdvancedState, cycle: int) -> None:
        state.resolved = True
        state.router_resolve_cycle = cycle
        true_ids = set(state.experts)
        for request in state.requests:
            if request.request_type != "speculative":
                continue
            if request.expert_id in true_ids:
                request.committed = True
                request.useful = True
                state.prefetched_true_experts.add(request.expert_id)
                for key in request.object_keys:
                    if key in state.ocm.objects:
                        state.ocm.commit(key)
            else:
                request.useful = False
                if not request.started and state.memory.cancel(request):
                    for key in request.object_keys:
                        state.ocm.release(key, cycle)
                else:
                    for key in request.object_keys:
                        state.ocm.mark_false(key)
                        if request.end_cycle is not None and request.end_cycle <= cycle:
                            state.ocm.release(key, cycle)
        total_true_bytes = 0
        resident_true = 0
        for expert in state.experts.values():
            all_keys = self._expert_keys(state, expert)
            total_true_bytes += sum(size for _, size in all_keys)
            resident_bytes = sum(
                size for key, size in all_keys if state.ocm.is_resident(key, cycle)
            )
            resident_true += resident_bytes
            if self._startup_ready(state, expert, cycle):
                state.startable_at_resolve.add(expert.expert_id)
            startup = [(self._key(state, expert.expert_id, "gate_up", tile_id), expert.groups[tile_id].gate_up_bytes)
                       for tile_id in range(min(state.config.startup_gu_tiles, len(expert.groups)))]
            required_startup = sum(size for _, size in startup)
            resident_startup = sum(size for key, size in startup if state.ocm.is_resident(key, cycle))
            state.startup_readiness_at_resolve[expert.expert_id] = resident_startup / required_startup if required_startup else 1.0
            state.startup_gap_bytes_at_resolve[expert.expert_id] = required_startup - resident_startup
            if all(state.ocm.is_resident(key, cycle) for key, _ in all_keys):
                state.full_resident_at_resolve.add(expert.expert_id)
        state.resident_true_bytes_at_resolve = resident_true
        if state.scenario is AdvancedScenario.IDEAL:
            pass
        elif state.scenario is AdvancedScenario.DEMAND_SERIAL:
            self._issue_next_serial_expert(state, cycle)
        elif state.scenario in {
            AdvancedScenario.EXACT_FULL_EXPERT_LOAD,
            AdvancedScenario.PREDICTIVE_FULL_EXPERT,
        }:
            required = expert_weight_bytes(self.model) * min(
                state.config.max_parallel_experts, len(state.experts)
            )
            if required > self.profile.ocm_bytes:
                raise ValidationError(
                    "full-expert execution requires OCM for each configured parallel expert"
                )
            self._issue_full_load_batch(state, cycle)
        else:
            self._issue_all_missing_demand(state, cycle)
        state.queue.push(cycle, EventType.SCHEDULER_INVOKE)
        self._kick_memory(state, cycle)

    def _expert_keys(
        self, state: _AdvancedState, expert: _ExpertState,
    ) -> list[tuple[TileObjectKey, int]]:
        result = []
        for group in expert.groups:
            result.append((self._key(state, expert.expert_id, "gate_up", group.tile_id), group.gate_up_bytes))
            result.append((self._key(state, expert.expert_id, "down", group.tile_id), group.down_bytes))
        return result

    def _issue_all_missing_demand(self, state: _AdvancedState, cycle: int) -> None:
        for expert_id in sorted(state.experts):
            self._issue_expert_missing_demand(state, state.experts[expert_id], cycle)

    def _issue_full_load_batch(self, state: _AdvancedState, cycle: int) -> None:
        """Admit whole experts only for the next parallel execution wave."""
        in_pipeline = {
            request.expert_id for request in state.requests
            if request.request_type == "demand" and not request.canceled
            and request.expert_id in state.experts
            and not state.experts[request.expert_id].done
        } | state.active_experts
        slots = state.config.max_parallel_experts - len(in_pipeline)
        if slots <= 0:
            return
        pending = sorted(
            (expert for expert in state.experts.values()
             if not expert.done and not expert.launched and expert.expert_id not in in_pipeline),
            key=lambda expert: (-expert.token_count, expert.expert_id),
        )
        for expert in pending[:slots]:
            self._issue_expert_missing_demand(state, expert, cycle)
        self._kick_memory(state, cycle)

    def _refill_demand_window(self, state: _AdvancedState, cycle: int) -> None:
        """Retry deferred demand tiles after consumed buffers free OCM capacity."""
        if not state.resolved or state.scenario in {
            AdvancedScenario.DEMAND_SERIAL,
            AdvancedScenario.EXACT_FULL_EXPERT_LOAD,
            AdvancedScenario.PREDICTIVE_FULL_EXPERT,
            AdvancedScenario.IDEAL,
        }:
            return
        self._issue_all_missing_demand(state, cycle)
        self._kick_memory(state, cycle)

    def _issue_expert_missing_demand(
        self, state: _AdvancedState, expert: _ExpertState, cycle: int
    ) -> None:
        for key, size in self._expert_keys(state, expert):
            request = state.object_to_request.get(key)
            if request is not None and not request.canceled:
                continue
            if state.scenario is AdvancedScenario.IDEAL:
                continue
            self._issue_tile_request(state, key, size, "demand", cycle)

    def _issue_next_serial_expert(self, state: _AdvancedState, cycle: int) -> None:
        while state.demand_serial_order and state.experts[state.demand_serial_order[0]].done:
            state.demand_serial_order.pop(0)
        if not state.demand_serial_order:
            return
        expert = state.experts[state.demand_serial_order[0]]
        self._issue_expert_missing_demand(state, expert, cycle)
        self._kick_memory(state, cycle)

    def _memory_done(self, state: _AdvancedState, event: Event) -> None:
        request: MemoryRequest = event.payload["request"]
        state.memory.complete(request, event.cycle)
        for key in request.object_keys:
            state.ocm.set_running(key, False)
            if key in state.ocm.objects:
                state.ocm.mark_resident(key, event.cycle)
                if request.useful is False and state.resolved:
                    state.ocm.release(key, event.cycle)
        self._kick_memory(state, event.cycle)
        if state.resolved:
            state.queue.push(event.cycle, EventType.SCHEDULER_INVOKE)
            for expert in state.experts.values():
                if expert.launched and not expert.done:
                    self._try_progress_expert(state, expert, event.cycle)

    def _try_launch_experts(self, state: _AdvancedState, cycle: int) -> None:
        if not state.resolved:
            return
        slots = state.config.max_parallel_experts - len(state.active_experts)
        if state.scenario in {
            AdvancedScenario.DEMAND_SERIAL, AdvancedScenario.EXACT_PIPELINE,
        }:
            slots = min(slots, 1)
        if slots <= 0:
            return
        eligible = [
            expert for expert in state.experts.values()
            if not expert.launched and not expert.done and self._expert_ready_to_launch(state, expert, cycle)
        ]
        eligible.sort(key=lambda expert: (-expert.token_count, expert.expert_id))
        for expert in eligible[:slots]:
            expert.launched = True
            state.pending_experts.discard(expert.expert_id)
            state.active_experts.add(expert.expert_id)
            expert.weight_wait_start = state.router_resolve_cycle
            self._try_progress_expert(state, expert, cycle)

    def _expert_ready_to_launch(
        self, state: _AdvancedState, expert: _ExpertState, cycle: int
    ) -> bool:
        if state.scenario is AdvancedScenario.IDEAL:
            return True
        if state.scenario in {
            AdvancedScenario.DEMAND_SERIAL, AdvancedScenario.EXACT_PIPELINE,
            AdvancedScenario.EXACT_FULL_EXPERT_LOAD,
            AdvancedScenario.PREDICTIVE_FULL_EXPERT,
        }:
            return all(state.ocm.is_resident(key, cycle) for key, _ in self._expert_keys(state, expert))
        return self._startup_ready(state, expert, cycle)

    def _startup_ready(self, state: _AdvancedState, expert: _ExpertState, cycle: int) -> bool:
        if state.scenario is AdvancedScenario.IDEAL:
            return True
        depth = min(state.config.startup_gu_tiles, len(expert.groups))
        return all(state.ocm.is_resident(
            self._key(state, expert.expert_id, "gate_up", tile_id), cycle
        ) for tile_id in range(depth))

    def _try_progress_expert(
        self, state: _AdvancedState, expert: _ExpertState, cycle: int
    ) -> None:
        if expert.done or expert.compute_busy or expert.stage not in {"waiting_gu", "waiting_down"}:
            return
        group = expert.groups[expert.tile_id]
        matrix = "gate_up" if expert.stage == "waiting_gu" else "down"
        key = self._key(state, expert.expert_id, matrix, group.tile_id)
        if state.scenario is not AdvancedScenario.IDEAL and not state.ocm.is_resident(key, cycle):
            if expert.weight_wait_start is None:
                expert.weight_wait_start = cycle
            return
        ready_cycle = (
            cycle if state.scenario is AdvancedScenario.IDEAL
            else int(state.ocm.objects[key].ready_cycle or cycle)
        )
        if matrix == "gate_up" and group.tile_id == 0:
            expert.record.first_tile_ready_time = ready_cycle
            expert.record.initial_memory_stall = max(0, ready_cycle - state.router_resolve_cycle)
        wait = max(0, cycle - (expert.weight_wait_start if expert.weight_wait_start is not None else cycle))
        if wait:
            expert.record.total_aie_wait_for_weights += wait
            if group.tile_id > 0 or matrix == "down":
                expert.record.inter_tile_starvation_cycles += wait
        expert.weight_wait_start = None
        operation_count = group.gate_up_ops if matrix == "gate_up" else group.down_ops
        task = AIEPoolTask(
            matrix, operation_count, state.config.aie_units_per_expert,
            state.event.layer_id, expert.expert_id, group.tile_id,
        )
        start, end = state.aie.reserve(task, cycle)
        expert.compute_busy = True
        expert.stage = matrix
        if expert.record.compute_start_time is None:
            expert.record.compute_start_time = start
        if state.scenario is not AdvancedScenario.IDEAL:
            state.ocm.set_running(key, True)
        state.timeline.append(TimelineEntry(
            "AIE_POOL", f"{matrix}_e{expert.expert_id}_f{group.tile_id}_u{state.config.aie_units_per_expert}",
            start, end, state.event.request_id, state.event.layer_id, expert.expert_id,
            tile_id=group.tile_id, tile_size_bytes=(
                group.gate_up_bytes if matrix == "gate_up" else group.down_bytes
            ), compute_ops=operation_count, ready_cycle=ready_cycle,
        ))
        state.queue.push(end, EventType.AIE_OPERATION_DONE, {
            "kind": "expert", "expert_id": expert.expert_id, "matrix": matrix,
        })

    def _nonlinear_done(self, state: _AdvancedState, event: Event) -> None:
        expert = state.experts[event.payload["expert_id"]]
        expert.stage = "waiting_down"
        expert.weight_wait_start = event.cycle
        self._try_progress_expert(state, expert, event.cycle)

    def _build_result(
        self, state: _AdvancedState, makespan: int
    ) -> AdvancedSimulationResult:
        true_ids = set(state.experts)
        candidate_ids = tuple(candidate.expert_id for candidate in state.candidates)
        candidate_set = set(candidate_ids)
        predictive = bool(state.predictor_output)
        stats = batch_workload_stats(state.event)
        records = [state.experts[expert].record for expert in sorted(state.experts)]
        starts = [int(record.compute_start_time or makespan) for record in records]
        true_weight_bytes = len(true_ids) * expert_weight_bytes(self.model)
        useful_ops = sum(record.compute_ops for record in records)
        configured_units = min(
            self.profile.total_aie_units,
            state.config.aie_units_per_expert * state.config.max_parallel_experts,
        )
        roofline = calculate_roofline_metrics(
            useful_ops, true_weight_bytes, makespan, self.profile,
            configured_aie_units=max(1, configured_units),
            balance_tolerance=state.config.roofline_balance_tolerance,
        )
        memory_metrics = state.memory.metrics(makespan)
        aie_metrics = state.aie.metrics(makespan)
        overlap = self._interval_overlap_cycles(
            [(interval.start_cycle, interval.end_cycle) for interval in state.aie.intervals],
            [(interval.start_cycle, interval.end_cycle)
             for channel in state.memory.channels for interval in channel.intervals],
        )
        for record in records:
            record.streaming_overlap_cycles = self._interval_overlap_cycles(
                [(interval.start_cycle, interval.end_cycle) for interval in state.aie.intervals
                 if interval.task.expert_id == record.expert_id],
                [(interval.start_cycle, interval.end_cycle)
                 for channel in state.memory.channels for interval in channel.intervals],
            )
        prediction_ready = (
            state.predictor_output.prediction_ready_cycle if state.predictor_output else None
        )
        window = max(0, state.router_resolve_cycle - prediction_ready) if prediction_ready is not None else 0
        window_values = self._prefetch_window_metrics(state, prediction_ready, state.router_resolve_cycle)
        prefix_values = [state.candidate_depths.get(expert, (0, 0))[0] for expert in candidate_ids]
        weighted_prefix_numerator = sum(
            state.candidate_depths.get(expert, (0, 0))[0] * state.experts[expert].token_count
            for expert in true_ids if expert in state.candidate_depths
        )
        ranks = [candidate_ids.index(expert) for expert in true_ids if expert in candidate_set]
        rank_stats = {
            "min": min(ranks) if ranks else None,
            "mean": fmean(ranks) if ranks else None,
            "median": median(ranks) if ranks else None,
            "max": max(ranks) if ranks else None,
        }
        startable = state.startable_at_resolve
        state.memory.refresh_speculative_accounting()
        result = AdvancedSimulationResult(
            state.scenario.value, state.event.request_id, state.event.layer_id,
            state.router_resolve_cycle, min(starts), max(starts), makespan,
            stats, records, stats.tokens_per_expert,
            len(candidate_set & true_ids) / len(true_ids) if predictive else None,
            bool(candidate_set & true_ids) if predictive else None,
            true_ids <= candidate_set if predictive else None,
            len(startable) / len(true_ids),
            sum(state.experts[expert].token_count for expert in startable) / stats.token_expert_assignments,
            bool(startable), startable == true_ids,
            len(state.prefetched_true_experts) / len(true_ids),
            state.resident_true_bytes_at_resolve / true_weight_bytes,
            len(state.full_resident_at_resolve) / len(true_ids), rank_stats,
            fmean(prefix_values) if prefix_values else 0.0,
            weighted_prefix_numerator / stats.token_expert_assignments,
            memory_metrics, aie_metrics,
            sum(record.total_aie_wait_for_weights for record in records),
            sum(interval.end_cycle - interval.start_cycle for interval in state.aie.intervals),
            overlap, prediction_ready, window,
            window_values[0], window_values[1], window_values[2],
            window_values[3], window_values[4],
            state.ocm.peak_ocm_bytes, state.ocm.average_ocm_bytes(makespan),
            state.ocm.rejected_prefetch_bytes, state.ocm.evicted_speculative_bytes,
            state.demand_blocked_cycles, state.speculative_blocked_cycles,
            roofline, self.profile.metadata(), state.timeline,
        )
        readiness = state.startup_readiness_at_resolve
        gaps = state.startup_gap_bytes_at_resolve
        result.startup_readiness = readiness
        result.startup_gap_bytes = gaps
        result.mean_startup_readiness = fmean(readiness.values()) if readiness else 0.0
        result.weighted_startup_readiness = sum(
            readiness[expert] * state.experts[expert].token_count for expert in readiness
        ) / stats.token_expert_assignments if readiness else 0.0
        result.total_startup_gap_bytes = sum(gaps.values())
        speculative = [
            request for request in state.requests
            if request.request_type == "speculative" and request.started
        ]
        logical = {(request.expert_id, tile_id) for request in speculative for tile_id in request.tile_ids}
        useful_logical = {
            (request.expert_id, tile_id) for request in speculative if request.useful is True
            for tile_id in request.tile_ids
        }
        wasted_logical = logical - useful_logical
        result.tile_metrics = {
            "prefetched_tiles": len(logical),
            "useful_prefetched_tiles": len(useful_logical),
            "wasted_prefetched_tiles": len(wasted_logical),
            "prefetched_bytes": memory_metrics["speculative_bytes"],
            "useful_prefetched_bytes": memory_metrics["useful_speculative_bytes"],
            "wasted_prefetched_bytes": memory_metrics["wasted_speculative_bytes"],
            "prefetch_efficiency": (
                memory_metrics["useful_speculative_bytes"] / memory_metrics["speculative_bytes"]
                if memory_metrics["speculative_bytes"] else 0.0
            ),
            "tile_startup_latency_cycles": min(starts) - state.router_resolve_cycle,
            "memory_stall_cycles": result.aie_starvation_cycles,
            "compute_idle_aie_unit_cycles": aie_metrics["aie_idle_cycles"],
            "candidate_tile_coverage": (
                len(logical) / sum(len(self._geometry_groups(state, candidate.expert_id)) for candidate in state.candidates)
                if state.candidates else 0.0
            ),
        }
        allocated_requests = [
            request for request in state.requests if request.request_type == "speculative"
        ]
        useful_transfers = [request for request in allocated_requests if request.started and request.useful is True]
        wasted_transfers = [request for request in allocated_requests if request.started and request.useful is False]
        canceled_transfers = [request for request in allocated_requests if request.canceled and not request.started]
        result.tile_metrics.update({
            "allocated_tile_transfers": len(allocated_requests),
            "useful_tile_transfers": len(useful_transfers),
            "wasted_tile_transfers": len(wasted_transfers),
            "cancelled_tile_transfers": len(canceled_transfers),
            "allocated_transfer_bytes": sum(request.transfer_bytes for request in allocated_requests),
            "cancelled_transfer_bytes": sum(request.transfer_bytes for request in canceled_transfers),
        })
        if state.tile_prefetch_plan is not None:
            result.tile_prefetch_plan = {
                "scheduler": state.tile_prefetch_plan.scheduler_name,
                "budget_bytes": state.tile_prefetch_plan.budget_bytes,
                "allocated_bytes": state.tile_prefetch_plan.allocated_bytes,
                "allocated_tiles": state.tile_prefetch_plan.allocated_tiles,
                "depths_by_expert": state.tile_prefetch_plan.depths_by_expert,
                "dma_tile_order": tuple(
                    (item.tile.expert_id, item.tile.tile_id)
                    for item in state.tile_prefetch_plan.transfers
                ),
            }
        return result

    @staticmethod
    def _interval_overlap_cycles(
        first: list[tuple[int, int]], second: list[tuple[int, int]],
    ) -> int:
        points = sorted({point for interval in first + second for point in interval})
        return sum(
            right - left for left, right in zip(points, points[1:])
            if any(start < right and end > left for start, end in first)
            and any(start < right and end > left for start, end in second)
        )

    def _prefetch_window_metrics(
        self, state: _AdvancedState, start: int | None, end: int,
    ) -> tuple[int, int, int, float, float]:
        if start is None or end <= start:
            return 0, 0, 0, 0.0, 0.0
        transferred = 0.0
        useful = 0.0
        for request in state.requests:
            if request.request_type != "speculative" or request.start_cycle is None or request.end_cycle is None:
                continue
            payload_start = request.start_cycle + self.profile.dma_setup_cycles
            overlap = max(0, min(end, request.end_cycle) - max(start, payload_start))
            bytes_in_window = min(
                request.transfer_bytes,
                overlap * self.profile.effective_bytes_per_cycle_per_channel * len(request.channel_ids),
            )
            transferred += bytes_in_window
            if request.useful is True:
                useful += bytes_in_window
        capacity_cycles = (end - start) * len(state.memory.channels)
        busy = sum(
            max(0, min(end, interval.end_cycle) - max(start, interval.start_cycle))
            for channel in state.memory.channels for interval in channel.intervals
        )
        capacity_bytes = (
            (end - start) * self.profile.effective_bytes_per_cycle_per_channel
            * len(state.memory.channels)
        )
        return (
            int(transferred), int(useful), max(0, capacity_cycles - busy),
            transferred / capacity_bytes if capacity_bytes else 0.0,
            useful / capacity_bytes if capacity_bytes else 0.0,
        )
