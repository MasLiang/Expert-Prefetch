"""Unified M5-causal/M6-resource multi-layer event simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Mapping, Sequence

from .advanced_schema import AdvancedExecutionConfig
from .prefetch_config import make_prefetch_scheduler
from .batch_predictor import BatchPredictor
from .batch_routing import BatchRoutingEvent
from .event_queue import EventQueue
from .events import Event, EventType
from .f_tile import FTileGroup, build_f_tile_groups
from .hardware.aie_pool import AIEPool, AIEPoolTask
from .hardware.capacity_ocm import CapacityOnChipMemory
from .hardware.multi_channel_memory import MemoryRequest, MultiChannelMemory
from .hardware.nonlinear import NonlinearEngine, NonlinearTask
from .hardware_profile import HardwareProfile
from .schema import CandidatePrefetch, ModelConfig, NonlinearConfig, ValidationError
from .unified_schema import (
    UnifiedLayerResult, UnifiedLayerSequence, UnifiedPredictionRecord,
    UnifiedPredictionSpec, UnifiedSequenceResult,
)
from .work_model import expert_weight_bytes, router_ops
from .tile_scheduler import (
    CurrentTimelineState, ExpertTile, TileAllocationScheduler,
    TilePredictionResult, TileSchedulingHardware,
)

Key = tuple[int, int, str, int]

@dataclass
class _Expert:
    expert_id: int; token_count: int; groups: tuple[FTileGroup, ...]
    tile_id: int = 0; stage: str = "waiting_gu"; launched: bool = False
    done: bool = False; compute_busy: bool = False; wait_start: int | None = None
    start_cycle: int | None = None; finish_cycle: int | None = None
    starvation: int = 0
    memory_ready_cycle: int | None = None; aie_request_cycle: int | None = None
    aie_ready_cycle: int | None = None; gu_start_cycle: int | None = None
    gu_finish_cycle: int | None = None; nonlinear_finish_cycle: int | None = None
    down_start_cycle: int | None = None

@dataclass
class _Layer:
    event: BatchRoutingEvent; experts: dict[int, _Expert]
    started: bool = False; resolved: bool = False; done: bool = False
    start_cycle: int = 0; resolve_cycle: int = 0; done_cycle: int = 0
    active: set[int] = field(default_factory=set)
    tile_owner: int | None = None
    route_pending: bool = False
    completion_pending: bool = False
    demands_released: bool = False
    scheduler_ready_cycle: int = 0
    startable_at_resolve: set[int] = field(default_factory=set)
    startup_readiness: dict[int, float] = field(default_factory=dict)
    resident_true_before_resolve: int = 0
    startup_progress: dict[int, float] = field(default_factory=dict)
    startup_inflight_bytes: dict[int, int] = field(default_factory=dict)

@dataclass
class _State:
    sequence: UnifiedLayerSequence; config: AdvancedExecutionConfig
    queue: EventQueue; memory: MultiChannelMemory; aie: AIEPool
    ocm: CapacityOnChipMemory; nonlinear: NonlinearEngine
    layers: dict[int, _Layer]; predictions: tuple[UnifiedPredictionSpec, ...]
    predictors: Mapping[tuple[int, int], BatchPredictor]
    tile_prefetch_scheduler: TileAllocationScheduler | None = None
    prefetch_fragment_sequence: int = 0
    requests: list[MemoryRequest] = field(default_factory=list)
    by_key: dict[Key, MemoryRequest] = field(default_factory=dict)
    prediction_records: dict[tuple[int, int], UnifiedPredictionRecord] = field(default_factory=dict)
    prediction_outputs: dict[tuple[int, int], object] = field(default_factory=dict)
    triggered: set[tuple[int, int]] = field(default_factory=set)
    current_cycle: int = 0; inference_done: int = 0
    demand_blocked: int = 0; trace_enabled: bool = False
    trace: list[dict[str, object]] = field(default_factory=list)
    inter_fixed_done: dict[tuple[int, int], int] = field(default_factory=dict)
    inter_background_done: set[tuple[int, int]] = field(default_factory=set)
    inter_started: set[tuple[int, int]] = field(default_factory=set)
    inter_retirement_done: dict[tuple[int, int], int] = field(default_factory=dict)
    blocking_records: list[dict[str, object]] = field(default_factory=list)
    deferred_predictions: dict[tuple[int, int], UnifiedPredictionSpec] = field(default_factory=dict)
    pending_prefetch_pairs: list[tuple[Key, int, int, float, int, tuple[int, int]]] = field(default_factory=list)
    admitted_pairs: set[tuple[int, int, int]] = field(default_factory=set)
    credit_wakeup_cycles: set[int] = field(default_factory=set)
    resolver_entries: dict[int, MemoryRequest] = field(default_factory=dict)
    commit_ready_cycles: dict[Key, int] = field(default_factory=dict)
    resolver_available_cycles: dict[int, int] = field(default_factory=dict)
    resolver_blocked_requests: dict[int, int] = field(default_factory=dict)
    prefetch_next_pair_cycle: int = 0

class UnifiedSequenceSimulator:
    """Run batch/Top-K MoE layers on one queue and one shared M6 machine."""

    def __init__(self, model: ModelConfig, profile: HardwareProfile,
                 nonlinear: NonlinearEngine | None = None) -> None:
        self.model, self.profile, self.nonlinear_override = model, profile, nonlinear

    def new_resource_state(self, sequence: UnifiedLayerSequence,
                           predictions: tuple[UnifiedPredictionSpec, ...],
                           config: AdvancedExecutionConfig | None = None) -> object:
        """Return a validated shared resource bundle for inspection/tests."""
        ids={layer.layer_id for layer in sequence.layers}
        if any(spec.source_layer_id not in ids or spec.target_layer_id not in ids for spec in predictions):
            raise ValidationError("prediction source/target must belong to sequence")
        from types import SimpleNamespace
        rules = config.execution_rules if config is not None else None
        return SimpleNamespace(queue=EventQueue(), memory=MultiChannelMemory(self.profile,
            config.placement_policy if config is not None else 'round_robin_expert',
            pair_fifo=rules is not None and rules.request_order == 'pair_fifo', rules=rules),
            aie=AIEPool(self.profile), ocm=CapacityOnChipMemory(self.profile.ocm_bytes, rules),
            nonlinear=self.nonlinear_override, predictions=list(predictions))

    def simulate(self, sequence: UnifiedLayerSequence, config: AdvancedExecutionConfig,
                 *, predictions: tuple[UnifiedPredictionSpec, ...] = (),
                 predictors: Mapping[tuple[int, int], BatchPredictor] | None = None,
                 trace_enabled: bool = False,
                 tile_prefetch_scheduler: TileAllocationScheduler | None = None) -> UnifiedSequenceResult:
        """Execute a causal sequence; prediction ready time derives from source events."""
        for event in sequence.layers: event.validate(self.model)
        rules = config.execution_rules
        if tile_prefetch_scheduler is not None and config.prefetch.allocator != 'legacy':
            raise ValidationError('choose configured allocator or scheduler object, not both')
        tile_prefetch_scheduler = tile_prefetch_scheduler or make_prefetch_scheduler(
            config.prefetch, config.placement_policy, self.profile.num_memory_channels)
        if config.prefetch.dma_order != 'legacy' and tile_prefetch_scheduler is None:
            raise ValidationError('explicit DMA order requires an explicit allocator')
        if tile_prefetch_scheduler is not None and (
            rules.ocm_slot_bytes or rules.descriptor_entries or rules.dma_fifo_depth
        ):
            if not rules.atomic_tile_pair:
                raise ValidationError('byte OCM adapter or atomic pair admission required')
            if config.prefetch.dma_order == 'up_first':
                raise ValidationError('atomic pair admission requires pair_first DMA order')
        if rules.wait_full_expert:
            groups = (self.model.intermediate_size + config.f_tile_size - 1) // config.f_tile_size
            required = 2 * groups * rules.ocm_slot_bytes if rules.ocm_slot_bytes else expert_weight_bytes(self.model)
            if required > self.profile.ocm_bytes:
                raise ValidationError('wait_full_expert requires OCM to fit all Expert GU/DOWN weights')
        if rules.compute_policy == 'tile_ready_first' and (
            self.profile.total_aie_units != 1 or config.aie_units_per_expert != 1 or
            config.max_parallel_experts != 1 or config.startup_gu_tiles != 1):
            raise ValidationError('tile_ready_first 当前支持单共享 Engine；不能用 Expert 数替代物理资源数')
        if rules.retry_future_prefetch and not rules.atomic_tile_pair:
            raise ValidationError('future pair 重试当前要求 atomic_tile_pair')
        ids = {event.layer_id for event in sequence.layers}
        if config.demand_loading != 'tile_streaming':
            if config.max_parallel_experts != 1 or rules.compute_policy != 'expert_locked' or rules.atomic_tile_pair:
                raise ValidationError('non-streaming modes require one sequential Expert, non-atomic expert_locked execution')
            if predictions and config.demand_loading == 'tile_serial':
                raise ValidationError('tile_serial baseline requires no prediction')
            if predictions and any(spec.prefetch_policy != 'full_expert_prefetch' for spec in predictions):
                raise ValidationError('full_expert_load currently supports full-expert prediction only')
            if config.startup_gu_tiles != 1 or rules.ocm_slot_bytes or rules.descriptor_entries or rules.dma_fifo_depth:
                raise ValidationError('weak demand baselines require startup_gu_tiles=1 and byte OCM with unlimited request credits')
            if config.demand_loading == 'full_expert_load' and expert_weight_bytes(self.model) > self.profile.ocm_bytes:
                raise ValidationError('full_expert_load requires OCM to fit one complete Expert')
        if any(spec.source_layer_id not in ids or spec.target_layer_id not in ids for spec in predictions):
            raise ValidationError("prediction source/target must belong to sequence")
        predictor_map = dict(predictors or {})
        if any((spec.source_layer_id, spec.target_layer_id) not in predictor_map for spec in predictions):
            raise ValidationError("every prediction spec requires a source-target predictor")
        layers = {}
        for event in sequence.layers:
            counts = event.expert_token_counts()
            if rules.compute_policy == 'tile_ready_first' or rules.request_order == 'pair_fifo':
                router_order = dict.fromkeys(expert for route in event.routes for expert in route)
                counts = {expert: counts[expert] for expert in router_order}
            experts = {expert: _Expert(expert, count, build_f_tile_groups(
                self.model, expert, count, config.f_tile_size))
                for expert, count in counts.items()}
            layers[event.layer_id] = _Layer(event, experts)
        state = _State(sequence, config, EventQueue(), MultiChannelMemory(
            self.profile, config.placement_policy, pair_fifo=rules.request_order == 'pair_fifo', rules=rules), AIEPool(self.profile),
            CapacityOnChipMemory(self.profile.ocm_bytes, rules), self.nonlinear_override or NonlinearEngine(
                NonlinearConfig(config.nonlinear_elements_per_cycle,
                                config.nonlinear_startup_cycles)), layers,
            predictions, predictor_map, trace_enabled=trace_enabled,
            tile_prefetch_scheduler=tile_prefetch_scheduler)
        state.queue.push(0, EventType.LAYER_START, {"layer_id": sequence.layers[0].layer_id})
        while state.queue:
            event = state.queue.pop(); state.current_cycle = event.cycle
            self._trace(state, event)
            self._handle(state, event)
        if not all(layer.done for layer in layers.values()):
            raise RuntimeError("unified sequence ended before required computation")
        inference = max(layer.done_cycle for layer in layers.values())
        drain = max([inference, state.memory.final_retirement_cycle] + [request.end_cycle or 0 for request in state.requests if request.started])
        results = {layer_id: self._layer_result(state, layer) for layer_id, layer in layers.items()}
        for key, record in state.prediction_records.items():
            target = layers[key[1]]; record.target_layer_start_cycle = target.start_cycle
            record.target_router_resolve_cycle = target.resolve_cycle
            lead = max(0, record.realized_lead_cycles)
            available = lead * self.profile.effective_memory_bandwidth_bytes_per_sec / self.profile.clock_hz
            record.theoretical_bytes_in_lead_window = available
            record.prefetch_opportunity_expert_equivalents = available / expert_weight_bytes(self.model)
            related = [r for r in state.requests if getattr(r, "prediction_key", None) == key]
            record.speculative_bytes = sum(r.transfer_bytes for r in related if r.started)
            record.useful_bytes = sum(r.transfer_bytes for r in related if r.started and r.useful is True)
            record.wasted_bytes = sum(r.transfer_bytes for r in related if r.started and r.useful is False)
            record.canceled_bytes = sum(r.transfer_bytes for r in related if r.canceled)
            before = [r for r in related if r.end_cycle is not None and r.end_cycle <= target.resolve_cycle]
            record.actual_speculative_bytes_before_router = sum(r.transfer_bytes for r in before)
            record.useful_true_bytes_before_router = sum(r.transfer_bytes for r in before if r.useful is True)
            record.cancelled_before_issue_speculative_bytes = sum(r.transfer_bytes for r in related if r.canceled and not r.started)
            record.issued_speculative_bytes = sum(r.transfer_bytes for r in related if r.started)
            record.completed_speculative_bytes = sum(r.transfer_bytes for r in related if r.end_cycle is not None and r.end_cycle <= drain)
            record.completed_speculative_bytes_before_router = record.actual_speculative_bytes_before_router
            active = [r for r in related if r.started and r.start_cycle is not None and r.start_cycle <= target.resolve_cycle and (r.end_cycle is None or target.resolve_cycle < r.end_cycle)]
            record.partial_inflight_speculative_bytes_at_router = sum(self._transferred_bytes_at(r,target.resolve_cycle) for r in active)
            record.physical_speculative_bytes_transferred_by_router = record.completed_speculative_bytes_before_router + record.partial_inflight_speculative_bytes_at_router
            record.useful_completed_speculative_bytes_before_router = record.useful_true_bytes_before_router
            record.useful_physical_speculative_bytes_by_router = record.useful_completed_speculative_bytes_before_router + sum(self._transferred_bytes_at(r,target.resolve_cycle) for r in active if r.useful is True)
            record.wasted_physical_speculative_bytes = record.wasted_bytes
            record.pending_unissued_speculative_bytes = max(0,record.initial_allocated_speculative_bytes-record.cancelled_before_issue_speculative_bytes-record.issued_speculative_bytes)
        memory_metrics = state.memory.metrics(max(1, drain), include_fifo_timeline=(trace_enabled
            and (rules.fifo_push_blocks_dequeue or rules.layer_retirement_wait_for_ingress)))
        memory_metrics["background_bytes"] = sum(
            request.transfer_bytes for request in state.requests
            if request.request_type == "background" and request.started)
        critical=sum(int(r["blocked_cycles"]) for r in state.blocking_records
                     if layers[int(r["blocked_request_layer"])].experts[int(r["blocked_expert"])].finish_cycle == self._critical_finish(layers[int(r["blocked_request_layer"])]))
        for record in state.blocking_records:
            record["was_blocked_request_on_final_critical_path"] = layers[int(record["blocked_request_layer"])].experts[int(record["blocked_expert"])].finish_cycle == self._critical_finish(layers[int(record["blocked_request_layer"])])
        request_records=[]
        for request in state.requests:
            if request.request_type!="speculative":continue
            target=layers[request.layer_id];at_router=self._transferred_bytes_at(request,target.resolve_cycle) if request.started else 0
            request_records.append({"request_id":request.request_id,"source_layer":getattr(request,"source_layer_id",None),"target_layer":request.layer_id,"expert_id":request.expert_id,"matrix_type":request.matrix_type,"tile_id":request.tile_ids[0],"request_bytes":request.transfer_bytes,"queue_cycle":request.issue_cycle,"issue_cycle":request.start_cycle,"completion_cycle":request.end_cycle,"cancelled_cycle":target.resolve_cycle if request.canceled else None,"bytes_transferred_at_target_router":at_router,"true_after_router":request.useful,"committed":request.committed,"cancelled_before_issue":request.canceled and not request.started})
        return UnifiedSequenceResult(
            makespan_cycles=inference, layer_results=results,
            predictions=list(state.prediction_records.values()),
            demand_blocked_by_future_speculation_cycles=state.demand_blocked,
            peak_ocm_bytes=state.ocm.peak_ocm_bytes,
            rejected_future_prefetch_bytes=state.ocm.rejected_prefetch_bytes,
            evicted_future_speculative_bytes=state.ocm.evicted_speculative_bytes,
            final_resource_idle_cycle=drain,
            memory_metrics=memory_metrics,
            aie_metrics=state.aie.metrics(max(1, inference)), trace=state.trace,
            blocking_records=tuple(state.blocking_records),
            critical_path_future_speculation_blocking_cycles=critical,
            speculative_request_records=tuple(request_records),
        )

    def _handle(self, s: _State, e: Event) -> None:
        if e.event_type is EventType.LAYER_START: self._layer_start(s,e)
        elif e.event_type is EventType.AIE_OPERATION_DONE: self._aie_done(s,e)
        elif e.event_type is EventType.PREDICTOR_READY: self._prediction_ready(s,e)
        elif e.event_type is EventType.DMA_DONE: self._dma_done(s,e)
        elif e.event_type is EventType.NONLINEAR_DONE: self._nonlinear_done(s,e)
        elif e.event_type is EventType.PREFETCH_ALLOCATION_READY:
            self._retry_prefetch_pairs(s, e.cycle)
            self._kick_memory(s, e.cycle)
        elif e.event_type is EventType.INTER_LAYER_START: self._inter_layer_start(s,e)
        elif e.event_type is EventType.INTER_LAYER_FIXED_DONE: self._try_start_inter_target(s,(e.payload["source_layer_id"],e.payload["target_layer_id"]),e.cycle)
        elif e.event_type is EventType.INTER_LAYER_RETIRE_CHECK: self._retirement_check(s, e)
        elif e.event_type is EventType.INGRESS_DONE:
            self._append_trace(s, e.cycle, 'ingress_idle', e.payload['layer_id'], resource='INGRESS')
            for layer in s.layers.values():
                if layer.resolved and not layer.done:
                    self._refill_demands(s, layer, e.cycle)
            self._retry_prefetch_pairs(s, e.cycle)
            self._kick_memory(s, e.cycle)
        elif e.event_type is EventType.SCHEDULER_INVOKE: self._schedule(s,e.cycle,e.payload.get("layer_id"))
        elif e.event_type is EventType.ROUTER_START: self._router_start(s, e)
        elif e.event_type is EventType.ROUTER_DONE: self._router_resolve(s, e)
        elif e.event_type is EventType.LAYER_COMBINE_DONE: self._finish_layer(s, s.layers[e.payload['layer_id']], e.cycle)
        elif e.event_type is EventType.DISPATCH_DONE:
            layer = s.layers[e.payload['layer_id']]
            if (s.config.execution_rules.route_commit_visibility_cycles
                    and any(r.layer_id == layer.event.layer_id for r in s.resolver_entries.values())):
                s.queue.push(e.cycle + 1, EventType.DISPATCH_DONE, e.payload)
                return
            layer.demands_released = True
            self._refill_demands(s, layer, e.cycle)
        elif e.event_type is EventType.CONTROL_CREDIT_READY:
            s.credit_wakeup_cycles.discard(e.cycle)
            self._kick_memory(s, e.cycle)
        elif e.event_type is EventType.DMA_COMMIT:
            request = e.payload['request']
            blocked = s.resolver_blocked_requests.get(request.layer_id)
            if blocked is not None and blocked != request.request_id:
                s.queue.push(e.cycle + 1, EventType.DMA_COMMIT, e.payload)
                return
            available = s.resolver_available_cycles.get(request.layer_id, e.cycle)
            if e.cycle < available:
                s.queue.push(available, EventType.DMA_COMMIT, e.payload)
                return
            if not s.memory.promote(request, e.cycle):
                # 有限promotion FIFO：保持resolver token，不能丢弃或提前启动计算。
                for key in request.object_keys:
                    s.commit_ready_cycles[key] = e.cycle + 1
                s.queue.push(e.cycle + 1, EventType.DMA_COMMIT, e.payload)
                s.resolver_available_cycles[request.layer_id] = e.cycle + 1
                s.resolver_blocked_requests[request.layer_id] = request.request_id
                self._kick_memory(s, e.cycle)
                return
            s.resolver_available_cycles[request.layer_id] = e.cycle + s.config.execution_rules.route_scan_cycles_per_entry
            s.resolver_blocked_requests.pop(request.layer_id, None)
            for key in request.object_keys:
                s.commit_ready_cycles[key] = e.cycle
            for key in request.object_keys:
                if key in s.ocm.objects:
                    s.ocm.commit(key)
                    self._append_trace(s, e.cycle, 'descriptor_commit_visible',
                                       key[0], key[1], key[3], key[2], 'CONTROLLER')
            s.resolver_entries.pop(e.payload['resolver_index'], None)
            layer = s.layers[request.layer_id]
            s.queue.push(e.cycle, EventType.SCHEDULER_INVOKE, {'layer_id': request.layer_id})
            for expert in layer.experts.values():
                if expert.launched and not expert.done:
                    self._progress(s, layer, expert, e.cycle)
        elif e.event_type is EventType.DMA_CANCEL:
            request = e.payload['request']
            if request.layer_id in s.resolver_blocked_requests:
                s.queue.push(e.cycle + 1, EventType.DMA_CANCEL, e.payload)
                return
            available = s.resolver_available_cycles.get(request.layer_id, e.cycle)
            if e.cycle < available:
                s.queue.push(available, EventType.DMA_CANCEL, e.payload)
                return
            s.resolver_available_cycles[request.layer_id] = e.cycle + s.config.execution_rules.route_scan_cycles_per_entry
            s.resolver_entries.pop(e.payload['resolver_index'], None)
            self._cancel_resolved_request(s, request, e.cycle)
            self._append_trace(s, e.cycle, 'descriptor_cancel_visible', request.layer_id,
                               request.expert_id, request.tile_ids[0], request.matrix_type,
                               'CONTROLLER')
            s.memory.retire_descriptors(e.cycle)
            self._retry_prefetch_pairs(s, e.cycle)
            self._kick_memory(s, e.cycle)
        elif e.event_type is EventType.AIE_OPERATION_START:
            layer = s.layers[e.payload['layer_id']]
            expert = layer.experts[e.payload['expert_id']]
            matrix = e.payload['matrix']
            key = self._key(layer.event.layer_id, expert.expert_id, matrix, expert.tile_id)
            if key in s.commit_ready_cycles and s.ocm.objects[key].state != 'committed':
                s.queue.push(e.cycle + 1, EventType.AIE_OPERATION_START, e.payload)
                return
            expert.compute_busy = False
            expert.stage = 'waiting_gu' if matrix == 'gate_up' else 'waiting_down'
            self._progress(s, layer, expert, e.cycle, dispatch_paid=True)
        elif e.event_type is EventType.OCM_PAYLOAD_READY:
            request = e.payload['request']
            for key in request.object_keys:
                if key in s.ocm.objects:
                    s.ocm.mark_resident(key, e.cycle)
                    self._append_trace(s, e.cycle, 'payload_ready', key[0], key[1],
                                       key[3], key[2], 'OCM')
                    if request.useful is False and s.layers[key[0]].resolved:
                        s.ocm.release(key, e.cycle)
            layer = s.layers[request.layer_id]
            if layer.resolved:
                s.queue.push(e.cycle, EventType.SCHEDULER_INVOKE, {'layer_id': request.layer_id})
                for expert in layer.experts.values():
                    if expert.launched and not expert.done:
                        self._progress(s, layer, expert, e.cycle)
            self._refill_demands(s, layer, e.cycle)
            self._retry_prefetch_pairs(s, e.cycle)
            self._kick_memory(s, e.cycle)
        else: raise RuntimeError(f"unsupported unified event {e.event_type}")

    def _layer_start(self, s: _State, e: Event) -> None:
        layer=s.layers[e.payload["layer_id"]]; layer.started=True; layer.start_cycle=e.cycle
        self._retry_prefetch_pairs(s, e.cycle)
        self._milestone(s,layer,"source_layer_start",e.cycle)
        for spec in s.predictions:
            if spec.source_layer_id==layer.event.layer_id and spec.source_milestone=="custom_offset":
                self._trigger_prediction(s,spec,e.cycle+spec.custom_offset_cycles)
        load = s.config.execution_rules.router_load_cycles
        if load:
            self._append_trace(s, e.cycle, 'router_weight_load', layer.event.layer_id, resource='ROUTER_LOAD', end_cycle=e.cycle + load)
            s.queue.push(e.cycle + load, EventType.ROUTER_START, {'layer_id': layer.event.layer_id})
        else:
            self._router_start(s, e)

    def _router_start(self, s: _State, e: Event) -> None:
        """权重加载完成后预留 Router MAC；常驻时 load=0。"""
        layer = s.layers[e.payload['layer_id']]
        task=AIEPoolTask("router",router_ops(self.model,layer.event.batch_size),s.config.router_aie_units,layer.event.layer_id)
        start,end=s.aie.reserve(task,e.cycle); self._append_trace(s,start,"router_start",layer.event.layer_id,resource="AIE_POOL")
        s.queue.push(end,EventType.AIE_OPERATION_DONE,{"kind":"router","layer_id":layer.event.layer_id})

    def _aie_done(self,s:_State,e:Event)->None:
        layer=s.layers[e.payload["layer_id"]]
        if e.payload["kind"]=="router":
            control = (s.config.execution_rules.router_control_cycles
                       + s.config.execution_rules.router_truth_emit_cycles)
            if control:
                self._append_trace(s, e.cycle, 'router_control', layer.event.layer_id, resource='CONTROLLER', end_cycle=e.cycle + control)
                s.queue.push(e.cycle + control, EventType.ROUTER_DONE, {'layer_id': layer.event.layer_id})
            else:
                self._router_resolve(s, e)
            return
        expert=layer.experts[e.payload["expert_id"]]; group=expert.groups[expert.tile_id]; matrix=e.payload["matrix"]; expert.compute_busy=False
        key=self._key(layer.event.layer_id,expert.expert_id,matrix,group.tile_id); s.ocm.release(key,e.cycle)
        self._append_trace(s, e.cycle, 'slot_release', layer.event.layer_id, expert.expert_id,
                           group.tile_id, matrix, 'OCM')
        self._refill_demands(s,layer,e.cycle)
        if matrix=="gate_up":
            if expert.tile_id==0:expert.gu_finish_cycle=e.cycle
            if group.tile_id==0:self._milestone(s,layer,"source_gate_up_done",e.cycle)
            expert.stage="nonlinear"; task=NonlinearTask(group.nonlinear_elements); start=s.nonlinear.earliest_start(task,e.cycle+s.config.execution_rules.nonlinear_handoff_cycles);end=s.nonlinear.reserve(task,start)
            s.queue.push(end,EventType.NONLINEAR_DONE,{"layer_id":layer.event.layer_id,"expert_id":expert.expert_id})
        else:
            if s.config.execution_rules.compute_policy == 'tile_ready_first':
                layer.tile_owner = None
            expert.tile_id+=1
            if expert.tile_id==len(expert.groups):
                expert.done=True;expert.finish_cycle=e.cycle;layer.active.discard(expert.expert_id);self._milestone(s,layer,"source_expert_done",e.cycle)
                if s.config.demand_loading != 'tile_streaming':
                    self._refill_demands(s, layer, e.cycle)
                layer.scheduler_ready_cycle = max(layer.scheduler_ready_cycle,
                    e.cycle + s.config.execution_rules.expert_combine_cycles)
                if layer.scheduler_ready_cycle > e.cycle:
                    self._append_trace(s, e.cycle, 'expert_combine', layer.event.layer_id,
                                       expert.expert_id, resource='CONTROLLER',
                                       end_cycle=layer.scheduler_ready_cycle)
                    s.queue.push(layer.scheduler_ready_cycle, EventType.SCHEDULER_INVOKE,
                                 {'layer_id': layer.event.layer_id})
                self._maybe_layer_done(s,layer,e.cycle)
            else:
                expert.stage="waiting_gu";expert.wait_start=e.cycle
                self._refill_demands(s,layer,e.cycle);self._progress(s,layer,expert,e.cycle)
            if s.config.execution_rules.compute_policy == 'tile_ready_first':
                s.queue.push(e.cycle, EventType.SCHEDULER_INVOKE, {'layer_id': layer.event.layer_id})
        self._retry_prefetch_pairs(s, e.cycle)
        self._kick_memory(s, e.cycle)

    def _resolve(self,s:_State,layer:_Layer,cycle:int)->None:
        true=set(layer.experts)
        for request in s.requests:
            if request.layer_id!=layer.event.layer_id or request.request_type!="speculative":continue
            if request.expert_id in true:
                index = next((i for i, value in s.resolver_entries.items() if value is request), None)
                rules = s.config.execution_rules
                if index is not None and rules.route_commit_visibility_cycles:
                    visible = (cycle + rules.route_resolver_drain_cycles
                               + index * rules.route_scan_cycles_per_entry
                               + rules.route_commit_visibility_cycles)
                    for key in request.object_keys:
                        s.commit_ready_cycles[key] = visible
                    s.queue.push(visible, EventType.DMA_COMMIT,
                                 {'request': request, 'resolver_index': index})
                else:
                    if not s.memory.promote(request, cycle):
                        raise RuntimeError('Router resolve 未获得 promotion credit')
                    for key in request.object_keys:
                        if key in s.ocm.objects:s.ocm.commit(key)
            else:
                index = next((i for i, value in s.resolver_entries.items() if value is request), None)
                rules = s.config.execution_rules
                if index is not None and rules.route_commit_visibility_cycles:
                    visible = (cycle + rules.route_resolver_drain_cycles
                               + index * rules.route_scan_cycles_per_entry
                               + rules.route_commit_visibility_cycles)
                    s.queue.push(visible, EventType.DMA_CANCEL,
                                 {'request': request, 'resolver_index': index})
                else:
                    self._cancel_resolved_request(s, request, cycle)
        s.memory.retire_descriptors(cycle)
        for expert in layer.experts.values():
            startup=self._startup_keys(s,layer,expert); required=sum(size for _,size in startup);resident=sum(size for key,size in startup if s.ocm.is_resident(key,cycle))
            layer.startup_readiness[expert.expert_id]=resident/required if required else 1.0
            inflight=0
            for key,size in startup:
                request=s.by_key.get(key)
                if request is not None and request.started and request.start_cycle is not None and request.start_cycle<=cycle and (request.end_cycle is None or cycle<request.end_cycle):
                    inflight+=min(size,self._transferred_bytes_at(request,cycle))
            layer.startup_inflight_bytes[expert.expert_id]=inflight
            layer.startup_progress[expert.expert_id]=(resident+inflight)/required if required else 1.0
            if resident==required:layer.startable_at_resolve.add(expert.expert_id)
        rules = s.config.execution_rules
        delay = rules.demand_dispatch_cycles + rules.descriptor_entries * rules.route_scan_cycles_per_entry
        if delay:
            self._append_trace(s, cycle, 'demand_dispatch', layer.event.layer_id, resource='CONTROLLER', end_cycle=cycle + delay)
            s.queue.push(cycle + delay, EventType.DISPATCH_DONE, {'layer_id': layer.event.layer_id})
        else:
            layer.demands_released = True
            self._refill_demands(s,layer,cycle)
        self._kick_memory(s,cycle);s.queue.push(cycle,EventType.SCHEDULER_INVOKE,{"layer_id":layer.event.layer_id})

    def _cancel_resolved_request(self, s: _State, request: MemoryRequest, cycle: int) -> None:
        """取消未开始的错误请求；已开始的DMA保持不可抢占，单位cycle。"""
        request.useful = False
        if not request.started and s.memory.cancel(request):
            for key in request.object_keys:
                s.ocm.release(key, cycle)
            self._append_trace(s, cycle, 'dma_cancel', request.layer_id, request.expert_id,
                               request.tile_ids[0], request.matrix_type, 'MEMORY', 'canceled',
                               getattr(request, 'source_layer_id', None), request.layer_id)
        else:
            for key in request.object_keys:
                s.ocm.mark_false(key)
                if request.end_cycle is not None and request.end_cycle <= cycle:
                    s.ocm.release(key, cycle)

    def _router_resolve(self, s: _State, e: Event) -> None:
        """truth 提交须等 queued hit 的 promoted FIFO credit，不更改真实路由。"""
        layer = s.layers[e.payload['layer_id']]
        layer.route_pending = True
        for request in (() if s.config.execution_rules.route_commit_visibility_cycles else s.requests):
            if request.layer_id == layer.event.layer_id and request.request_type == 'speculative' and request.expert_id in layer.experts:
                if not s.memory.promote(request, e.cycle):
                    self._kick_memory(s, e.cycle)
                    return
        layer.route_pending = False
        layer.resolve_cycle = e.cycle
        layer.resolved = True
        layer.scheduler_ready_cycle = e.cycle + s.config.execution_rules.truth_scheduler_refresh_cycles
        if layer.scheduler_ready_cycle > e.cycle:
            s.queue.push(layer.scheduler_ready_cycle, EventType.SCHEDULER_INVOKE,
                         {'layer_id': layer.event.layer_id})
        self._append_trace(s, e.cycle, 'router_done', layer.event.layer_id, resource='AIE_POOL')
        self._milestone(s, layer, 'source_router_done', e.cycle)
        self._resolve(s, layer, e.cycle)

    def _refill_demands(self,s:_State,layer:_Layer,cycle:int)->None:
        """Retry required tiles that finite OCM could not admit earlier."""
        if not layer.resolved or not layer.demands_released:return
        if s.config.demand_loading != 'tile_streaming':
            pending = [expert for expert in layer.experts.values() if not expert.done]
            if not pending:
                return
            expert = min(pending, key=lambda item: item.expert_id)
            if s.config.demand_loading == 'full_expert_load':
                needed = self._all_keys(layer, expert) if not expert.launched else []
            elif not expert.compute_busy and expert.stage in {'waiting_gu', 'waiting_down'}:
                group = expert.groups[expert.tile_id]
                matrix = 'gate_up' if expert.stage == 'waiting_gu' else 'down'
                needed = [(self._key(layer.event.layer_id, expert.expert_id, matrix, group.tile_id),
                           group.gate_up_bytes if matrix == 'gate_up' else group.down_bytes)]
            else:
                needed = []
            for key, size in needed:
                request = s.by_key.get(key)
                if request is None or request.canceled:
                    self._issue(s, key, size, 'demand', cycle)
            self._kick_memory(s, cycle)
            return
        if s.config.execution_rules.atomic_tile_pair:
            # Router order、Expert 内 prefix，阻塞时不能绕过当前 demand 游标。
            for expert in layer.experts.values():
                for group in expert.groups:
                    pair = (layer.event.layer_id, expert.expert_id, group.tile_id)
                    if pair in s.admitted_pairs:
                        continue
                    key = self._key(*pair[:2], 'gate_up', pair[2])
                    if not self._issue_pair(s, key, group.gate_up_bytes, group.down_bytes, 'demand', cycle):
                        self._kick_memory(s, cycle)
                        return
            self._kick_memory(s, cycle)
            return
        for expert in layer.experts.values():
            if expert.done:continue
            group=expert.groups[expert.tile_id]
            if not expert.launched:
                needed = tuple(self._startup_keys(s, layer, expert)) + ((
                    self._key(layer.event.layer_id, expert.expert_id, "down", group.tile_id),
                    group.down_bytes),)
            else:
                needed=((self._key(layer.event.layer_id,expert.expert_id,"gate_up",group.tile_id),group.gate_up_bytes),
                        (self._key(layer.event.layer_id,expert.expert_id,"down",group.tile_id),group.down_bytes))
            for key,size in needed:
                request=s.by_key.get(key)
                if request is None or request.canceled:self._issue(s,key,size,"demand",cycle)
        self._kick_memory(s,cycle)

    def _prediction_ready(self,s:_State,e:Event)->None:
        spec:UnifiedPredictionSpec=e.payload["spec"];target=s.layers[spec.target_layer_id]
        if target.resolved:return
        if spec.wait_for_previous_layer_weights and not self._previous_weights_done(s,spec,e.cycle):
            s.deferred_predictions[(spec.source_layer_id,spec.target_layer_id)]=spec
            return
        allocation = s.config.execution_rules.prefetch_allocator_cycles
        if allocation and not e.payload.get('allocation_ready', False):
            self._append_trace(s, e.cycle, 'prefetch_allocator', target.event.layer_id,
                               resource='CONTROLLER', end_cycle=e.cycle + allocation)
            s.queue.push(e.cycle + allocation, EventType.PREDICTOR_READY,
                         {'spec': spec, 'allocation_ready': True})
            return
        s.prediction_records[(spec.source_layer_id,spec.target_layer_id)].prefetch_release_cycle=e.cycle
        output=s.prediction_outputs[(spec.source_layer_id,spec.target_layer_id)]
        if output is None:return
        candidates=output.aggregate_candidates(spec.candidate_count)
        planned=0;record=s.prediction_records[(spec.source_layer_id,spec.target_layer_id)]
        if s.tile_prefetch_scheduler is not None:
            self._allocate_prefetch(s, spec, candidates, e.cycle)
            self._kick_memory(s, e.cycle)
            return
        if s.config.execution_rules.atomic_tile_pair:
            if target.started:
                return
            for rank, candidate in enumerate(candidates):
                groups = build_f_tile_groups(self.model, candidate.expert_id, 1, s.config.f_tile_size)
                depth = len(groups) if spec.prefetch_policy == 'full_expert_prefetch' else (
                    min(spec.candidate_depths[rank], len(groups)) if spec.candidate_depths else
                    min(max(0, spec.prefix_depth - rank) if spec.prefetch_policy == 'ranked_prefix' else spec.prefix_depth, len(groups)))
                for group in groups[:depth]:
                    size = group.gate_up_bytes + group.down_bytes
                    if s.config.speculative_byte_budget is not None and planned + size > s.config.speculative_byte_budget:
                        break
                    planned += size
                    key = self._key(target.event.layer_id, candidate.expert_id, 'gate_up', group.tile_id)
                    item = (key, group.gate_up_bytes, group.down_bytes, candidate.score, rank,
                            (spec.source_layer_id, spec.target_layer_id))
                    if s.config.execution_rules.retry_future_prefetch:
                        s.pending_prefetch_pairs.append(item)
                    else:
                        if not self._issue_pair(s, key, group.gate_up_bytes, group.down_bytes, 'speculative',
                                                e.cycle, candidate.score, rank, item[-1]):
                            s.ocm.rejected_prefetch_bytes += size
            record.planned_speculative_bytes = planned
            self._retry_prefetch_pairs(s, e.cycle)
            self._kick_memory(s, e.cycle)
            return
        for rank,candidate in enumerate(candidates):
            groups=target.experts[candidate.expert_id].groups if candidate.expert_id in target.experts else build_f_tile_groups(self.model,candidate.expert_id,1,s.config.f_tile_size)
            if spec.prefetch_policy=="full_expert_prefetch": depth=len(groups)
            elif spec.candidate_depths: depth=min(spec.candidate_depths[rank],len(groups))
            elif spec.prefetch_policy=="ranked_prefix": depth=min(max(0,spec.prefix_depth-rank),len(groups))
            else: depth=min(spec.prefix_depth,len(groups))
            for group in groups[:depth]:
                if s.config.speculative_byte_budget is not None and planned+group.gate_up_bytes>s.config.speculative_byte_budget:continue
                planned+=group.gate_up_bytes
                self._issue(s,self._key(target.event.layer_id,candidate.expert_id,"gate_up",group.tile_id),group.gate_up_bytes,"speculative",e.cycle,candidate.score,rank,(spec.source_layer_id,spec.target_layer_id))
            if spec.prefetch_policy=="full_expert_prefetch":
                for group in groups:
                    if s.config.speculative_byte_budget is not None and planned+group.down_bytes>s.config.speculative_byte_budget:continue
                    planned+=group.down_bytes
                    self._issue(s,self._key(target.event.layer_id,candidate.expert_id,"down",group.tile_id),group.down_bytes,"speculative",e.cycle,candidate.score,rank,(spec.source_layer_id,spec.target_layer_id))
        record.initial_allocated_speculative_bytes=planned
        self._kick_memory(s,e.cycle)

    def _allocate_prefetch(self, s: _State, spec: UnifiedPredictionSpec,
                           candidates: Sequence[CandidatePrefetch], cycle: int) -> None:
        """Allocate logical prefixes, then independently order physical fragments.

        Atomic plans use payload credits; actual OCM slots/FIFO credits are
        checked at pair admission, and pending plans retry only when configured.
        Byte-mode plans retain historical current-free-OCM budgeting.
        """
        geometry = {
            candidate.expert_id: tuple(
                ExpertTile(candidate.expert_id, group.tile_id,
                           group.gate_up_bytes + group.down_bytes,
                           group.gate_up_bytes, group.down_bytes, rank,
                           float(candidate.score))
                for group in build_f_tile_groups(self.model, candidate.expert_id, 1,
                                                s.config.f_tile_size)
            ) for rank, candidate in enumerate(candidates)
        }
        scheduler = s.tile_prefetch_scheduler
        if scheduler is None:
            raise RuntimeError('missing tile allocation scheduler')
        plan = scheduler.schedule(
            TilePredictionResult(tuple((c.expert_id, float(c.score)) for c in candidates)),
            geometry,
            TileSchedulingHardware(self.profile.ocm_bytes,
                self.profile.effective_bytes_per_cycle_per_channel,
                self.profile.num_memory_channels),
            CurrentTimelineState(cycle, 0 if s.config.execution_rules.atomic_tile_pair else s.ocm.current_bytes,
                                 s.config.speculative_byte_budget),
        )
        record = s.prediction_records[(spec.source_layer_id, spec.target_layer_id)]
        record.allocation_policy = type(scheduler).__name__
        record.allocation_budget_bytes = plan.budget_bytes
        record.planned_candidate_depths = tuple(
            plan.depths_by_expert.get(c.expert_id, 0) for c in candidates)
        record.planned_speculative_bytes = plan.allocated_bytes
        atomic = s.config.execution_rules.atomic_tile_pair
        dma_order = s.config.prefetch.dma_order
        if dma_order == 'legacy':
            dma_order = 'pair_first' if atomic else 'up_first'
        record.dma_order = dma_order
        if atomic:
            if s.layers[spec.target_layer_id].started:
                return
            for transfer in plan.transfers:
                tile = transfer.tile
                item = (self._key(spec.target_layer_id, tile.expert_id, 'gate_up', tile.tile_id),
                        tile.gate_up_bytes, tile.down_bytes, tile.confidence,
                        tile.predicted_rank, (spec.source_layer_id, spec.target_layer_id))
                if s.config.execution_rules.retry_future_prefetch:
                    s.pending_prefetch_pairs.append(item)
                else:
                    self._issue_pair(s, *item[:3], 'speculative', cycle, *item[3:])
            self._retry_prefetch_pairs(s, cycle)
            return
        record.initial_allocated_speculative_bytes = plan.allocated_bytes
        fragments = (
            [(matrix, transfer) for matrix in ('gate_up', 'down') for transfer in plan.transfers]
            if dma_order == 'up_first' else
            [(matrix, transfer) for transfer in plan.transfers for matrix in ('gate_up', 'down')])
        order = 0
        for matrix, transfer in fragments:
            tile = transfer.tile
            size = tile.gate_up_bytes if matrix == 'gate_up' else tile.down_bytes
            self._issue(s, self._key(spec.target_layer_id, tile.expert_id,
                        matrix, tile.tile_id), size, 'speculative', cycle,
                        tile.confidence, tile.predicted_rank,
                        (spec.source_layer_id, spec.target_layer_id), order)
            order += 1

    def _issue_pair(self, s: _State, key: Key, gu_bytes: int, down_bytes: int,
                    kind: str, cycle: int, score: float = 0.0, rank: int | None = None,
                    prediction_key: tuple[int, int] | None = None) -> bool:
        """同时保留 GU/DOWN slot；只在完整 pair admission 成功后创建 FIFO 请求。"""
        pair = (key[0], key[1], key[3])
        if pair in s.admitted_pairs:
            return True
        if cycle < s.memory.ingress_available_cycle:
            s.memory.admission_backpressure_count += 1
            return False
        if not s.memory.can_admit(kind, 2, current_cycle=cycle):
            s.memory.admission_backpressure_count += 1
            return False
        if (kind == 'speculative' and s.config.execution_rules.route_commit_visibility_cycles
                and len(s.resolver_entries) + 2 + s.config.execution_rules.reserved_demand_descriptors > s.config.execution_rules.descriptor_entries):
            return False
        down_key = (key[0], key[1], 'down', key[3])
        admission = s.ocm.admit_group(((key, gu_bytes), (down_key, down_bytes)), kind, score, cycle)
        if not admission.admitted:
            return False
        s.admitted_pairs.add(pair)
        ingress_end = s.memory.reserve_ingress_pair(cycle)
        if ingress_end > cycle:
            self._append_trace(s, cycle, 'ingress_pair', key[0], key[1], key[3],
                               resource='INGRESS', request_state=kind, end_cycle=ingress_end)
            s.queue.push(ingress_end, EventType.INGRESS_DONE, {'layer_id': key[0]})
        if kind == 'speculative' and prediction_key is not None:
            s.prediction_records[prediction_key].initial_allocated_speculative_bytes += gu_bytes + down_bytes
        self._append_trace(s, cycle, 'pair_admit', key[0], key[1], key[3],
                           request_state=kind)
        if s.trace_enabled:
            s.trace[-1].update(payload_bytes=gu_bytes + down_bytes,
                allocated_bytes=2 * s.config.execution_rules.ocm_slot_bytes,
                occupied_bytes=s.ocm.current_bytes,
                occupied_bytes_per_bank=[sum(obj.allocated_bytes for obj in s.ocm.objects.values()
                    if obj.bank_id == bank) for bank in range(s.config.execution_rules.ocm_banks)])
        order = s.prefetch_fragment_sequence if kind == 'speculative' and s.tile_prefetch_scheduler is not None else None
        self._issue(s, key, gu_bytes, kind, cycle, score, rank, prediction_key, order)
        self._issue(s, down_key, down_bytes, kind, cycle, score, rank, prediction_key,
                    None if order is None else order + 1)
        if order is not None:
            s.prefetch_fragment_sequence += 2
        return True

    def _retry_prefetch_pairs(self, s: _State, cycle: int) -> None:
        """资源释放后重试未 admission 的计划；目标开始后停止新预取，不重复提交。"""
        if not s.config.execution_rules.retry_future_prefetch:
            return
        if (s.config.execution_rules.block_prefetch_during_layer_retirement
                and not any(layer.started and not layer.done for layer in s.layers.values())):
            return
        pending = (sorted(s.pending_prefetch_pairs, key=lambda v: v[0][0])
                   if s.tile_prefetch_scheduler is not None else
                   sorted(s.pending_prefetch_pairs, key=lambda v: (v[0][0], v[4], v[0][3])))
        s.pending_prefetch_pairs = []
        # 当前 demand 游标占有 ingress，不能因其目标 bank 满而从其他 bank 偷跑。
        # 这是 admission 仲裁约束，不是“上一层权重全部 DMA 完成”屏障。
        demand_owns_ingress = s.config.execution_rules.demand_ingress_blocks_prefetch and any(
            layer.resolved and layer.demands_released and not layer.done and any(
                (layer.event.layer_id, expert.expert_id, group.tile_id) not in s.admitted_pairs
                for expert in layer.experts.values() for group in expert.groups)
            for layer in s.layers.values())
        blocked = demand_owns_ingress or cycle < s.prefetch_next_pair_cycle
        for item in pending:
            key, gu_bytes, down_bytes, score, rank, prediction_key = item
            if s.layers[key[0]].started:
                s.prediction_records[prediction_key].canceled_unadmitted_bytes += gu_bytes + down_bytes
                self._append_trace(s, cycle, 'prefetch_stop', key[0], key[1], key[3],
                                   request_state='unadmitted_canceled')
                continue
            if blocked or not self._issue_pair(s, key, gu_bytes, down_bytes, 'speculative',
                                               cycle, score, rank, prediction_key):
                blocked = True
                s.pending_prefetch_pairs.append(item)
            elif s.config.execution_rules.prefetch_pair_interval_cycles:
                s.prefetch_next_pair_cycle = cycle + s.config.execution_rules.prefetch_pair_interval_cycles
                s.queue.push(s.prefetch_next_pair_cycle, EventType.PREFETCH_ALLOCATION_READY, {})
                blocked = True

    def _previous_weights_done(self,s:_State,spec:UnifiedPredictionSpec,cycle:int)->bool:
        """Check actual completion of every true weight in the preceding layer.

        This check runs on observed demand state after preceding Router resolve;
        it does not inspect unresolved routing truth or forecast future cycles.
        """
        order=[event.layer_id for event in s.sequence.layers]
        previous=s.layers[order[order.index(spec.target_layer_id)-1]]
        if not previous.resolved:return False
        for expert in previous.experts.values():
            for group in expert.groups:
                for matrix in ("gate_up","down"):
                    request=s.by_key.get(self._key(previous.event.layer_id,expert.expert_id,matrix,group.tile_id))
                    if request is None or request.canceled or request.end_cycle is None or request.end_cycle>cycle:
                        return False
        return True

    def _release_deferred_predictions(self,s:_State,cycle:int)->None:
        """Release saved predictions at the observed preceding weight barrier."""
        for key,spec in sorted(tuple(s.deferred_predictions.items())):
            if s.layers[spec.target_layer_id].resolved:
                del s.deferred_predictions[key]
            elif self._previous_weights_done(s,spec,cycle):
                del s.deferred_predictions[key]
                s.queue.push(cycle,EventType.PREDICTOR_READY,{"spec":spec})

    def _issue(self,s:_State,key:Key,size:int,kind:str,cycle:int,score:float=0.0,rank:int|None=None,prediction_key:tuple[int,int]|None=None,schedule_order:int|None=None)->MemoryRequest|None:
        old=s.by_key.get(key)
        if old is not None and not old.canceled:return old
        admission=s.ocm.admit(key,size,"speculative" if kind=="speculative" else "demand",score,cycle)
        for evicted in admission.evicted_keys:
            req=s.by_key.get(evicted)
            if req is not None and not req.started:s.memory.cancel(req)
        if not admission.admitted:return None
        request=s.memory.new_request(layer_id=key[0],expert_id=key[1],matrix_type=key[2],tile_ids=(key[3],),object_keys=(key,),transfer_bytes=size,request_type=kind,issue_cycle=cycle,score=score,candidate_rank=rank,speculative_scope="future_layer" if kind=="speculative" else "")
        request.schedule_order = schedule_order
        request.prediction_key=prediction_key;request.source_layer_id=prediction_key[0] if prediction_key else None;request.target_layer_id=key[0]
        queued=s.memory.enqueue(request);s.by_key[key]=queued
        if queued is request:s.requests.append(request)
        if (queued is request and kind == 'speculative'
                and s.config.execution_rules.route_commit_visibility_cycles):
            index = next((i for i in range(s.config.execution_rules.descriptor_entries)
                          if i not in s.resolver_entries), None)
            if index is None:
                raise RuntimeError('speculative resolver capacity exhausted after admission')
            s.resolver_entries[index] = request
        if kind=="demand":
            blockers={}
            for channel_id in request.channel_ids:
                active=s.memory.channels[channel_id].active_request
                if active is not None and active.request_type=="speculative" and active.burst_end_cycle is not None:blockers.setdefault(active.request_id,(active,[]))[1].append(channel_id)
            for active,channels in blockers.values():
                blocked=max(0,(active.burst_end_cycle or cycle)-cycle);s.demand_blocked+=blocked
                s.blocking_records.append({"blocked_request_id":request.request_id,"blocked_request_layer":request.layer_id,"blocked_expert":request.expert_id,"blocked_matrix":request.matrix_type,"blocked_tile":request.tile_ids[0],"blocking_request_id":active.request_id,"blocking_source_layer":getattr(active,"source_layer_id",None),"blocking_target_layer":active.layer_id,"blocking_expert":active.expert_id,"channel_ids":tuple(channels),"block_start_cycle":cycle,"block_end_cycle":active.burst_end_cycle,"blocked_cycles":blocked,"blocking_request_state":"active_speculative"})
        return queued

    def _kick_memory(self,s:_State,cycle:int)->None:
        before = len(s.memory.live_descriptors)
        promotion_before = len(s.memory.tokens['promoted'])
        for request in s.memory.start_ready(cycle):
            for key in request.object_keys:s.ocm.set_running(key,True)
            s.queue.push(request.burst_end_cycle or cycle,EventType.DMA_DONE,{"request":request})
            if len(request.burst_history) == 1:
                self._append_trace(s,cycle,"dma_start",request.layer_id,request.expert_id,request.tile_ids[0],request.matrix_type,"MEMORY",request.request_type,getattr(request,"source_layer_id",None),request.layer_id,end_cycle=request.end_cycle)
            if s.config.execution_rules.burst_arbitration:
                self._append_trace(s,cycle,'dma_burst_start',request.layer_id,request.expert_id,request.tile_ids[0],request.matrix_type,
                    'MEMORY',request.request_type,end_cycle=request.burst_end_cycle)
                if s.trace_enabled:
                    s.trace[-1]['burst_bytes'] = request.burst_bytes
        if len(s.memory.live_descriptors) < before:
            for current in s.layers.values():
                if current.resolved and not current.done:
                    self._refill_demands(s, current, cycle)
            self._retry_prefetch_pairs(s, cycle)
        if len(s.memory.tokens['promoted']) < promotion_before:
            for pending in s.layers.values():
                if pending.route_pending:
                    self._router_resolve(s, Event(cycle, 4, 0, EventType.ROUTER_DONE, {'layer_id': pending.event.layer_id}))
        deadlines = set(s.memory.retirement_deadlines.values()) | {
            r.enqueue_ready_cycle for queues in s.memory.tokens.values() for r in queues}
        deadlines.update(end for entries in s.memory.fifo_dequeue_pending.values() for _, end in entries)
        for deadline in deadlines:
            if deadline > cycle and deadline not in s.credit_wakeup_cycles:
                s.credit_wakeup_cycles.add(deadline)
                s.queue.push(deadline, EventType.CONTROL_CREDIT_READY)

    def _dma_done(self,s:_State,e:Event)->None:
        request:MemoryRequest=e.payload["request"]
        complete = s.memory.complete(request,e.cycle)
        if not complete:
            self._kick_memory(s, e.cycle)
            return
        edge=getattr(request,"inter_layer_edge",None)
        if edge is not None:
            s.inter_background_done.add(edge);self._append_trace(s,e.cycle,"background_dma_done",request.layer_id,resource="MEMORY",request_state="background",source_layer=edge[0],target_layer=edge[1]);self._kick_memory(s,e.cycle);self._try_start_inter_target(s,edge,e.cycle);return
        for key in request.object_keys:
            s.ocm.set_running(key,False)
            if not s.config.execution_rules.ocm_payload_visibility_cycles:
                if key in s.ocm.objects:s.ocm.mark_resident(key,e.cycle)
                if request.useful is False and s.layers[request.layer_id].resolved:s.ocm.release(key,e.cycle)
        if s.config.execution_rules.ocm_payload_visibility_cycles:
            s.queue.push(e.cycle + s.config.execution_rules.ocm_payload_visibility_cycles,
                         EventType.OCM_PAYLOAD_READY, {'request': request})
        self._append_trace(s,e.cycle,"dma_done",request.layer_id,request.expert_id,request.tile_ids[0],request.matrix_type,"MEMORY",request.request_type,getattr(request,"source_layer_id",None),request.layer_id)
        s.memory.retire_descriptors()
        for pending in s.layers.values():
            if pending.route_pending:
                self._router_resolve(s, Event(e.cycle, 4, 0, EventType.ROUTER_DONE, {'layer_id': pending.event.layer_id}))
        self._release_deferred_predictions(s,e.cycle)
        if s.config.execution_rules.atomic_tile_pair:
            for current in s.layers.values():
                if current.resolved and not current.done:
                    self._refill_demands(s, current, e.cycle)
        self._retry_prefetch_pairs(s, e.cycle)
        self._kick_memory(s,e.cycle)
        layer=s.layers[request.layer_id]
        if layer.resolved:
            s.queue.push(e.cycle,EventType.SCHEDULER_INVOKE,{"layer_id":layer.event.layer_id})
            for expert in layer.experts.values():
                if expert.launched and not expert.done:self._progress(s,layer,expert,e.cycle)

    def _schedule(self,s:_State,cycle:int,layer_id:int|None)->None:
        if layer_id is None:return
        layer=s.layers[layer_id]
        if not layer.resolved or layer.done:return
        if cycle < layer.scheduler_ready_cycle:
            return
        if s.config.execution_rules.compute_policy == 'tile_ready_first':
            if layer.tile_owner is not None:
                return
            for expert in layer.experts.values():
                if expert.done or expert.compute_busy or expert.stage != 'waiting_gu':
                    continue
                key = self._key(layer_id, expert.expert_id, 'gate_up', expert.tile_id)
                down = self._key(layer_id, expert.expert_id, 'down', expert.tile_id)
                if s.config.execution_rules.wait_full_expert and not expert.launched and not all(
                        s.ocm.is_resident(k, cycle) for k, _ in self._all_keys(layer, expert)):
                    continue
                if s.ocm.is_resident(key, cycle) and down in s.ocm.objects:
                    layer.tile_owner = expert.expert_id
                    self._append_trace(s, cycle, 'tile_select', layer_id, expert.expert_id,
                                       expert.tile_id, 'gate_up', 'CONTROLLER')
                    expert.launched = True
                    layer.active.add(expert.expert_id)
                    if expert.wait_start is None:
                        expert.wait_start = layer.resolve_cycle if expert.start_cycle is None else cycle
                    self._progress(s, layer, expert, cycle)
                    break
            return
        slots=s.config.max_parallel_experts-len(layer.active)
        eligible=[expert for expert in layer.experts.values() if not expert.launched and self._startup_ready(s,layer,expert,cycle)]
        eligible.sort(key=lambda x:(-x.token_count,x.expert_id))
        for expert in eligible[:max(0,slots)]:
            expert.launched=True;layer.active.add(expert.expert_id);expert.wait_start=layer.resolve_cycle
            self._progress(s,layer,expert,cycle)

    def _progress(self,s:_State,layer:_Layer,expert:_Expert,cycle:int, *, dispatch_paid: bool=False)->None:
        if s.config.execution_rules.compute_policy == 'tile_ready_first' and layer.tile_owner != expert.expert_id:
            return
        if expert.done or expert.compute_busy or expert.stage not in {"waiting_gu","waiting_down"}:return
        group=expert.groups[expert.tile_id];matrix="gate_up" if expert.stage=="waiting_gu" else "down";key=self._key(layer.event.layer_id,expert.expert_id,matrix,group.tile_id)
        if not s.ocm.is_resident(key,cycle):
            if expert.wait_start is None:expert.wait_start=cycle
            return
        expert.starvation+=max(0,cycle-(expert.wait_start if expert.wait_start is not None else cycle));expert.wait_start=None
        ops=group.gate_up_ops if matrix=="gate_up" else group.down_ops;task=AIEPoolTask(matrix,ops,s.config.aie_units_per_expert,layer.event.layer_id,expert.expert_id,group.tile_id)
        dispatch = s.config.execution_rules.tile_dispatch_cycles if matrix == 'gate_up' and not dispatch_paid else 0
        committed = s.commit_ready_cycles.get(key, cycle)
        launch = s.config.execution_rules.committed_launch_cycles if key in s.commit_ready_cycles else 0
        if s.config.execution_rules.route_commit_visibility_cycles and not dispatch_paid:
            if expert.start_cycle is None and expert.memory_ready_cycle is None:
                # payload可用和GEMM发射分开；不能将commit/dispatch记作DDR等待。
                expert.memory_ready_cycle = cycle
                expert.aie_request_cycle = cycle
            expert.compute_busy = True
            expert.stage = matrix
            s.ocm.set_running(key, True)
            if dispatch:
                self._append_trace(s, cycle, 'tile_dispatch', layer.event.layer_id, expert.expert_id,
                                   group.tile_id, matrix, 'CONTROLLER', end_cycle=cycle + dispatch)
            s.queue.push(max(cycle + dispatch, committed + launch), EventType.AIE_OPERATION_START,
                         {'layer_id': layer.event.layer_id, 'expert_id': expert.expert_id, 'matrix': matrix})
            return
        start,end=s.aie.reserve(task,max(cycle + dispatch, committed + launch));expert.compute_busy=True;expert.stage=matrix;s.ocm.set_running(key,True)
        if dispatch:
            self._append_trace(s, cycle, 'tile_dispatch', layer.event.layer_id, expert.expert_id,
                               group.tile_id, matrix, 'CONTROLLER', end_cycle=cycle + dispatch)
        if expert.start_cycle is None:
            if expert.memory_ready_cycle is None:
                expert.memory_ready_cycle=cycle;expert.aie_request_cycle=cycle
            expert.aie_ready_cycle=start;expert.gu_start_cycle=start
            expert.start_cycle=start;self._milestone(s,layer,"source_expert_start",start)
        if matrix=="down" and expert.down_start_cycle is None:expert.down_start_cycle=start
        if matrix=="gate_up" and expert.tile_id+1<len(expert.groups) and not s.config.execution_rules.atomic_tile_pair and s.config.demand_loading == 'tile_streaming':
            nxt=expert.groups[expert.tile_id+1]
            self._issue(s,self._key(layer.event.layer_id,expert.expert_id,"gate_up",nxt.tile_id),nxt.gate_up_bytes,"demand",cycle)
            self._kick_memory(s,cycle)
        s.queue.push(end,EventType.AIE_OPERATION_DONE,{"kind":"expert","layer_id":layer.event.layer_id,"expert_id":expert.expert_id,"matrix":matrix})
        self._append_trace(s,start,"aie_start",layer.event.layer_id,expert.expert_id,group.tile_id,matrix,"AIE_POOL",end_cycle=end)

    def _nonlinear_done(self,s:_State,e:Event)->None:
        layer=s.layers[e.payload["layer_id"]];expert=layer.experts[e.payload["expert_id"]];expert.nonlinear_finish_cycle=e.cycle;expert.stage="waiting_down";expert.wait_start=e.cycle
        if s.config.demand_loading == 'tile_serial':
            self._refill_demands(s, layer, e.cycle)
        self._progress(s,layer,expert,e.cycle)

    def _maybe_layer_done(self,s:_State,layer:_Layer,cycle:int)->None:
        if not all(expert.done for expert in layer.experts.values()):
            s.queue.push(cycle,EventType.SCHEDULER_INVOKE,{"layer_id":layer.event.layer_id});return
        if layer.completion_pending or layer.done:
            return
        combine = s.config.execution_rules.layer_combine_cycles
        if combine:
            layer.completion_pending = True
            self._append_trace(s, cycle, 'layer_combine', layer.event.layer_id, resource='CONTROLLER', end_cycle=cycle + combine)
            s.queue.push(cycle + combine, EventType.LAYER_COMBINE_DONE, {'layer_id': layer.event.layer_id})
        else:
            self._finish_layer(s, layer, cycle)

    def _finish_layer(self, s: _State, layer: _Layer, cycle: int) -> None:
        """推理输出完成；层间 retirement 独立于最后一层数值输出。"""
        layer.done=True;layer.done_cycle=cycle;s.inference_done=max(s.inference_done,cycle)
        order=[event.layer_id for event in s.sequence.layers];index=order.index(layer.event.layer_id)
        if index+1<len(order):
            target=order[index+1]
            s.queue.push(cycle,EventType.INTER_LAYER_START,{"source_layer_id":layer.event.layer_id,"target_layer_id":target})

    def _inter_layer_start(self,s:_State,e:Event)->None:
        source,target=e.payload["source_layer_id"],e.payload["target_layer_id"];edge=(source,target)
        segment=s.sequence.inter_layer_config.segment(source,target)
        legacy_gap=s.sequence.inter_layer_cycles.get(edge,0)
        _,aie_end=s.aie.reserve_fixed(segment.aie_cycles,self.profile.total_aie_units,e.cycle,source)
        nonlinear_start=max(e.cycle,s.nonlinear.available_cycle)
        nonlinear_end=s.nonlinear.reserve_fixed(segment.nonlinear_cycles,nonlinear_start)
        rules = s.config.execution_rules
        retirement_floor = e.cycle if rules.layer_retirement_wait_for_ingress else e.cycle + rules.layer_retirement_cycles
        fixed=max(aie_end,nonlinear_end,e.cycle+segment.controller_cycles,e.cycle+legacy_gap,
                  retirement_floor)
        if rules.layer_retirement_wait_for_ingress:
            s.queue.push(e.cycle + rules.layer_retirement_probe_cycles,
                         EventType.INTER_LAYER_RETIRE_CHECK, dict(source_layer_id=source,target_layer_id=target))
        s.inter_fixed_done[edge]=fixed
        self._append_trace(s,e.cycle,"synthetic_inter_layer",source,resource="CONTROLLER",source_layer=source,target_layer=target,end_cycle=fixed)
        if segment.background_dma_bytes:
            request=s.memory.new_request(layer_id=target,expert_id=-1,matrix_type="background",tile_ids=(0,),object_keys=((source,target,"background",0),),transfer_bytes=segment.background_dma_bytes,request_type="background",issue_cycle=e.cycle)
            request.inter_layer_edge=edge;s.memory.enqueue(request);s.requests.append(request);self._kick_memory(s,e.cycle)
        else:s.inter_background_done.add(edge)
        s.queue.push(fixed,EventType.INTER_LAYER_FIXED_DONE,{"source_layer_id":source,"target_layer_id":target})

    def _retirement_check(self, s: _State, e: Event) -> None:
        """等待实际ingress/local resolver退休，再执行剩余层切换控制链；不等未来DMA。"""
        source, target = e.payload['source_layer_id'], e.payload['target_layer_id']
        edge = (source, target)
        if edge in s.inter_retirement_done:
            return
        ingress = s.memory.ingress_available_cycle
        resolver_pending = any(request.layer_id == source for request in s.resolver_entries.values())
        source_control = s.layers[source].route_pending or not s.layers[source].demands_released
        if ingress > e.cycle or resolver_pending or source_control:
            wake = max(e.cycle + 1, ingress)
            self._append_trace(s, e.cycle, 'retirement_resource_wait', source,
                               resource='CONTROLLER', source_layer=source, target_layer=target,
                               end_cycle=wake)
            if s.trace_enabled:
                s.trace[-1].update(ingress_pending=ingress > e.cycle,
                                  resolver_pending=resolver_pending, source_control_pending=source_control)
            s.queue.push(wake, EventType.INTER_LAYER_RETIRE_CHECK, e.payload)
            return
        rules = s.config.execution_rules
        end = e.cycle + rules.layer_retirement_cycles - rules.layer_retirement_probe_cycles
        s.inter_retirement_done[edge] = end
        self._append_trace(s, e.cycle, 'retirement_resources_ready', source, resource='CONTROLLER',
                           source_layer=source, target_layer=target, end_cycle=end)
        s.queue.push(max(end, s.inter_fixed_done[edge]), EventType.INTER_LAYER_FIXED_DONE, e.payload)

    def _try_start_inter_target(self,s:_State,edge:tuple[int,int],cycle:int)->None:
        if edge in s.inter_started or edge not in s.inter_background_done or cycle<s.inter_fixed_done.get(edge,cycle+1):return
        if (s.config.execution_rules.layer_retirement_wait_for_ingress
                and cycle < s.inter_retirement_done.get(edge, cycle + 1)):
            return
        s.inter_started.add(edge);s.queue.push(cycle,EventType.LAYER_START,{"layer_id":edge[1]})

    def _milestone(self,s:_State,layer:_Layer,name:str,cycle:int)->None:
        for spec in s.predictions:
            key=(spec.source_layer_id,spec.target_layer_id)
            if spec.source_layer_id!=layer.event.layer_id or spec.source_milestone!=name or key in s.triggered:continue
            self._trigger_prediction(s,spec,cycle)

    def _trigger_prediction(self,s:_State,spec:UnifiedPredictionSpec,feature_cycle:int)->None:
        key=(spec.source_layer_id,spec.target_layer_id)
        if key in s.triggered:return
        s.triggered.add(key);ready=spec.synthetic_prediction_ready_override if spec.synthetic_prediction_ready_override is not None else feature_cycle+spec.predictor_latency_cycles
        output=s.predictors[key].predict(s.layers[spec.target_layer_id].event);trace_id=getattr(output,"trace_id",f"prediction:{key}")
        if output is None:raise ValidationError(f"predictor returned no output for {key}")
        s.prediction_outputs[key]=output
        candidates=output.aggregate_candidates(spec.candidate_count) if output is not None else ()
        s.prediction_records[key]=UnifiedPredictionRecord(spec.source_layer_id,spec.target_layer_id,feature_cycle,ready,predictor_trace_id=trace_id,synthetic_lead_time=spec.synthetic_prediction_ready_override is not None,candidate_ids=tuple(x.expert_id for x in candidates),candidate_scores=tuple(x.score for x in candidates),prefetch_policy=spec.prefetch_policy,source_milestone=spec.source_milestone,predictor_latency_cycles=spec.predictor_latency_cycles)
        s.queue.push(ready,EventType.PREDICTOR_READY,{"spec":spec})
        self._append_trace(s,feature_cycle,"prediction_trigger",spec.source_layer_id,source_layer=spec.source_layer_id,target_layer=spec.target_layer_id,end_cycle=ready)

    def _layer_result(self,s:_State,layer:_Layer)->UnifiedLayerResult:
        starts=[expert.start_cycle or layer.done_cycle for expert in layer.experts.values()];counts=layer.event.expert_token_counts();total=sum(counts.values());startable=layer.startable_at_resolve
        related=[r for r in s.requests if r.layer_id==layer.event.layer_id]
        timing_rows=[]
        for x in sorted(layer.experts.values(),key=lambda item:item.expert_id):
            required=sum(size for _,size in self._startup_keys(s,layer,x));resident=int(layer.startup_readiness.get(x.expert_id,0.0)*required);inflight=layer.startup_inflight_bytes.get(x.expert_id,0)
            timing_rows.append({"expert_id":x.expert_id,"token_count":x.token_count,"start_cycle":x.start_cycle,"finish_cycle":x.finish_cycle,"router_to_start_cycles":(x.start_cycle-layer.resolve_cycle) if x.start_cycle is not None else None,"memory_wait_cycles":x.starvation,"earliest_memory_ready_cycle":x.memory_ready_cycle,"earliest_nonmemory_dependency_ready_cycle":layer.resolve_cycle,"aie_request_cycle":x.aie_request_cycle,"aie_available_cycle":x.aie_ready_cycle,"aie_wait_cycles":(x.aie_ready_cycle-x.aie_request_cycle) if x.aie_ready_cycle is not None and x.aie_request_cycle is not None else 0,"allocated_aie_groups":s.config.aie_units_per_expert,"gu_start_cycle":x.gu_start_cycle,"gu_finish_cycle":x.gu_finish_cycle,"nonlinear_finish_cycle":x.nonlinear_finish_cycle,"down_start_cycle":x.down_start_cycle,"startup_required_bytes":required,"startup_readiness":layer.startup_readiness.get(x.expert_id,0.0),"startup_inflight_bytes_at_router":inflight,"startup_transfer_progress_at_router":layer.startup_progress.get(x.expert_id,0.0),"startup_gap_bytes":max(0,required-resident-inflight),"startup_gap_cycles":(x.start_cycle-layer.resolve_cycle) if x.start_cycle is not None else None,"startable_at_router":x.expert_id in startable,"critical_path":x.finish_cycle==self._critical_finish(layer)})
        timings=tuple(timing_rows)
        critical_ids=tuple(sorted(x.expert_id for x in layer.experts.values() if x.finish_cycle==self._critical_finish(layer)))
        return UnifiedLayerResult(layer.event.layer_id,layer.start_cycle,layer.resolve_cycle,layer.done_cycle,tuple(sorted(layer.experts)),counts,min(starts),max(starts),len(startable)/len(layer.experts),sum(counts[x] for x in startable)/total,sum(layer.startup_readiness[x]*counts[x] for x in counts)/total,sum(r.transfer_bytes for r in related if r.request_type=="demand" and r.started),sum(r.transfer_bytes for r in related if r.request_type=="speculative" and r.started),sum(r.transfer_bytes for r in related if r.request_type=="speculative" and r.started and r.useful is True),sum(r.transfer_bytes for r in related if r.request_type=="speculative" and r.started and r.useful is False),sum(x.starvation for x in layer.experts.values()),timings,critical_ids)

    @staticmethod
    def _critical_finish(layer: _Layer) -> int:
        """关键 Expert 以最后 DOWN 完成为准，不把 combine 尾部误归给 Expert。"""
        return max((expert.finish_cycle or 0 for expert in layer.experts.values()), default=0)

    def _all_keys(self,layer:_Layer,expert:_Expert)->list[tuple[Key,int]]:
        return [(self._key(layer.event.layer_id,expert.expert_id,matrix,g.tile_id),g.gate_up_bytes if matrix=="gate_up" else g.down_bytes) for g in expert.groups for matrix in ("gate_up","down")]
    def _startup_keys(self,s:_State,layer:_Layer,expert:_Expert)->list[tuple[Key,int]]:
        if s.config.demand_loading == 'full_expert_load' or s.config.execution_rules.wait_full_expert:
            return self._all_keys(layer, expert)
        return [(self._key(layer.event.layer_id,expert.expert_id,"gate_up",i),expert.groups[i].gate_up_bytes) for i in range(min(s.config.startup_gu_tiles, len(expert.groups)))]
    def _startup_ready(self,s:_State,layer:_Layer,expert:_Expert,cycle:int)->bool:return all(s.ocm.is_resident(key,cycle) for key,_ in self._startup_keys(s,layer,expert))
    def _transferred_bytes_at(self,request:MemoryRequest,cycle:int)->int:
        """Return physical payload bytes transferred by a cycle, excluding setup."""
        rate = self.profile.effective_bytes_per_cycle_per_channel * len(request.channel_ids) / request.payload_cycle_multiplier
        return min(request.transfer_bytes, sum(min(size, int(max(0, min(cycle, end) - payload_start) * rate))
            for _, end, size, payload_start in request.burst_history))
    @staticmethod
    def _key(layer:int,expert:int,matrix:str,tile:int)->Key:return(layer,expert,matrix,tile)
    def _trace(self,s:_State,e:Event)->None:
        if s.trace_enabled:
            payload={key:value for key,value in e.payload.items() if isinstance(value,(str,int,float,bool,type(None)))}
            s.trace.append({"cycle":e.cycle,"event_type":e.event_type.name,**payload})
    def _append_trace(self,s:_State,cycle:int,event_type:str,layer_id:int,expert_id:int|None=None,tile_id:int|None=None,matrix_type:str|None=None,resource:str|None=None,request_state:str|None=None,source_layer:int|None=None,target_layer:int|None=None,end_cycle:int|None=None)->None:
        if s.trace_enabled:s.trace.append({"cycle":cycle,"end_cycle":end_cycle,"event_type":event_type,"layer_id":layer_id,"source_layer_id":source_layer,"target_layer_id":target_layer,"expert_id":expert_id,"tile_id":tile_id,"matrix_type":matrix_type,"resource":resource,"request_state":request_state})
