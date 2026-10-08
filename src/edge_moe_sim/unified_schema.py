"""Schema for the shared-resource, cross-layer advanced simulator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Mapping
from .advanced_schema import AdvancedExecutionConfig
from .batch_routing import BatchRoutingEvent
from .schema import InterLayerConfig, ValidationError

SourceMilestone = Literal["source_layer_start", "source_router_done", "source_expert_start", "source_gate_up_done", "source_expert_done", "custom_offset"]

@dataclass(frozen=True)
class UnifiedLayerSequence:
    """Ordered batch-routed MoE layers sharing one physical resource timeline."""
    layers: tuple[BatchRoutingEvent, ...]
    inter_layer_cycles: Mapping[tuple[int, int], int] = field(default_factory=dict)
    inter_layer_config: InterLayerConfig = field(default_factory=InterLayerConfig)
    def __post_init__(self) -> None:
        if not self.layers or any(right.layer_id <= left.layer_id for left,right in zip(self.layers,self.layers[1:])):
            raise ValidationError("unified layers must be non-empty and strictly layer ordered")
        if len({event.batch_size for event in self.layers}) != 1:
            raise ValidationError("unified sequence currently requires a common batch size")
        if any(value < 0 for value in self.inter_layer_cycles.values()):
            raise ValidationError("inter-layer cycles must be non-negative")
        adjacent = {(left.layer_id, right.layer_id) for left, right in zip(self.layers, self.layers[1:])}
        if any(key not in adjacent for key in self.inter_layer_cycles):
            raise ValidationError("inter-layer cycle keys must identify adjacent sequence layers")

@dataclass(frozen=True)
class UnifiedPredictionSpec:
    """Source-milestone prediction with resource-aware release by default.

    ``wait_for_previous_layer_weights=True`` is an optional conservative policy,
    not a hardware dependency. With False, ready predictions immediately attempt
    admission; DMA priority/non-preemption and finite OCM still apply.
    """
    source_layer_id: int
    target_layer_id: int
    source_milestone: SourceMilestone = "source_router_done"
    predictor_latency_cycles: int = 0
    candidate_count: int = 1
    prefetch_policy: Literal["full_expert_prefetch", "uniform_prefix", "ranked_prefix"] = "uniform_prefix"
    prefix_depth: int = 1
    candidate_depths: tuple[int, ...] = ()
    custom_offset_cycles: int = 0
    synthetic_prediction_ready_override: int | None = None
    wait_for_previous_layer_weights: bool = False
    def __post_init__(self) -> None:
        if self.source_layer_id >= self.target_layer_id or self.predictor_latency_cycles < 0 or self.candidate_count <= 0 or self.prefix_depth < 0 or self.custom_offset_cycles < 0 or any(depth < 0 for depth in self.candidate_depths):
            raise ValidationError("invalid unified prediction specification")
        if self.candidate_depths and len(self.candidate_depths) != self.candidate_count:
            raise ValidationError("candidate_depths must match candidate_count")
        if self.source_milestone not in {"source_layer_start", "source_router_done", "source_expert_start", "source_gate_up_done", "source_expert_done", "custom_offset"}:
            raise ValidationError("unknown unified source milestone")
        if self.prefetch_policy not in {"full_expert_prefetch", "uniform_prefix", "ranked_prefix"}:
            raise ValidationError("unknown unified prefetch policy")
        if self.synthetic_prediction_ready_override is not None and self.synthetic_prediction_ready_override < 0:
            raise ValidationError("synthetic prediction ready override must be non-negative")

@dataclass
class UnifiedPredictionRecord:
    """Observed source-to-target prediction timing and realized lead time."""
    source_layer_id: int; target_layer_id: int; source_milestone_cycle: int
    predictor_ready_cycle: int; target_router_resolve_cycle: int = 0
    predictor_trace_id: str = ""; synthetic_lead_time: bool = False
    candidate_ids: tuple[int, ...] = ()
    candidate_scores: tuple[float, ...] = ()
    prefetch_policy: str = ""
    source_milestone: str = ""
    predictor_latency_cycles: int = 0
    target_layer_start_cycle: int = 0
    speculative_bytes: int = 0
    useful_bytes: int = 0
    wasted_bytes: int = 0
    canceled_bytes: int = 0
    actual_speculative_bytes_before_router: int = 0
    useful_true_bytes_before_router: int = 0
    theoretical_bytes_in_lead_window: float = 0.0
    prefetch_opportunity_expert_equivalents: float = 0.0
    initial_allocated_speculative_bytes: int = 0
    planned_speculative_bytes: int = 0
    canceled_unadmitted_bytes: int = 0
    cancelled_before_issue_speculative_bytes: int = 0
    pending_unissued_speculative_bytes: int = 0
    prefetch_release_cycle: int | None = None
    issued_speculative_bytes: int = 0
    completed_speculative_bytes: int = 0
    completed_speculative_bytes_before_router: int = 0
    partial_inflight_speculative_bytes_at_router: int = 0
    physical_speculative_bytes_transferred_by_router: int = 0
    useful_completed_speculative_bytes_before_router: int = 0
    useful_physical_speculative_bytes_by_router: int = 0
    wasted_physical_speculative_bytes: int = 0
    allocation_policy: str = ""
    dma_order: str = "legacy"
    allocation_budget_bytes: int = 0
    planned_candidate_depths: tuple[int, ...] = ()
    @property
    def realized_lead_cycles(self) -> int: return self.target_router_resolve_cycle-self.predictor_ready_cycle
    @property
    def realized_lead_before_target_start(self) -> int:
        return max(0,self.target_layer_start_cycle-self.predictor_ready_cycle)
    @property
    def opportunity_utilization(self) -> float:
        return self.actual_speculative_bytes_before_router/self.theoretical_bytes_in_lead_window if self.theoretical_bytes_in_lead_window else 0.0
    @property
    def useful_opportunity_utilization(self) -> float:
        return self.useful_true_bytes_before_router/self.theoretical_bytes_in_lead_window if self.theoretical_bytes_in_lead_window else 0.0
    @property
    def physical_opportunity_utilization(self) -> float:
        """Return physical Router-time speculative traffic over theoretical opportunity."""
        return self.physical_speculative_bytes_transferred_by_router/self.theoretical_bytes_in_lead_window if self.theoretical_bytes_in_lead_window else 0.0
    @property
    def useful_physical_opportunity_utilization(self) -> float:
        """Return useful physical Router-time traffic over theoretical opportunity."""
        return self.useful_physical_speculative_bytes_by_router/self.theoretical_bytes_in_lead_window if self.theoretical_bytes_in_lead_window else 0.0

@dataclass
class UnifiedLayerResult:
    """Observed timing, workload, and readiness for one unified layer."""
    layer_id: int
    layer_start_cycle: int
    router_resolve_cycle: int
    layer_done_cycle: int
    true_expert_ids: tuple[int, ...]
    per_expert_token_counts: Mapping[int, int]
    first_true_expert_start: int
    last_true_expert_start: int
    startable_recall: float
    weighted_startable_recall: float
    weighted_startup_readiness: float
    demand_bytes: int
    speculative_bytes: int
    useful_speculative_bytes: int
    wasted_speculative_bytes: int
    aie_starvation_cycles: int
    expert_timings: tuple[Mapping[str, int | float | bool | None], ...] = ()
    critical_completion_expert_ids: tuple[int, ...] = ()

@dataclass
class UnifiedSequenceResult:
    """Sequence-level outcome for shared M6 resources and M5 causal semantics."""
    makespan_cycles: int
    layer_results: Mapping[int, object]
    predictions: list[UnifiedPredictionRecord]
    demand_blocked_by_future_speculation_cycles: int
    peak_ocm_bytes: int
    rejected_future_prefetch_bytes: int = 0
    evicted_future_speculative_bytes: int = 0
    final_resource_idle_cycle: int = 0
    memory_metrics: Mapping[str, object] = field(default_factory=dict)
    aie_metrics: Mapping[str, object] = field(default_factory=dict)
    trace: list[dict[str, object]] = field(default_factory=list)
    blocking_records: tuple[Mapping[str, object], ...] = ()
    critical_path_future_speculation_blocking_cycles: int = 0
    speculative_request_records: tuple[Mapping[str, object], ...] = ()
    @property
    def inference_makespan_cycles(self) -> int:
        """Return required-computation completion, excluding irrelevant drain."""
        return self.makespan_cycles
