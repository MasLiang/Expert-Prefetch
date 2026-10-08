"""Typed input and output schemas used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


class ValidationError(ValueError):
    """Raised when a simulator configuration or trace record is invalid."""


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value!r}")


def _nonnegative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True)
class ModelConfig:
    """Static MoE dimensions and numeric formats."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    weight_bits: int
    activation_bits: int
    fuse_gate_up: bool = True
    router_weights_resident: bool = True
    moe_layer_indices: tuple[int, ...] = ()
    expert_architecture: Literal["swiglu", "relu_ffn"] = "swiglu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "moe_layer_indices", tuple(self.moe_layer_indices))
        for name in (
            "hidden_size", "intermediate_size", "num_experts", "top_k",
            "weight_bits", "activation_bits",
        ):
            _positive(name, getattr(self, name))
        if not self.name:
            raise ValidationError("model name must not be empty")
        if self.top_k > self.num_experts:
            raise ValidationError("top_k cannot exceed num_experts")
        if self.expert_architecture not in ("swiglu", "relu_ffn"):
            raise ValidationError("expert_architecture must be swiglu or relu_ffn")
        if self.expert_architecture == "swiglu" and not self.fuse_gate_up:
            raise ValidationError("Milestone 1/2 supports only fused Gate-Up")
        if not self.router_weights_resident:
            raise ValidationError("Milestone 1/2 requires resident router weights")
        if any(not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0
               for layer_id in self.moe_layer_indices):
            raise ValidationError("moe_layer_indices must contain non-negative integers")
        if tuple(sorted(set(self.moe_layer_indices))) != self.moe_layer_indices:
            raise ValidationError("moe_layer_indices must be strictly increasing")

    @property
    def input_projection_count(self) -> int:
        """Number of H-by-F input weight matrices: SwiGLU=2, ReLU FFN=1."""
        return 2 if self.expert_architecture == "swiglu" else 1

    @property
    def nonlinear_operation(self) -> str:
        """Nonlinear workload identity; effective throughput remains configurable."""
        return "silu_mul" if self.expert_architecture == "swiglu" else "relu"

    def next_moe_layer(self, source_layer_id: int) -> int | None:
        """Return the configured MoE layer following ``source_layer_id``."""
        try:
            index = self.moe_layer_indices.index(source_layer_id)
        except ValueError as exc:
            raise ValidationError(
                f"source layer {source_layer_id} is not in model.moe_layer_indices"
            ) from exc
        return (
            self.moe_layer_indices[index + 1]
            if index + 1 < len(self.moe_layer_indices)
            else None
        )


@dataclass(frozen=True)
class AIEConfig:
    """Effective AIE rates and per-operation overheads, all in cycles."""

    router_ops_per_cycle: int
    gate_up_ops_per_cycle: int
    down_ops_per_cycle: int
    startup_cycles: int
    operation_switch_cycles: int
    first_operation_pays_switch_cycles: bool = False
    matrix_startup_cycles: int | None = None
    tile_pipeline_cycles: int = 0

    def __post_init__(self) -> None:
        for name in ("router_ops_per_cycle", "gate_up_ops_per_cycle", "down_ops_per_cycle"):
            _positive(name, getattr(self, name))
        _nonnegative("aie.startup_cycles", self.startup_cycles)
        _nonnegative("aie.operation_switch_cycles", self.operation_switch_cycles)
        if self.matrix_startup_cycles is not None:
            _nonnegative("aie.matrix_startup_cycles", self.matrix_startup_cycles)
        _nonnegative("aie.tile_pipeline_cycles", self.tile_pipeline_cycles)

    @property
    def effective_matrix_startup_cycles(self) -> int:
        """Return the streaming startup, falling back to legacy startup."""
        return self.startup_cycles if self.matrix_startup_cycles is None else self.matrix_startup_cycles


@dataclass(frozen=True)
class DMAConfig:
    """Single-engine DMA bandwidth and startup configuration."""

    bytes_per_cycle: int
    startup_cycles: int
    max_inflight: int = 1
    preemptive: bool = False

    def __post_init__(self) -> None:
        _positive("dma.bytes_per_cycle", self.bytes_per_cycle)
        _nonnegative("dma.startup_cycles", self.startup_cycles)
        if self.max_inflight != 1:
            raise ValidationError("Milestone 1/2 supports dma.max_inflight=1 only")
        if self.preemptive:
            raise ValidationError("Milestone 1/2 supports non-preemptive DMA only")


@dataclass(frozen=True)
class NonlinearConfig:
    """Independent nonlinear-engine rate and startup cost."""

    elements_per_cycle: int
    startup_cycles: int

    def __post_init__(self) -> None:
        _positive("nonlinear.elements_per_cycle", self.elements_per_cycle)
        _nonnegative("nonlinear.startup_cycles", self.startup_cycles)


@dataclass(frozen=True)
class PLControlConfig:
    """PL controller latencies in integer cycles."""

    topk_cycles: int
    dispatch_cycles: int
    commit_cycles: int
    cancel_cycles: int
    combine_cycles: int

    def __post_init__(self) -> None:
        for name in ("topk_cycles", "dispatch_cycles", "commit_cycles", "cancel_cycles", "combine_cycles"):
            _nonnegative(f"pl_control.{name}", getattr(self, name))


@dataclass(frozen=True)
class HardwareConfig:
    """Complete abstract hardware configuration."""

    clock_hz: int
    aie: AIEConfig
    dma: DMAConfig
    nonlinear: NonlinearConfig
    pl_control: PLControlConfig

    def __post_init__(self) -> None:
        _positive("hardware.clock_hz", self.clock_hz)


@dataclass(frozen=True)
class PrefetchConfig:
    """Stage-level speculative-prefetch policy controls."""

    scope: Literal["gate_up_only", "full_expert"] = "gate_up_only"
    max_candidates: int = 4
    policy: Literal[
        "fixed_depth", "fixed_byte_budget", "progressive_rank", "predictor_specified"
    ] = "fixed_depth"
    gate_up_tiles_per_candidate: int = 2
    down_tiles_per_candidate: int = 0
    byte_budget: int | None = None
    include_down: bool = False
    gate_up_depths: tuple[int, ...] = (4, 3, 2, 1)
    down_depths: tuple[int, ...] = (0, 0, 0, 0)
    predictor_default_gate_up_tiles: int | None = None
    predictor_default_down_tiles: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"gate_up_only", "full_expert"}:
            raise ValidationError("prefetch.scope must be 'gate_up_only' or 'full_expert'")
        _positive("prefetch.max_candidates", self.max_candidates)
        if self.policy not in {
            "fixed_depth", "fixed_byte_budget", "progressive_rank", "predictor_specified"
        }:
            raise ValidationError("unsupported prefetch.policy")
        for name in ("gate_up_tiles_per_candidate", "down_tiles_per_candidate"):
            _nonnegative(f"prefetch.{name}", getattr(self, name))
        if self.byte_budget is not None:
            _positive("prefetch.byte_budget", self.byte_budget)
        for name in ("gate_up_depths", "down_depths"):
            if any(depth < 0 for depth in getattr(self, name)):
                raise ValidationError(f"prefetch.{name} entries must be non-negative")


@dataclass(frozen=True)
class SimulationConfig:
    """Select stage-level or detailed tile-level execution."""

    granularity: Literal["stage", "tile"] = "stage"

    def __post_init__(self) -> None:
        if self.granularity not in {"stage", "tile"}:
            raise ValidationError("simulation.granularity must be 'stage' or 'tile'")


@dataclass(frozen=True)
class TileConfig:
    """Fixed-byte tiling parameters."""

    tile_size_bytes: int
    min_tiles_per_matrix: int = 1

    def __post_init__(self) -> None:
        _positive("tile.tile_size_bytes", self.tile_size_bytes)
        _positive("tile.min_tiles_per_matrix", self.min_tiles_per_matrix)


@dataclass(frozen=True)
class WeightTile:
    """One indivisible weight-transfer and streaming-compute unit."""

    layer_id: int
    expert_id: int
    matrix_type: Literal["gate_up", "down"]
    tile_id: int
    num_tiles: int
    size_bytes: int
    compute_ops: int

    def __post_init__(self) -> None:
        _positive("weight_tile.num_tiles", self.num_tiles)
        if not 0 <= self.tile_id < self.num_tiles:
            raise ValidationError("weight_tile.tile_id must be in [0, num_tiles)")
        if self.matrix_type not in {"gate_up", "down"}:
            raise ValidationError("weight_tile.matrix_type must be gate_up or down")
        _positive("weight_tile.size_bytes", self.size_bytes)
        _nonnegative("weight_tile.compute_ops", self.compute_ops)


TileKey = tuple[int, int, str, int]


@dataclass
class TileExecutionRecord:
    """Readiness, execution, and starvation timestamps for one tile."""

    matrix_type: str
    tile_id: int
    weight_ready_cycle: int
    compute_eligible_cycle: int
    compute_start_cycle: int
    compute_done_cycle: int
    starvation_cycles: int
    dma_request_type: str | None
    speculative_useful: bool | None


@dataclass(frozen=True)
class RoutingEvent:
    """One decode-layer routing outcome from an input trace."""

    request_id: int
    token_position: int
    layer_id: int
    phase: Literal["decode"]
    true_expert_ids: tuple[int, ...]
    true_gating_weights: tuple[float, ...]
    expert_token_counts: dict[int, int]

    def validate(self, model: ModelConfig) -> None:
        """Validate this record against a model configuration."""
        if self.phase != "decode":
            raise ValidationError("Milestone 1/2 supports phase='decode' only")
        if len(self.true_expert_ids) != model.top_k:
            raise ValidationError("true_expert_ids length must equal model.top_k")
        if len(set(self.true_expert_ids)) != len(self.true_expert_ids):
            raise ValidationError("true_expert_ids must not contain duplicates")
        if len(self.true_gating_weights) != len(self.true_expert_ids):
            raise ValidationError("true_gating_weights length must match true_expert_ids")
        for expert_id in self.true_expert_ids:
            if not 0 <= expert_id < model.num_experts:
                raise ValidationError(f"expert ID {expert_id} is outside [0, {model.num_experts})")
            if expert_id not in self.expert_token_counts:
                raise ValidationError(f"missing token count for true expert {expert_id}")
            _positive(f"expert_token_counts[{expert_id}]", self.expert_token_counts[expert_id])


@dataclass(frozen=True)
class TargetKey:
    """Identity of one future routing target."""

    request_id: int
    token_position: int
    layer_id: int


@dataclass(frozen=True)
class LayerSequence:
    """Strictly ordered same-token MoE routing events."""

    request_id: int
    token_position: int
    routing_events: tuple[RoutingEvent, ...]

    def validate(self, model: ModelConfig) -> None:
        """Validate identity, ordering, model membership, and routing records."""
        if not self.routing_events:
            raise ValidationError("layer sequence must contain at least one routing event")
        layer_ids = tuple(event.layer_id for event in self.routing_events)
        if tuple(sorted(set(layer_ids))) != layer_ids:
            raise ValidationError("layer sequence layer_id values must be strictly increasing")
        configured = set(model.moe_layer_indices)
        if not configured:
            raise ValidationError("model.moe_layer_indices is required for layer sequences")
        for event in self.routing_events:
            if event.request_id != self.request_id:
                raise ValidationError("all sequence events must share request_id")
            if event.token_position != self.token_position:
                raise ValidationError("all sequence events must share token_position")
            if event.layer_id not in configured:
                raise ValidationError(
                    f"routing layer {event.layer_id} is not in model.moe_layer_indices"
                )
            event.validate(model)


class RoutingTraceIndex:
    """Deterministic lookup index for routing truth."""

    def __init__(self, sequences: tuple[LayerSequence, ...] | list[LayerSequence]) -> None:
        self._events: dict[tuple[int, int, int], RoutingEvent] = {}
        for sequence in sequences:
            for event in sequence.routing_events:
                key = (event.request_id, event.token_position, event.layer_id)
                if key in self._events:
                    raise ValidationError(f"duplicate routing trace key {key}")
                self._events[key] = event

    def get_event(self, request_id: int, token_position: int, layer_id: int) -> RoutingEvent:
        """Return one event or raise a clear validation error."""
        key = (request_id, token_position, layer_id)
        try:
            return self._events[key]
        except KeyError as exc:
            raise ValidationError(f"routing event {key} was not found") from exc


@dataclass(frozen=True)
class InterLayerSegment:
    """Parallel abstract work between two supplied MoE layers."""

    source_layer_id: int
    target_layer_id: int
    aie_cycles: int
    nonlinear_cycles: int = 0
    controller_cycles: int = 0
    background_dma_bytes: int = 0
    name: str = "inter_layer_compute"

    def __post_init__(self) -> None:
        if self.source_layer_id >= self.target_layer_id:
            raise ValidationError("inter-layer source_layer_id must be less than target_layer_id")
        for name in ("aie_cycles", "nonlinear_cycles", "controller_cycles", "background_dma_bytes"):
            _nonnegative(f"inter_layer.{name}", getattr(self, name))
        if not self.name:
            raise ValidationError("inter-layer name must not be empty")


@dataclass(frozen=True)
class InterLayerConfig:
    """Default and per-edge inter-layer resource segments."""

    default_aie_cycles: int = 0
    default_nonlinear_cycles: int = 0
    default_controller_cycles: int = 0
    default_background_dma_bytes: int = 0
    overrides: Mapping[str, InterLayerSegment] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "default_aie_cycles", "default_nonlinear_cycles",
            "default_controller_cycles", "default_background_dma_bytes",
        ):
            _nonnegative(f"inter_layer.{name}", getattr(self, name))

    def segment(self, source_layer_id: int, target_layer_id: int) -> InterLayerSegment:
        """Return the configured segment for one MoE-layer edge."""
        key = f"{source_layer_id}->{target_layer_id}"
        override = self.overrides.get(key)
        if override is not None:
            if (override.source_layer_id, override.target_layer_id) != (
                source_layer_id, target_layer_id
            ):
                raise ValidationError(f"inter-layer override {key} has mismatched endpoints")
            return override
        return InterLayerSegment(
            source_layer_id, target_layer_id, self.default_aie_cycles,
            self.default_nonlinear_cycles, self.default_controller_cycles,
            self.default_background_dma_bytes,
        )


@dataclass(frozen=True)
class CandidatePrefetch:
    """One ranked Expert candidate with optional matrix tile depths."""

    expert_id: int
    score: float
    gate_up_tiles: int | None = None
    down_tiles: int | None = None

    def __post_init__(self) -> None:
        if self.gate_up_tiles is not None:
            _nonnegative("candidate.gate_up_tiles", self.gate_up_tiles)
        if self.down_tiles is not None:
            _nonnegative("candidate.down_tiles", self.down_tiles)


@dataclass(frozen=True)
class PredictionEvent:
    """A source-triggered prediction for a future target layer."""

    request_id: int
    source_token_position: int
    source_layer_id: int
    target_token_position: int
    target_layer_id: int
    prediction_stage: str
    feature_type: str
    feature_ready_cycle: int
    predictor_latency_cycles: int
    prediction_ready_cycle: int
    candidates: tuple[CandidatePrefetch, ...]
    prediction_id: str = ""
    configured_candidate_count: int | None = None
    configured_generation_mode: str | None = None
    configured_target_recall: float | None = None
    sampled_hit_count: int | None = None

    def __post_init__(self) -> None:
        if self.source_token_position != self.target_token_position:
            raise ValidationError("Milestone 5 supports same-token cross-layer prediction only")
        if self.source_layer_id >= self.target_layer_id:
            raise ValidationError("prediction source layer must precede target layer")
        _nonnegative("prediction.feature_ready_cycle", self.feature_ready_cycle)
        _nonnegative("prediction.predictor_latency_cycles", self.predictor_latency_cycles)
        if self.prediction_ready_cycle != (
            self.feature_ready_cycle + self.predictor_latency_cycles
        ):
            raise ValidationError(
                "prediction_ready_cycle must equal feature_ready_cycle + predictor_latency_cycles"
            )
        ids = [candidate.expert_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValidationError("prediction candidates must not contain duplicate expert IDs")


@dataclass(frozen=True, init=False)
class PredictorOutput:
    """Progressive candidates with a backward-compatible legacy constructor."""

    request_id: int
    token_position: int
    layer_id: int
    candidates: tuple[CandidatePrefetch, ...]
    ready_offset_cycles: int

    def __init__(
        self,
        request_id: int,
        token_position: int,
        layer_id: int,
        candidate_ids: tuple[int, ...] | None = None,
        candidate_scores: tuple[float, ...] | None = None,
        ready_offset_cycles: int = 0,
        *,
        candidates: tuple[CandidatePrefetch, ...] | None = None,
    ) -> None:
        if candidates is not None and (candidate_ids is not None or candidate_scores is not None):
            raise ValidationError("provide candidates or legacy candidate_ids/scores, not both")
        if candidates is None:
            ids = candidate_ids or ()
            scores = candidate_scores or ()
            if len(ids) != len(scores):
                raise ValidationError("candidate_scores length must match candidate_ids")
            candidates = tuple(CandidatePrefetch(expert_id, score) for expert_id, score in zip(ids, scores))
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "token_position", token_position)
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "ready_offset_cycles", ready_offset_cycles)

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        """Return legacy candidate IDs derived from progressive candidates."""
        return tuple(candidate.expert_id for candidate in self.candidates)

    @property
    def candidate_scores(self) -> tuple[float, ...]:
        """Return legacy scores derived from progressive candidates."""
        return tuple(candidate.score for candidate in self.candidates)

    def validate(self, model: ModelConfig) -> None:
        """Validate candidate IDs, scores, and ready time."""
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValidationError("candidate_ids must not contain duplicates")
        if len(self.candidate_scores) != len(self.candidate_ids):
            raise ValidationError("candidate_scores length must match candidate_ids")
        _nonnegative("ready_offset_cycles", self.ready_offset_cycles)
        for expert_id in self.candidate_ids:
            if not 0 <= expert_id < model.num_experts:
                raise ValidationError(f"candidate expert ID {expert_id} is outside [0, {model.num_experts})")


@dataclass(frozen=True)
class ExpertTask:
    """One routed expert and its assigned token count."""

    layer_id: int
    expert_id: int
    token_count: int
    gating_weight: float


@dataclass
class ExpertExecutionRecord:
    """Cycle timestamps and accounting for one completed expert."""

    expert_id: int
    gate_up_weight_ready_cycle: int | None = None
    gate_up_start_cycle: int | None = None
    gate_up_done_cycle: int | None = None
    nonlinear_start_cycle: int | None = None
    nonlinear_done_cycle: int | None = None
    down_weight_ready_cycle: int | None = None
    down_start_cycle: int | None = None
    down_done_cycle: int | None = None
    memory_stall_cycles: int = 0
    compute_cycles: int = 0
    token_count: int = 0
    gate_up_tile_records: list[TileExecutionRecord] = field(default_factory=list)
    down_tile_records: list[TileExecutionRecord] = field(default_factory=list)
    first_tile_stall_cycles: int = 0
    inter_tile_stall_cycles: int = 0


@dataclass
class SimulationResult:
    """Summary metrics for one routed layer execution."""

    scenario: str
    request_id: int
    token_position: int
    layer_id: int
    layer_start_cycle: int
    router_start_cycle: int
    router_done_cycle: int
    topk_done_cycle: int
    dispatch_done_cycle: int
    resolve_cycle: int
    layer_done_cycle: int
    total_cycles: int
    aie_router_busy_cycles: int
    aie_expert_busy_cycles: int
    aie_memory_stall_cycles: int
    aie_idle_cycles: int
    nonlinear_busy_cycles: int
    dma_busy_cycles: int
    demand_bytes: int
    speculative_bytes: int
    useful_speculative_bytes: int
    wasted_speculative_bytes: int
    canceled_speculative_bytes: int
    static_recall: float | None
    any_hit: bool | None
    full_hit: bool | None
    ready_recall_at_router_done: float | None
    ready_recall_at_resolve: float | None
    expert_records: list[ExpertExecutionRecord] = field(default_factory=list)
    aie_control_transition_cycles: int = 0
    aie_dependency_wait_cycles: int = 0
    aie_control_gap_cycles: int = 0
    aie_unutilized_cycles: int = 0
    demand_queue_wait_cycles: int = 0
    demand_blocked_by_speculative_cycles: int = 0
    speculative_requests_issued: int = 0
    speculative_requests_committed: int = 0
    speculative_requests_canceled: int = 0
    speedup_vs_demand_serial: float | None = None
    speedup_vs_exact_pipeline: float | None = None
    speedup_vs_ideal: float | None = None
    tile_mode: bool = False
    tile_size_bytes: int | None = None
    total_tiles_required: int = 0
    total_tiles_transferred: int = 0
    demand_tiles: int = 0
    speculative_tiles: int = 0
    useful_speculative_tiles: int = 0
    wasted_speculative_tiles: int = 0
    canceled_speculative_tiles: int = 0
    true_tiles_resident_at_resolve: int = 0
    true_tile_resident_fraction_at_resolve: float = 0.0
    true_experts_first_tile_ready_at_resolve: int = 0
    aie_first_tile_stall_cycles: int = 0
    aie_inter_tile_stall_cycles: int = 0
    aie_gate_up_tile_stall_cycles: int = 0
    aie_down_tile_stall_cycles: int = 0
    mean_useful_prefetch_lead_cycles: float | None = None


@dataclass
class LayerTimingRecord:
    """Absolute milestones for one layer in a continuous sequence."""

    request_id: int
    token_position: int
    layer_id: int
    layer_start_cycle: int
    router_start_cycle: int = 0
    router_done_cycle: int = 0
    topk_done_cycle: int = 0
    resolve_cycle: int = 0
    first_expert_start_cycle: int | None = None
    gate_up_done_cycle: int | None = None
    expert_phase_done_cycle: int | None = None
    layer_done_cycle: int = 0


@dataclass
class PredictionMetrics:
    """Quality, lead-time, and target-readiness metrics for one prediction."""

    prediction_id: str
    source_layer_id: int
    target_layer_id: int
    prediction_ready_cycle: int
    target_layer_start_cycle: int
    target_router_start_cycle: int
    target_resolve_cycle: int
    prediction_lead_to_layer_start: int
    prediction_lead_to_router_start: int
    prediction_lead_to_resolve: int
    configured_candidate_count: int
    configured_generation_mode: str
    configured_target_recall: float | None
    sampled_hit_count: int
    static_recall: float
    any_hit: bool
    full_hit: bool
    candidate_ranks_of_true_experts: tuple[int, ...]
    ready_at_source_layer_end: float
    ready_at_target_layer_start: float
    ready_at_target_router_start: float
    ready_at_target_resolve: float
    ready_at_scheduled_start: float


@dataclass
class SequenceSimulationResult:
    """Metrics and records for one continuous multi-layer execution."""

    scenario: str
    request_id: int
    token_position: int
    start_cycle: int
    finish_cycle: int
    total_cycles: int
    layer_results: list[SimulationResult]
    layer_timings: list[LayerTimingRecord]
    prediction_events: list[PredictionEvent]
    prediction_metrics: list[PredictionMetrics]
    total_demand_bytes: int
    total_speculative_bytes: int
    total_useful_speculative_bytes: int
    total_wasted_speculative_bytes: int
    total_canceled_speculative_bytes: int
    total_aie_memory_stall_cycles: int
    total_demand_blocked_by_future_speculative_cycles: int
    future_speculative_bytes: int = 0
    useful_future_speculative_bytes: int = 0
    wasted_future_speculative_bytes: int = 0
    canceled_future_speculative_bytes: int = 0
    target_layer_cycles: int = 0
    target_layer_memory_stall_cycles: int = 0
    target_true_tile_resident_fraction_at_source_end: float | None = None
    target_true_tile_resident_fraction_at_target_start: float | None = None
    target_true_tile_resident_fraction_at_resolve: float | None = None
    speedup_vs_sequence_exact: float | None = None
    speedup_vs_cross_layer_oracle: float | None = None
    target_stall_reduction_cycles: int | None = None
