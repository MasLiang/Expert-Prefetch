"""Milestone 6 execution configuration and comprehensive result schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Mapping

from .batch_routing import BatchWorkloadStats
from .roofline_metrics import RooflineMetrics
from .schema import ValidationError
from .execution_rules import ExecutionRules
from .prefetch_config import PrefetchConfig


class AdvancedScenario(str, Enum):
    """Batch/F-tile scenarios used by Roofline and prefetch DSE."""

    DEMAND_SERIAL = "demand_serial"
    EXACT_PIPELINE = "exact_pipeline"
    EXACT_FULL_EXPERT_LOAD = "exact_full_expert_load"
    EXACT_TILE_STREAMING = "exact_tile_streaming"
    PREDICTIVE_FULL_EXPERT = "predictive_full_expert"
    PREDICTIVE_UNIFORM_PREFIX = "predictive_uniform_prefix"
    PREDICTIVE_RANKED_PREFIX = "predictive_ranked_prefix"
    PREDICTIVE_DSE = "predictive_DSE"
    ORACLE_OPTIMAL_DEPTH = "OracleOptimalDepth"
    IDEAL = "ideal"


@dataclass(frozen=True)
class AdvancedExecutionConfig:
    """Streaming, prefetch, memory placement, and AIE allocation controls."""

    f_tile_size: int
    placement_policy: str = "round_robin_expert"
    aie_units_per_expert: int = 1
    max_parallel_experts: int = 1
    router_aie_units: int = 1
    startup_gu_tiles: int = 1
    candidate_count: int = 1
    uniform_gu_prefetch_depth: int = 1
    down_prefetch_depth: int = 0
    rank_based_depths: tuple[int, ...] = ()
    explicit_depths: tuple[int, ...] = ()
    confidence_depth_scale: float = 1.0
    speculative_byte_budget: int | None = None
    coalesce_adjacent_tiles: bool = False
    nonlinear_elements_per_cycle: int = 1024
    nonlinear_startup_cycles: int = 0
    roofline_balance_tolerance: float = 0.10
    execution_rules: ExecutionRules = field(default_factory=ExecutionRules)
    demand_loading: Literal["tile_streaming", "tile_serial", "full_expert_load"] = "tile_streaming"
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.prefetch, PrefetchConfig):
            raise ValidationError('prefetch must be a PrefetchConfig dataclass')
        if self.demand_loading not in {"tile_streaming", "tile_serial", "full_expert_load"}:
            raise ValidationError("unknown demand_loading policy")
        if not isinstance(self.execution_rules, ExecutionRules):
            raise ValidationError('execution_rules 必须为 ExecutionRules dataclass')
        for name in (
            "f_tile_size", "aie_units_per_expert", "max_parallel_experts",
            "router_aie_units", "startup_gu_tiles", "candidate_count",
            "nonlinear_elements_per_cycle",
        ):
            if getattr(self, name) <= 0:
                raise ValidationError(f"advanced.{name} must be positive")
        for name in ("uniform_gu_prefetch_depth", "down_prefetch_depth", "nonlinear_startup_cycles"):
            if getattr(self, name) < 0:
                raise ValidationError(f"advanced.{name} must be non-negative")
        if self.speculative_byte_budget is not None and self.speculative_byte_budget < 0:
            raise ValidationError("speculative_byte_budget must be non-negative")


@dataclass
class AdvancedExpertRecord:
    """Per-expert reuse, streaming readiness, and latency record."""

    expert_id: int
    token_count: int
    weight_bytes: int
    compute_ops: int
    first_tile_ready_time: int | None = None
    compute_start_time: int | None = None
    finish_time: int | None = None
    initial_memory_stall: int = 0
    inter_tile_starvation_cycles: int = 0
    total_aie_wait_for_weights: int = 0
    streaming_overlap_cycles: int = 0

    @property
    def expert_first_tile_ready_time(self) -> int | None:
        """Return the first startup GU tile readiness time."""
        return self.first_tile_ready_time

    @property
    def expert_compute_start_time(self) -> int | None:
        """Return the first true tile compute start time."""
        return self.compute_start_time

    @property
    def expert_finish_time(self) -> int | None:
        """Return the final Down tile completion time."""
        return self.finish_time


@dataclass
class AdvancedSimulationResult:
    """One batch-layer execution with resource, prediction, and Roofline metrics."""

    scenario: str
    request_id: int
    layer_id: int
    router_resolve_cycle: int
    first_true_expert_start: int
    last_true_expert_start: int
    layer_makespan_cycles: int
    workload: BatchWorkloadStats
    expert_records: list[AdvancedExpertRecord]
    per_expert_token_counts: Mapping[int, int]
    recall_at_m: float | None
    any_hit: bool | None
    full_hit: bool | None
    startable_recall: float
    weighted_startable_recall: float
    any_startable_hit: bool
    full_startable_hit: bool
    true_expert_prefetched_fraction: float
    resident_fraction_at_router_resolve: float
    full_expert_cache_hit: float
    true_expert_rank_statistics: Mapping[str, float | int | None]
    average_prefix_depth: float
    weighted_prefix_depth: float
    memory_metrics: Mapping[str, object]
    aie_metrics: Mapping[str, float | int]
    aie_starvation_cycles: int
    compute_cycles: int
    dma_compute_overlap_cycles: int
    prediction_ready_time: int | None
    prefetch_window_cycles: int
    bytes_transferred_during_prefetch_window: int
    useful_bytes_transferred_during_prefetch_window: int
    memory_idle_cycles_during_prefetch_window: int
    prefetch_window_utilization: float
    useful_prefetch_window_utilization: float
    peak_ocm_bytes: int
    average_ocm_bytes: float
    rejected_prefetch_bytes: int
    evicted_speculative_bytes: int
    demand_blocked_cycles: int
    speculative_blocked_cycles: int
    roofline: RooflineMetrics
    hardware_metadata: Mapping[str, object]
    timeline: list[object] = field(default_factory=list)
    startup_readiness: Mapping[int, float] = field(default_factory=dict)
    startup_gap_bytes: Mapping[int, int] = field(default_factory=dict)
    mean_startup_readiness: float = 0.0
    weighted_startup_readiness: float = 0.0
    total_startup_gap_bytes: int = 0
    tile_metrics: Mapping[str, object] = field(default_factory=dict)
    tile_prefetch_plan: Mapping[str, object] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        """Return the routed batch size."""
        return self.workload.batch_size

    @property
    def router_top_k(self) -> int:
        """Return exact Router Top-K."""
        return self.workload.router_top_k

    @property
    def reuse_factor(self) -> float:
        """Return token-expert assignments per unique active expert."""
        return self.workload.reuse_factor

    @property
    def router_resolve_to_first_expert_start(self) -> int:
        """Return post-resolve launch latency in cycles."""
        return self.first_true_expert_start - self.router_resolve_cycle

    @property
    def total_aie_wait_for_weights(self) -> int:
        """Return aggregate expert weight-wait cycles."""
        return self.aie_starvation_cycles

    @property
    def streaming_overlap_cycles(self) -> int:
        """Return global DMA/AIE overlap cycles."""
        return self.dma_compute_overlap_cycles

    @property
    def speculative_bytes(self) -> int:
        """Return physically transferred speculative bytes."""
        return int(self.memory_metrics["speculative_bytes"])

    @property
    def useful_prefetched_bytes(self) -> int:
        """Return useful speculative transfer bytes."""
        return int(self.memory_metrics["useful_speculative_bytes"])

    @property
    def false_prefetched_bytes(self) -> int:
        """Return transferred false speculative bytes."""
        return int(self.memory_metrics["wasted_speculative_bytes"])

    @property
    def resident_bytes_at_router_resolve(self) -> int:
        """Return true required bytes resident at exact resolve."""
        required = sum(record.weight_bytes for record in self.expert_records)
        return round(self.resident_fraction_at_router_resolve * required)
