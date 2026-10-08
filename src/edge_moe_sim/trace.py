"""Routing JSONL input and independent-event aggregate metrics."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .schema import LayerSequence, ModelConfig, RoutingEvent, SimulationResult, ValidationError


def load_routing_jsonl(path: str | Path, model: ModelConfig) -> list[RoutingEvent]:
    """Load and validate decode routing events from JSONL."""
    input_path = Path(path)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read routing trace {input_path}: {exc}") from exc
    events: list[RoutingEvent] = []
    keys: set[tuple[int, int, int]] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            counts = {int(key): value for key, value in raw["expert_token_counts"].items()}
            event = RoutingEvent(
                raw["request_id"], raw["token_position"], raw["layer_id"], raw["phase"],
                tuple(raw["true_expert_ids"]), tuple(raw["true_gating_weights"]), counts,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid routing JSONL line {line_number}: {exc}") from exc
        event.validate(model)
        key = (event.request_id, event.token_position, event.layer_id)
        if key in keys:
            raise ValidationError(f"duplicate routing trace key {key} at line {line_number}")
        keys.add(key)
        events.append(event)
    if not events:
        raise ValidationError("routing trace is empty")
    return events


def load_layer_sequences_jsonl(path: str | Path, model: ModelConfig) -> list[LayerSequence]:
    """Group routing JSONL rows into ordered same-request, same-token sequences."""
    events = load_routing_jsonl(path, model)
    grouped: dict[tuple[int, int], list[RoutingEvent]] = {}
    for event in events:
        grouped.setdefault((event.request_id, event.token_position), []).append(event)
    sequences: list[LayerSequence] = []
    for (request_id, token_position), records in sorted(grouped.items()):
        sequence = LayerSequence(
            request_id, token_position, tuple(sorted(records, key=lambda record: record.layer_id))
        )
        sequence.validate(model)
        sequences.append(sequence)
    return sequences


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass
class TraceSimulationResult:
    """Aggregate of independent per-event simulations."""

    scenario: str
    num_events: int
    total_cycles: int
    mean_cycles: float
    median_cycles: float
    p95_cycles: float
    p99_cycles: float
    total_demand_bytes: int
    total_speculative_bytes: int
    mean_static_recall: float | None
    full_hit_rate: float | None
    mean_ready_recall: float | None
    total_aie_memory_stall_cycles: int
    cycle_semantics: str
    per_event: list[SimulationResult]
    tile_mode: bool = False
    tile_size_bytes: int | None = None
    total_tiles_required: int = 0
    total_tiles_transferred: int = 0
    total_demand_tiles: int = 0
    total_speculative_tiles: int = 0
    mean_true_tile_resident_fraction_at_resolve: float = 0.0
    total_aie_first_tile_stall_cycles: int = 0
    total_aie_inter_tile_stall_cycles: int = 0

    def to_dict(self, include_per_event: bool = False) -> dict[str, Any]:
        """Convert the aggregate to JSON-compatible data."""
        data = asdict(self)
        if not include_per_event:
            data.pop("per_event")
        return data


def aggregate_results(results: Sequence[SimulationResult]) -> TraceSimulationResult:
    """Aggregate results whose cycles are independent, not concurrent wall time."""
    if not results:
        raise ValueError("cannot aggregate an empty result sequence")
    cycles = [result.total_cycles for result in results]
    recalls = [result.static_recall for result in results if result.static_recall is not None]
    ready = [result.ready_recall_at_resolve for result in results if result.ready_recall_at_resolve is not None]
    hits = [result.full_hit for result in results if result.full_hit is not None]
    return TraceSimulationResult(
        results[0].scenario,
        len(results),
        sum(cycles),
        statistics.fmean(cycles),
        statistics.median(cycles),
        _percentile(cycles, 0.95),
        _percentile(cycles, 0.99),
        sum(result.demand_bytes for result in results),
        sum(result.speculative_bytes for result in results),
        statistics.fmean(recalls) if recalls else None,
        statistics.fmean(1.0 if hit else 0.0 for hit in hits) if hits else None,
        statistics.fmean(ready) if ready else None,
        sum(result.aie_memory_stall_cycles for result in results),
        "sum of independent event cycles; not continuous-batching wall-clock time",
        list(results),
        all(result.tile_mode for result in results),
        results[0].tile_size_bytes if all(
            result.tile_size_bytes == results[0].tile_size_bytes for result in results
        ) else None,
        sum(result.total_tiles_required for result in results),
        sum(result.total_tiles_transferred for result in results),
        sum(result.demand_tiles for result in results),
        sum(result.speculative_tiles for result in results),
        statistics.fmean(
            result.true_tile_resident_fraction_at_resolve for result in results
        ),
        sum(result.aie_first_tile_stall_cycles for result in results),
        sum(result.aie_inter_tile_stall_cycles for result in results),
    )
