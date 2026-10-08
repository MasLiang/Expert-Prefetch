"""Deterministic resource timeline records and CSV output."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TimelineEntry:
    """One half-open resource interval ``[start_cycle, end_cycle)``."""

    resource: str
    operation: str
    start_cycle: int
    end_cycle: int
    request_id: int
    layer_id: int
    expert_id: int | None = None
    request_type: str = ""
    useful: bool | None = None
    committed: bool | None = None
    candidate_rank: int | None = None
    tile_id: int | None = None
    num_tiles: int | None = None
    tile_size_bytes: int | None = None
    compute_ops: int | None = None
    starvation_cycles: int | None = None
    ready_cycle: int | None = None
    source_layer_id: int | None = None
    target_layer_id: int | None = None
    source_token_position: int | None = None
    target_token_position: int | None = None
    prediction_stage: str = ""
    speculative_scope: str = ""
    prediction_id: str = ""


def write_timeline_csv(entries: list[TimelineEntry], path: str | Path) -> None:
    """Write timeline entries in deterministic order to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TimelineEntry.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted(
            entries,
            key=lambda item: (
                item.start_cycle, item.resource, item.operation,
                -1 if item.expert_id is None else item.expert_id,
            ),
        ):
            writer.writerow(asdict(entry))
