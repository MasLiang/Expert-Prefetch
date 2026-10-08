"""Predictor adapters for speculative expert prefetch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .schema import CandidatePrefetch, ModelConfig, PredictorOutput, RoutingEvent, ValidationError


class Predictor(Protocol):
    """Interface producing candidates relative to layer start."""

    def predict(self, routing_event: RoutingEvent, layer_start_cycle: int) -> PredictorOutput | None:
        """Return a predictor output or ``None``."""
        ...


class NoPredictor:
    """Predictor adapter that never emits candidates."""

    def predict(self, routing_event: RoutingEvent, layer_start_cycle: int) -> PredictorOutput | None:
        """Always return ``None``."""
        return None


class OraclePredictor:
    """Emit true experts at a configurable offset without changing routing."""

    def __init__(self, ready_offset_cycles: int = 0) -> None:
        if ready_offset_cycles < 0:
            raise ValidationError("oracle ready_offset_cycles must be non-negative")
        self.ready_offset_cycles = ready_offset_cycles

    def predict(self, routing_event: RoutingEvent, layer_start_cycle: int) -> PredictorOutput:
        """Return all true experts with their true gating weights."""
        return PredictorOutput(
            routing_event.request_id,
            routing_event.token_position,
            routing_event.layer_id,
            routing_event.true_expert_ids,
            routing_event.true_gating_weights,
            self.ready_offset_cycles,
        )


class ExternalTracePredictor:
    """Lookup validated predictor outputs loaded from JSONL."""

    def __init__(
        self,
        path: str | Path,
        model: ModelConfig,
        *,
        allow_missing: bool = False,
    ) -> None:
        self.model = model
        self.allow_missing = allow_missing
        self._outputs: dict[tuple[int, int, int], PredictorOutput] = {}
        input_path = Path(path)
        try:
            lines = input_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError(f"cannot read predictor trace {input_path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if "candidates" in raw:
                    candidates = tuple(CandidatePrefetch(
                        expert_id=item["expert_id"],
                        score=item["score"],
                        gate_up_tiles=item.get("gate_up_tiles"),
                        down_tiles=item.get("down_tiles"),
                    ) for item in raw["candidates"])
                    output = PredictorOutput(
                        request_id=raw["request_id"],
                        token_position=raw["token_position"],
                        layer_id=raw["layer_id"],
                        ready_offset_cycles=raw["ready_offset_cycles"],
                        candidates=candidates,
                    )
                else:
                    output = PredictorOutput(
                        request_id=raw["request_id"],
                        token_position=raw["token_position"],
                        layer_id=raw["layer_id"],
                        candidate_ids=tuple(raw["candidate_ids"]),
                        candidate_scores=tuple(raw["candidate_scores"]),
                        ready_offset_cycles=raw["ready_offset_cycles"],
                    )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValidationError(f"invalid predictor JSONL line {line_number}: {exc}") from exc
            output.validate(model)
            key = (output.request_id, output.token_position, output.layer_id)
            if key in self._outputs:
                raise ValidationError(f"duplicate predictor trace key {key} at line {line_number}")
            self._outputs[key] = output

    def predict(self, routing_event: RoutingEvent, layer_start_cycle: int) -> PredictorOutput | None:
        """Return the unique matching record or apply explicit missing fallback."""
        key = (routing_event.request_id, routing_event.token_position, routing_event.layer_id)
        output = self._outputs.get(key)
        if output is None and not self.allow_missing:
            raise ValidationError(f"predictor trace has no record for key {key}")
        return output
