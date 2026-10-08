"""Progressive tile-prefetch allocation policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .schema import CandidatePrefetch, PredictorOutput, ValidationError
from .tiling import ExpertTileSet


@dataclass(frozen=True)
class TilePrefetchDecision:
    """A deterministic set of tiles allocated to one candidate matrix."""

    expert_id: int
    matrix_type: str
    tile_ids: tuple[int, ...]
    score: float
    candidate_rank: int


class PrefetchPolicy(Protocol):
    """Interface mapping ranked candidates and a budget to tile decisions."""

    def allocate(
        self,
        predictor_output: PredictorOutput,
        expert_tiles: Mapping[int, ExpertTileSet],
        byte_budget: int | None,
    ) -> list[TilePrefetchDecision]:
        """Allocate whole tiles without exceeding the optional budget."""
        ...


def _ranked(output: PredictorOutput) -> list[tuple[int, CandidatePrefetch]]:
    limited = list(enumerate(output.candidates))
    return sorted(limited, key=lambda pair: (-pair[1].score, pair[0], pair[1].expert_id))


def _depth_decisions(
    output: PredictorOutput,
    expert_tiles: Mapping[int, ExpertTileSet],
    gate_depths: list[int],
    down_depths: list[int],
) -> list[TilePrefetchDecision]:
    decisions: list[TilePrefetchDecision] = []
    for rank, candidate in enumerate(output.candidates):
        tile_set = expert_tiles.get(candidate.expert_id)
        if tile_set is None:
            raise ValidationError(f"no tile geometry for candidate expert {candidate.expert_id}")
        for matrix_type, tiles, depths in (
            ("gate_up", tile_set.gate_up_tiles, gate_depths),
            ("down", tile_set.down_tiles, down_depths),
        ):
            depth = depths[rank] if rank < len(depths) else 0
            tile_ids = tuple(range(min(depth, len(tiles))))
            if tile_ids:
                decisions.append(TilePrefetchDecision(
                    candidate.expert_id, matrix_type, tile_ids, candidate.score, rank
                ))
    return decisions


class FixedDepthPolicy:
    """Allocate the same Gate-Up and Down depth to every candidate."""

    def __init__(self, gate_up_tiles_per_candidate: int = 2, down_tiles_per_candidate: int = 0) -> None:
        if gate_up_tiles_per_candidate < 0 or down_tiles_per_candidate < 0:
            raise ValidationError("fixed prefetch depths must be non-negative")
        self.gate_depth = gate_up_tiles_per_candidate
        self.down_depth = down_tiles_per_candidate

    def allocate(
        self, predictor_output: PredictorOutput, expert_tiles: Mapping[int, ExpertTileSet],
        byte_budget: int | None,
    ) -> list[TilePrefetchDecision]:
        """Return equal-depth prefix decisions."""
        count = len(predictor_output.candidates)
        decisions = _depth_decisions(
            predictor_output, expert_tiles,
            [self.gate_depth] * count, [self.down_depth] * count,
        )
        return _enforce_budget(decisions, expert_tiles, byte_budget)


class ProgressiveRankPolicy:
    """Allocate configured depths by predictor rank."""

    def __init__(self, gate_up_depths: tuple[int, ...], down_depths: tuple[int, ...]) -> None:
        self.gate_depths = list(gate_up_depths)
        self.down_depths = list(down_depths)

    def allocate(
        self, predictor_output: PredictorOutput, expert_tiles: Mapping[int, ExpertTileSet],
        byte_budget: int | None,
    ) -> list[TilePrefetchDecision]:
        """Return rank-dependent prefix decisions."""
        decisions = _depth_decisions(
            predictor_output, expert_tiles, self.gate_depths, self.down_depths
        )
        return _enforce_budget(decisions, expert_tiles, byte_budget)


class PredictorSpecifiedPolicy:
    """Use candidate-specific depths with optional explicit fallbacks."""

    def __init__(self, default_gate_up_tiles: int | None, default_down_tiles: int | None) -> None:
        self.default_gate = default_gate_up_tiles
        self.default_down = default_down_tiles

    def allocate(
        self, predictor_output: PredictorOutput, expert_tiles: Mapping[int, ExpertTileSet],
        byte_budget: int | None,
    ) -> list[TilePrefetchDecision]:
        """Allocate predictor depths or fail when no fallback is configured."""
        gate_depths: list[int] = []
        down_depths: list[int] = []
        for candidate in predictor_output.candidates:
            gate = candidate.gate_up_tiles if candidate.gate_up_tiles is not None else self.default_gate
            down = candidate.down_tiles if candidate.down_tiles is not None else self.default_down
            if gate is None or down is None:
                raise ValidationError(
                    f"candidate {candidate.expert_id} lacks tile depth and no fallback is configured"
                )
            gate_depths.append(gate)
            down_depths.append(down)
        return _enforce_budget(
            _depth_decisions(predictor_output, expert_tiles, gate_depths, down_depths),
            expert_tiles, byte_budget,
        )


class FixedByteBudgetPolicy:
    """Allocate whole tiles in score-ranked breadth-first rounds."""

    def __init__(self, include_down: bool = False) -> None:
        self.include_down = include_down

    def allocate(
        self, predictor_output: PredictorOutput, expert_tiles: Mapping[int, ExpertTileSet],
        byte_budget: int | None,
    ) -> list[TilePrefetchDecision]:
        """Allocate tile zero to each candidate, then deeper round-robin prefixes."""
        if byte_budget is None:
            raise ValidationError("fixed_byte_budget policy requires byte_budget")
        ranked = _ranked(predictor_output)
        allocated: dict[tuple[int, str], list[int]] = {}
        used_bytes = 0

        def rounds(matrix_type: str) -> None:
            nonlocal used_bytes
            depth = 0
            while True:
                any_available = False
                allocated_this_round = False
                for original_rank, candidate in ranked:
                    tile_set = expert_tiles[candidate.expert_id]
                    tiles = tile_set.gate_up_tiles if matrix_type == "gate_up" else tile_set.down_tiles
                    if depth >= len(tiles):
                        continue
                    if depth > 0 and depth - 1 not in allocated.get(
                        (candidate.expert_id, matrix_type), []
                    ):
                        continue
                    any_available = True
                    tile = tiles[depth]
                    if used_bytes + tile.size_bytes <= byte_budget:
                        allocated.setdefault((candidate.expert_id, matrix_type), []).append(depth)
                        used_bytes += tile.size_bytes
                        allocated_this_round = True
                if not any_available or not allocated_this_round:
                    break
                depth += 1

        rounds("gate_up")
        if self.include_down and used_bytes < byte_budget:
            rounds("down")
        candidate_by_id = {candidate.expert_id: (rank, candidate) for rank, candidate in enumerate(predictor_output.candidates)}
        return [
            TilePrefetchDecision(expert_id, matrix_type, tuple(tile_ids), candidate.score, rank)
            for (expert_id, matrix_type), tile_ids in allocated.items()
            for rank, candidate in [candidate_by_id[expert_id]]
        ]


def _enforce_budget(
    decisions: list[TilePrefetchDecision],
    expert_tiles: Mapping[int, ExpertTileSet],
    byte_budget: int | None,
) -> list[TilePrefetchDecision]:
    if byte_budget is None:
        return decisions
    used = 0
    result: list[TilePrefetchDecision] = []
    for decision in decisions:
        tiles = (
            expert_tiles[decision.expert_id].gate_up_tiles
            if decision.matrix_type == "gate_up"
            else expert_tiles[decision.expert_id].down_tiles
        )
        accepted: list[int] = []
        for tile_id in decision.tile_ids:
            size = tiles[tile_id].size_bytes
            if used + size > byte_budget:
                break
            accepted.append(tile_id)
            used += size
        if accepted:
            result.append(TilePrefetchDecision(
                decision.expert_id, decision.matrix_type, tuple(accepted),
                decision.score, decision.candidate_rank,
            ))
    return result


def policy_from_config(config) -> PrefetchPolicy:
    """Construct a tile prefetch policy from ``PrefetchConfig``."""
    if config.policy == "fixed_depth":
        return FixedDepthPolicy(config.gate_up_tiles_per_candidate, config.down_tiles_per_candidate)
    if config.policy == "fixed_byte_budget":
        return FixedByteBudgetPolicy(config.include_down)
    if config.policy == "progressive_rank":
        return ProgressiveRankPolicy(config.gate_up_depths, config.down_depths)
    return PredictorSpecifiedPolicy(
        config.predictor_default_gate_up_tiles, config.predictor_default_down_tiles
    )
