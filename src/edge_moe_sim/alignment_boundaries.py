"""Shared RTL/performance-model functional boundaries, all integer cycle timestamps.

Metrics use half-open intervals, not per-edge state counters. This contract is
independent of either scheduler so a future main-model backend can reuse it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def analyze_boundaries(events: Sequence[Mapping[str, Any]], truth: Sequence[int],
                       tiles_per_expert: int, *, tile_start: str = "gu_and_down_ready") -> dict[str, Any]:
    """Validate a single-layer lifecycle and derive comparable per-tile timings.

Window-open means true work can be scheduled (not running or combine-blocked).
Payload-ready means physical bytes complete, not promotion or snapshot ready.
Acceptance is the compute request handshake; GU-start is a separate boundary.
An unmodeled acceptance-to-GU stage may explicitly have zero duration.
"""
    if tiles_per_expert <= 0 or not truth or len(set(truth)) != len(truth):
        raise ValueError("invalid boundary geometry")
    if tile_start not in ("gu_and_down_ready","gu_only_ready"):
        raise ValueError("unknown tile start rule")
    startup_matrices = ("gate_up",) if tile_start == "gu_only_ready" else ("gate_up","down")
    route = [int(e["cycle"]) for e in events if e["event"] == "ROUTE_RESOLVED"]
    windows = sorted(int(e["cycle"]) for e in events if e["event"] == "TILE_ISSUE_WINDOW_OPEN")
    count = len(truth)*tiles_per_expert
    if len(route) != 1 or len(windows) != count or len(set(windows)) != count:
        raise ValueError("missing/duplicate route or issue-window boundaries")
    by_tile: dict[tuple[int, int], dict[str, int]] = {}
    stages: dict[tuple[int, int, str], dict[str, int]] = {}
    tile_kinds = {"COMPUTE_REQUEST_ACCEPTED", "GU_START", "GU_DONE", "DOWN_START", "DOWN_DONE", "TILE_RETIRED"}
    stage_kinds = {"REQUEST_QUEUED", "MEMORY_SERVICE_START", "PAYLOAD_READY"}
    combines: dict[int, dict[str, int]] = {}
    for event in events:
        kind = str(event["event"])
        cycle = event["cycle"]
        if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0:
            raise ValueError("boundary cycles must be nonnegative integers")
        expert, tile = int(event["expert_id"]), int(event["tile_id"])
        if kind in tile_kinds:
            if expert not in truth or not 0 <= tile < tiles_per_expert:
                raise ValueError("compute boundary for non-true/out-of-range tile")
            values = by_tile.setdefault((expert,tile), {})
        elif kind in stage_kinds:
            if event["matrix"] not in ("gate_up", "down"):
                raise ValueError("invalid stage boundary matrix")
            values = stages.setdefault((expert,tile,str(event["matrix"])), {})
        elif kind in ("EXPERT_COMBINE_START", "EXPERT_COMBINE_DONE"):
            values = combines.setdefault(expert,{})
        else:
            continue
        if kind in values:
            raise ValueError(f"duplicate {kind} boundary")
        values[kind] = cycle
    expected = {(e,t) for e in truth for t in range(tiles_per_expert)}
    if set(by_tile) != expected or set(combines) != set(truth):
        raise ValueError("incomplete required tile/combine boundaries")
    for values in stages.values():
        if set(values) != stage_kinds or not (values["REQUEST_QUEUED"] <= values["MEMORY_SERVICE_START"] <= values["PAYLOAD_READY"]):
            raise ValueError("missing/reversed memory lifecycle")
    for key, values in by_tile.items():
        if set(values) != tile_kinds:
            raise ValueError(f"missing tile boundary: {key}")
        times = [values[k] for k in ("COMPUTE_REQUEST_ACCEPTED", "GU_START", "GU_DONE", "DOWN_START", "DOWN_DONE", "TILE_RETIRED")]
        if times != sorted(times):
            raise ValueError("reversed compute lifecycle")
    order = sorted(by_tile, key=lambda key: by_tile[key]["COMPUTE_REQUEST_ACCEPTED"])
    next_tiles = {e:0 for e in truth}
    previous_retire = route[0]
    previous_block_end = route[0]
    records: list[dict[str, int]] = []
    timeline: list[dict[str, Any]] = []
    for (expert,tile), window in zip(order,windows):
        values = by_tile[(expert,tile)]
        accepted = values["COMPUTE_REQUEST_ACCEPTED"]
        if tile != next_tiles[expert] or window < previous_block_end or accepted < window:
            raise ValueError("prefix/serial issue-window causality violated")
        try:
            pair_ready = max(stages[(expert,tile,m)]["PAYLOAD_READY"] for m in ("gate_up","down"))
            startup_ready = max(stages[(expert,tile,m)]["PAYLOAD_READY"] for m in startup_matrices)
            earliest = min(max(stages[(e,next_tiles[e],m)]["PAYLOAD_READY"] for m in startup_matrices)
                           for e in truth if next_tiles[e] < tiles_per_expert)
        except KeyError as exc:
            raise ValueError("missing required weight boundary") from exc
        if startup_ready > accepted or stages[(expert,tile,"down")]["PAYLOAD_READY"] > values["DOWN_START"]:
            raise ValueError("compute stage started before its required payload ready")
        memory_end = min(accepted,max(window,earliest))
        row = dict(expert_id=expert,tile_id=tile,issue_window_open_cycle=window,
                   pair_payload_ready_cycle=pair_ready,compute_request_accepted_cycle=accepted,
                   startup_payload_ready_cycle=startup_ready,
                   gu_start_cycle=values["GU_START"],gu_done_cycle=values["GU_DONE"],
                   down_start_cycle=values["DOWN_START"],down_done_cycle=values["DOWN_DONE"],
                   tile_retired_cycle=values["TILE_RETIRED"],
                   memory_wait_cycles=memory_end-window,
                   ready_to_accept_cycles=accepted-memory_end,
                   accept_to_gu_cycles=values["GU_START"]-accepted,
                   down_done_to_retire_cycles=values["TILE_RETIRED"]-values["DOWN_DONE"],
                   previous_retire_to_window_cycles=window-previous_retire)
        records.append(row)
        for operation,start,end in (("memory_wait",window,memory_end),
                                    ("ready_to_accept",memory_end,accepted),
                                    ("accept_to_gu",accepted,values["GU_START"])):
            if end > start:
                timeline.append(dict(resource="CANONICAL_ISSUE",operation=operation,start_cycle=start,
                    end_cycle=end,expert_id=expert,tile_id=tile,bytes=0,request_type="none"))
        next_tiles[expert] += 1
        previous_retire = values["TILE_RETIRED"]
        previous_block_end = previous_retire
        if tile == tiles_per_expert-1:
            combine = combines[expert]
            if set(combine) != {"EXPERT_COMBINE_START","EXPERT_COMBINE_DONE"} or not (
                previous_retire <= combine["EXPERT_COMBINE_START"] <= combine["EXPERT_COMBINE_DONE"]):
                raise ValueError("missing/reversed Expert combine lifecycle")
            previous_block_end = combine["EXPERT_COMBINE_DONE"]
    totals = {key:sum(row[key] for row in records) for key in (
        "memory_wait_cycles","ready_to_accept_cycles","accept_to_gu_cycles","down_done_to_retire_cycles",
        "previous_retire_to_window_cycles")}
    totals["ready_to_gu_cycles"] = totals["ready_to_accept_cycles"]+totals["accept_to_gu_cycles"]
    totals["issue_window_cycles"] = totals["memory_wait_cycles"]+totals["ready_to_accept_cycles"]
    totals["expert_combine_cycles"] = sum(c["EXPERT_COMBINE_DONE"]-c["EXPERT_COMBINE_START"] for c in combines.values())
    stage_records = [dict(expert_id=e,tile_id=t,matrix=matrix,
                          request_queued_cycle=v["REQUEST_QUEUED"],
                          memory_service_start_cycle=v["MEMORY_SERVICE_START"],
                          payload_ready_cycle=v["PAYLOAD_READY"],
                          queue_wait_cycles=v["MEMORY_SERVICE_START"]-v["REQUEST_QUEUED"],
                          service_to_payload_cycles=v["PAYLOAD_READY"]-v["MEMORY_SERVICE_START"])
                     for (e,t,matrix),v in sorted(stages.items())]
    return dict(contract_version="functional_boundaries_v1",tile_start=tile_start,route_resolved_cycle=route[0],
                totals=totals,tile_records=records,stage_records=stage_records,timeline=timeline)
