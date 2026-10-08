"""失效的历史 M10 pair-ready 对齐模型；不适用于当前 GU-ready/跨层 RTL。

Historical simulators remain unchanged. This adapter reuses their heap, compute,
striped memory, OCM and integer allocation models, not RTL completion timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping

from .event_queue import EventQueue
from .events import EventType
from .hardware.aie_pool import AIEPool, AIEPoolTask
from .hardware.capacity_ocm import CapacityOnChipMemory
from .hardware.multi_channel_memory import MemoryRequest, MultiChannelMemory
from .hardware_greedy_v1 import HardwareGreedyV1Candidate, HardwareGreedyV1Config, hardware_greedy_v1
from .hardware_profile import HardwareProfile
from .schema import ModelConfig, ValidationError
from .work_model import router_ops

Key = tuple[int, int, str, int]


class BurstServiceProfile(HardwareProfile):
    """Charge setup for each 256-byte physical burst, without per-beat simulation."""

    def dma_cycles(self, transfer_bytes: int, channel_count: int = 1) -> int:
        """Return the sum of configured setup plus payload cycles per burst."""
        if transfer_bytes <= 0 or channel_count <= 0:
            raise ValidationError("positive striped transfer required")
        return sum(super(BurstServiceProfile, self).dma_cycles(
            min(self.burst_bytes, transfer_bytes-offset), channel_count)
            for offset in range(0, transfer_bytes, self.burst_bytes))


@dataclass
class AlignmentSimulationResult:
    """Event-level metrics, half-open resource intervals and allocation order."""

    total_cycles: int
    router_done_cycle: int
    truth_ready_cycle: int
    weight_wait_cycles: int
    demand_bytes: int
    speculative_bytes: int
    useful_speculative_bytes: int
    wasted_speculative_bytes: int
    canceled_speculative_bytes: int
    peak_ocm_slots: int
    allocations: list[tuple[int, int]]
    compute_order: list[tuple[int, int]]
    timeline: list[dict[str, Any]] = field(default_factory=list)
    peak_ocm_slots_per_bank: list[int] = field(default_factory=list)
    memory_wait_cycles: int = 0
    ready_control_wait_cycles: int = 0
    expert_issue_idle_cycles: int = 0
    boundary_events: list[dict[str, Any]] = field(default_factory=list)


class M10AlignmentSimulator:
    """Single-token/layer M10 adapter, holding two slots until tile completion.

    One descriptor is non-preemptive across its bursts. All four memory channels
    participate in every stage, so distinct descriptors cannot run concurrently.
    Both GU and DOWN must be resident before a tile starts. Ready tiles are
    selected in Router order, with per-Expert prefix dependency.
    """

    compatibility_status = "INVALID_FOR_CURRENT_RTL_HISTORICAL_ONLY"

    def __init__(self, parameters: Mapping[str, Any], *, ddr_gaps: bool,
                 allow_historical: bool = False) -> None:
        """Configure fixed constraints and independently specified service rates."""
        if not allow_historical:
            raise ValidationError(
                "M10AlignmentSimulator 已失效：不支持当前 GU-ready、跨层预取 RTL。"
                "请使用 UnifiedSequenceSimulator 与 compare_unified_alignment.py；"
                "仅复现历史 pair-ready 结果时显式设置 allow_historical=True。")
        self.parameters = dict(parameters)
        p = self.parameters
        for name in ("hidden_size", "intermediate_size", "num_experts", "top_k", "f_tile_size",
                     "ocm_slots", "ocm_slot_bytes", "descriptor_entries", "dma_fifo_depth"):
            if not isinstance(p[name], int) or isinstance(p[name], bool) or p[name] <= 0:
                raise ValidationError(f"alignment.{name} must be a positive integer")
        if p["physical_memory_channels"] != 4 or p["ocm_slots"] < 4 or p["reserved_demand_slots"] != 2:
            raise ValidationError("M10 adapter requires four striped banks and two reserved demand slots")
        h, f = p["hidden_size"], p["f_tile_size"]
        if h%16 or f%16 or p["intermediate_size"]%f or 2*h*f > p["ocm_slot_bytes"]:
            raise ValidationError("unsupported aligned tile/slot geometry")
        if p.get("fixture_profile") == "default_u200" and (
            p.get("ocm_banks", 0)*p.get("slots_per_bank", 0) != p["ocm_slots"]
            or p.get("ocm_banks") != 4 or p.get("slots_per_bank") != 32):
            raise ValidationError("default_u200 needs the actual 4x32 banked OCM geometry")
        delay = p.get("demand_release_delay_cycles", 0)
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValidationError("demand_release_delay_cycles must be nonnegative integer cycles")
        post_accept = p.get("request_accept_to_gu_cycles", 0)
        if not isinstance(post_accept, int) or isinstance(post_accept, bool) or not 0 <= post_accept <= p["tile_dispatch_cycles"]:
            raise ValidationError("request_accept_to_gu_cycles must be integer cycles within tile_dispatch_cycles")
        self.model = ModelConfig("u200_rtl_fixture", h, p["intermediate_size"],
                                 p["num_experts"], p["top_k"], 8, 8)
        rate = p["gaps_bytes_per_cycle_per_bank" if ddr_gaps else "no_gaps_bytes_per_cycle_per_bank"]
        self.profile = BurstServiceProfile(
            name="u200_rtl_event_alignment_unmeasured", clock_hz=p["clock_hz"],
            num_memory_channels=4, bandwidth_bytes_per_sec_per_channel=rate*p["clock_hz"],
            bandwidth_efficiency=1.0, dma_setup_cycles=p["dma_setup_cycles_per_burst"],
            max_inflight_per_channel=1, burst_bytes=p["max_burst_bytes"],
            supports_channel_striping=True, total_aie_units=1,
            ops_per_cycle_per_aie=p["ops_per_cycle"], aie_efficiency=1.0,
            aie_task_startup_cycles=p["mac_startup_cycles"],
            ocm_bytes=p["ocm_slots"]*p["ocm_slot_bytes"],
            hardware_tag="RTL_TB_SERVICE_ESTIMATE_NOT_BOARD_MEASUREMENT")

    def simulate(self, case: Mapping[str, Any]) -> AlignmentSimulationResult:
        """Run shared stimulus without ingesting any measured RTL timing."""
        p = self.parameters
        truth = tuple(case["true_expert_ids"])
        predicted = int(case["predicted_expert"])
        if len(truth) != self.model.top_k or len(set(truth)) != len(truth) or any(
            e < 0 or e >= self.model.num_experts for e in (*truth, predicted)):
            raise ValidationError("invalid alignment route/candidate")
        h, f = self.model.hidden_size, p["f_tile_size"]
        tile_count = self.model.intermediate_size//f
        logical_bytes = 3*h*f
        queue = EventQueue()
        memory = MultiChannelMemory(self.profile, "striped_tile")
        compute = AIEPool(self.profile)
        ocm = CapacityOnChipMemory(self.profile.ocm_bytes)
        requests: dict[Key, MemoryRequest] = {}
        all_requests: list[MemoryRequest] = []
        pending_pairs: list[tuple[int, int, str]] = []
        allocations: list[tuple[int, int]] = []
        order: list[tuple[int, int]] = []
        next_tile = {e: 0 for e in truth}
        timeline: list[dict[str, Any]] = []
        boundaries: list[dict[str, Any]] = []
        issue_window_open = False
        active: tuple[int, int] | None = None
        resolved = False
        finish = 0
        wait_start: int | None = None
        wait_cycles = 0
        bank_count = p.get("ocm_banks", 1)
        peak_banks = [0]*bank_count
        start, router_done = compute.reserve(AIEPoolTask("router", router_ops(self.model, 1), 1),
                                             p["router_start_cycle"])
        truth_ready = router_done+p["router_control_cycles"]
        timeline.append(dict(resource="MAC", operation="router", start_cycle=start, end_cycle=router_done,
                             expert_id=-1, tile_id=-1, bytes=0, request_type="none"))
        timeline.append(dict(resource="CONTROL", operation="topk_softmax", start_cycle=router_done,
                             end_cycle=truth_ready, expert_id=-1, tile_id=-1, bytes=0, request_type="none"))
        plan = hardware_greedy_v1((HardwareGreedyV1Candidate(predicted, 0, int(case["confidence_q12"]),
                                   (logical_bytes,)*tile_count),),
                                 HardwareGreedyV1Config(1, int(case["budget_bytes"]),
                                                        (int(case["budget_bytes"]),)))
        queue.push(1, EventType.PREDICTOR_READY)
        queue.push(truth_ready, EventType.TOPK_DONE)

        def key(e: int, t: int, matrix: str) -> Key:
            return (0, e, matrix, t)

        def mark(cycle: int, event: str, expert: int = -1, tile: int = -1,
                 matrix: str = "none", size: int = 0, kind: str = "none") -> None:
            boundaries.append(dict(cycle=cycle,event=event,expert_id=expert,tile_id=tile,
                                   matrix=matrix,bytes=size,request_type=kind,value=0))

        def pair_ready(e: int, t: int, cycle: int) -> bool:
            return all(ocm.is_resident(key(e, t, matrix), cycle) for matrix in ("gate_up", "down"))

        def bank_of(k: Key) -> int:
            # Same epoch/layer/expert/tile/matrix XOR as RTL ownership.
            return (p.get("epoch_id", 1)^k[0]^k[1]^k[3]^int(k[2] == "down")) % bank_count

        def bank_occupancy() -> list[int]:
            counts = [0]*bank_count
            for k in ocm.objects:
                counts[bank_of(k)] += 1
            return counts

        def admit(cycle: int) -> None:
            while pending_pairs:
                e, t, kind = pending_pairs[0]
                limit = p["ocm_slots"]-(p["reserved_demand_slots"] if kind == "speculative" else 0)
                occupied = len(ocm.objects)
                if occupied+2 > min(limit, p["descriptor_entries"]):
                    break
                counts = bank_occupancy()
                for matrix in ("gate_up", "down"):
                    counts[bank_of(key(e, t, matrix))] += 1
                if max(counts) > p.get("slots_per_bank", p["ocm_slots"]):
                    break
                # FIFO_DEPTH is per demand/speculative queue; pair reservation needs two entries.
                queued_count = sum(not r.started and not r.canceled and r.request_type == kind for r in all_requests)
                if queued_count+2 > p["dma_fifo_depth"]:
                    break
                pending_pairs.pop(0)
                if kind == "speculative":
                    allocations.append((e, t))
                for matrix, size in (("gate_up", 2*h*f), ("down", h*f)):
                    k = key(e, t, matrix)
                    if not ocm.admit(k, p["ocm_slot_bytes"], kind, 1.0, cycle).admitted:
                        raise RuntimeError("atomic pair precheck inconsistent")
                    request = memory.new_request(layer_id=0, expert_id=e, matrix_type=matrix,
                        tile_ids=(t,), object_keys=(k,), transfer_bytes=size, request_type=kind,
                        issue_cycle=cycle, schedule_order=len(all_requests))
                    requests[k] = request
                    all_requests.append(request)
                    memory.enqueue(request)
                    mark(cycle,"REQUEST_QUEUED",e,t,matrix,size,kind)
                for bank, count in enumerate(counts):
                    peak_banks[bank] = max(peak_banks[bank], count)

        def kick(cycle: int) -> None:
            for request in memory.start_ready(cycle):
                ocm.set_running(request.object_keys[0], True)
                end = int(request.end_cycle)
                queue.push(end, EventType.DMA_DONE, {"request": request})
                mark(cycle,"MEMORY_SERVICE_START",request.expert_id,request.tile_ids[0],
                     request.matrix_type,request.transfer_bytes,request.request_type)
                timeline.append(dict(resource="DDR_STRIPED", operation=request.matrix_type,
                    start_cycle=cycle, end_cycle=end, expert_id=request.expert_id,
                    tile_id=request.tile_ids[0], bytes=request.transfer_bytes, request_type=request.request_type))

        def launch(cycle: int) -> None:
            nonlocal active, wait_start, wait_cycles, issue_window_open
            if not resolved or active is not None or cycle < finish or all(next_tile[e] == tile_count for e in truth):
                return
            if not issue_window_open:
                mark(cycle,"TILE_ISSUE_WINDOW_OPEN")
                issue_window_open = True
            ready = next((e for e in truth if next_tile[e] < tile_count and pair_ready(e, next_tile[e], cycle)), None)
            if ready is None:
                if wait_start is None:
                    wait_start = cycle
                return
            if wait_start is not None:
                wait_cycles += cycle-wait_start
                if cycle > wait_start:
                    timeline.append(dict(resource="EXPERT_ISSUE", operation="memory_wait",
                        start_cycle=wait_start, end_cycle=cycle, expert_id=-1, tile_id=-1,
                        bytes=0, request_type="none"))
                wait_start = None
            active = (ready, next_tile[ready])
            order.append(active)
            gu_start, gu_end = compute.reserve(AIEPoolTask("gate_up", 4*h*f, 1, 0, ready, active[1]),
                                               cycle+p["tile_dispatch_cycles"])
            mark(gu_start-p.get("request_accept_to_gu_cycles",0),"COMPUTE_REQUEST_ACCEPTED",ready,active[1])
            mark(gu_start,"GU_START",ready,active[1])
            issue_window_open = False
            if gu_start > cycle:
                timeline.append(dict(resource="EXPERT_ISSUE", operation="ready_control_wait",
                    start_cycle=cycle, end_cycle=gu_start, expert_id=ready, tile_id=active[1],
                    bytes=0, request_type="none"))
            timeline.append(dict(resource="MAC", operation="gate_up", start_cycle=gu_start, end_cycle=gu_end,
                expert_id=ready, tile_id=active[1], bytes=0, request_type="none"))
            queue.push(gu_end, EventType.AIE_OPERATION_DONE, {"stage": "gate_up"})

        while queue:
            event = queue.pop()
            cycle = event.cycle
            if event.event_type is EventType.PREDICTOR_READY:
                pending_pairs.extend((a.expert_id, a.tile_id, "speculative") for a in plan)
            elif event.event_type is EventType.TOPK_DONE:
                resolved = True
                mark(cycle,"ROUTE_RESOLVED")
                pending_pairs.clear()  # 尚未 admission 的预测不占实际 budget/OCM。
                for k, request in requests.items():
                    if request.request_type != "speculative":
                        continue
                    request.useful = request.expert_id in truth
                    if request.useful:
                        request.committed = True
                        ocm.commit(k)
                    elif not request.started:
                        memory.cancel(request)
                        ocm.release(k, cycle)
                    elif ocm.is_resident(k, cycle):
                        ocm.release(k, cycle)
                queue.push(cycle+p.get("demand_release_delay_cycles", 0), EventType.DMA_REQUEST,
                           {"demand_release": True})
            elif event.event_type is EventType.DMA_REQUEST:
                pending_pairs.extend((e, t, "demand") for e in truth for t in range(tile_count)
                    if key(e, t, "gate_up") not in requests)
            elif event.event_type is EventType.DMA_DONE:
                request = event.payload["request"]
                memory.complete(request, cycle)
                k = request.object_keys[0]
                ocm.set_running(k, False)
                ocm.mark_resident(k, cycle)
                mark(cycle,"PAYLOAD_READY",request.expert_id,request.tile_ids[0],
                     request.matrix_type,request.transfer_bytes,request.request_type)
                if request.useful is False:
                    ocm.release(k, cycle)
            elif event.event_type is EventType.AIE_OPERATION_DONE:
                if active is None:
                    raise RuntimeError("compute completion has no active tile")
                e, t = active
                if event.payload["stage"] == "gate_up":
                    mark(cycle,"GU_DONE",e,t)
                    end = cycle+ceil(f/p["nonlinear_elements_per_cycle"])
                    timeline.append(dict(resource="NONLINEAR", operation="swiglu", start_cycle=cycle,
                        end_cycle=end, expert_id=e, tile_id=t, bytes=0, request_type="none"))
                    queue.push(end, EventType.NONLINEAR_DONE)
                else:
                    mark(cycle,"DOWN_DONE",e,t)
                    mark(cycle,"TILE_RETIRED",e,t)
                    if t == tile_count-1:
                        mark(cycle,"EXPERT_COMBINE_START",e,t)
                        mark(cycle+p["combine_cycles"],"EXPERT_COMBINE_DONE",e,t)
                    for matrix in ("gate_up", "down"):
                        ocm.release(key(e, t, matrix), cycle)
                    next_tile[e] += 1
                    active = None
                    finish = cycle+p["combine_cycles"]
                    queue.push(finish, EventType.SCHEDULER_INVOKE)
                    # 不在 combine 前启动下一个 tile。
                    admit(cycle)
                    kick(cycle)
                    continue
            elif event.event_type is EventType.NONLINEAR_DONE:
                if active is None:
                    raise RuntimeError("nonlinear completion has no active tile")
                e, t = active
                start, end = compute.reserve(AIEPoolTask("down", 2*h*f, 1, 0, e, t), cycle)
                mark(start,"DOWN_START",e,t)
                timeline.append(dict(resource="MAC", operation="down", start_cycle=start, end_cycle=end,
                    expert_id=e, tile_id=t, bytes=0, request_type="none"))
                queue.push(end, EventType.AIE_OPERATION_DONE, {"stage": "down"})
            admit(cycle)
            kick(cycle)
            launch(cycle)
        if any(next_tile[e] != tile_count for e in truth):
            raise RuntimeError("M10 alignment deadlock: unfinished required tiles")
        spec = [r for r in all_requests if r.request_type == "speculative"]
        control_cycles = sum(r["end_cycle"]-r["start_cycle"] for r in timeline
                             if r["operation"] == "ready_control_wait")
        return AlignmentSimulationResult(finish, router_done, truth_ready, wait_cycles,
            sum(r.transfer_bytes for r in all_requests if r.started and r.request_type == "demand"),
            sum(r.transfer_bytes for r in spec if r.started),
            sum(r.transfer_bytes for r in spec if r.started and r.useful is True),
            sum(r.transfer_bytes for r in spec if r.started and r.useful is False),
            sum(r.transfer_bytes for r in spec if r.canceled),
            ocm.peak_ocm_bytes//p["ocm_slot_bytes"], allocations, order, timeline, peak_banks,
            memory_wait_cycles=wait_cycles, ready_control_wait_cycles=control_cycles,
            expert_issue_idle_cycles=wait_cycles+control_cycles,
            boundary_events=sorted(boundaries,key=lambda row:row["cycle"]))
