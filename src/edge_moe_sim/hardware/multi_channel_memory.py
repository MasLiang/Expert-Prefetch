"""Independent non-preemptive memory channels with deterministic placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Hashable, Literal

from ..hardware_profile import HardwareProfile
from ..schema import ValidationError
from ..execution_rules import ExecutionRules


class MemoryPlacementPolicy(str, Enum):
    """Supported expert/tile placement policies."""

    SINGLE_CHANNEL = "single_channel"
    ROUND_ROBIN_EXPERT = "round_robin_expert"
    STRIPED_TILE = "striped_tile"


MemoryRequestType = Literal["demand", "background", "speculative"]


@dataclass
class MemoryRequest:
    """One queued physical burst, optionally containing adjacent logical tiles."""

    request_id: int
    layer_id: int
    expert_id: int
    matrix_type: str
    tile_ids: tuple[int, ...]
    object_keys: tuple[Hashable, ...]
    transfer_bytes: int
    request_type: MemoryRequestType
    issue_cycle: int
    score: float = 0.0
    candidate_rank: int | None = None
    speculative_scope: str = ""
    useful: bool | None = None
    committed: bool = False
    canceled: bool = False
    started: bool = False
    start_cycle: int | None = None
    end_cycle: int | None = None
    channel_ids: tuple[int, ...] = ()
    schedule_order: int | None = None
    transferred_bytes: int = 0
    burst_bytes: int = 0
    burst_end_cycle: int | None = None
    burst_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    completed: bool = False
    promotion_token_created: bool = False
    active_queue_class: str = ''
    selection_needed: bool = True
    enqueue_ready_cycle: int = 0
    payload_cycle_multiplier: int = 1

    @property
    def priority_class(self) -> int:
        """Return current demand, background, future speculative priority."""
        if self.request_type == "demand" or self.committed:
            return 0
        if self.request_type == "background":
            return 1
        return 2


@dataclass(frozen=True)
class MemoryInterval:
    """One channel's participation in a half-open burst interval."""

    request_id: int
    start_cycle: int
    end_cycle: int
    bytes_on_channel: int
    request_type: str


@dataclass
class MemoryChannel:
    """Per-channel queue, active request, intervals, and byte counters."""

    channel_id: int
    active_request: MemoryRequest | None = None
    intervals: list[MemoryInterval] = field(default_factory=list)
    transferred_bytes: int = 0
    demand_bytes: int = 0
    speculative_bytes: int = 0
    useful_speculative_bytes: int = 0
    wasted_speculative_bytes: int = 0

    @property
    def busy_cycles(self) -> int:
        """Return the sum of occupied channel cycles."""
        return sum(interval.end_cycle - interval.start_cycle for interval in self.intervals)

    def idle_cycles(self, makespan_cycles: int) -> int:
        """Return channel-idle cycles inside a simulation makespan."""
        return max(0, makespan_cycles - self.busy_cycles)


class MultiChannelMemory:
    """Deterministic multi-channel DMA server with shared priority semantics."""

    def __init__(
        self, profile: HardwareProfile,
        placement: MemoryPlacementPolicy | str = MemoryPlacementPolicy.ROUND_ROBIN_EXPERT,
        *, pair_fifo: bool = False,
        rules: ExecutionRules | None = None,
    ) -> None:
        self.profile = profile
        self.pair_fifo = pair_fifo
        self.rules = rules or ExecutionRules()
        self.managed = bool(self.rules.descriptor_entries or self.rules.dma_fifo_depth or self.rules.burst_arbitration
                            or self.rules.dma_request_latency_cycles or self.rules.dma_request_interval_cycles)
        self.tokens: dict[str, list[MemoryRequest]] = {kind: [] for kind in ('demand', 'promoted', 'speculative')}
        self.live_descriptors: dict[int, MemoryRequest] = {}
        self.continuations: list[MemoryRequest] = []
        self.descriptor_peak = 0
        self.fifo_peaks: dict[str, int] = {kind: 0 for kind in self.tokens}
        self.fifo_occupancy_peaks: dict[str, int] = {kind: 0 for kind in self.tokens}
        self.fifo_reserved_credit_peaks: dict[str, int] = {kind: 0 for kind in self.tokens}
        self.fifo_unenqueued_credit_peaks: dict[str, int] = {kind: 0 for kind in self.tokens}
        self.fifo_dequeue_pending: dict[str, list[tuple[MemoryRequest, int]]] = {kind: [] for kind in self.tokens}
        self.fifo_snapshots: list[dict[str, object]] = []
        self.ingress_available_cycle = 0
        self.ingress_intervals: list[tuple[int, int]] = []
        self.tombstones_retired = 0
        self.admission_backpressure_count = 0
        self.current_cycle = 0
        self.retirement_deadlines: dict[int, int] = {}
        self.final_retirement_cycle = 0
        self.next_enqueue_cycle = 0
        self.placement = MemoryPlacementPolicy(placement)
        if profile.max_inflight_per_channel != 1:
            raise ValidationError(
                "Milestone 6 multi-channel model currently supports max_inflight_per_channel=1"
            )
        if self.placement is MemoryPlacementPolicy.STRIPED_TILE and not profile.supports_channel_striping:
            raise ValidationError("striped_tile requires supports_channel_striping")
        self.channels = [MemoryChannel(index) for index in range(profile.num_memory_channels)]
        self.pending: list[MemoryRequest] = []
        self.completed: list[MemoryRequest] = []
        self.canceled_bytes = 0
        self._next_request_id = 0

    def reserve_ingress_pair(self, current_cycle: int) -> int:
        """预留共享 GU/DOWN 发射控制器，返回 idle 可见的整数 cycle。"""
        if current_cycle < self.ingress_available_cycle:
            raise ValidationError('ingress manager 仍忙，不能接受另一 pair')
        self.ingress_available_cycle = current_cycle + self.rules.ingress_pair_cycles
        if self.rules.ingress_pair_cycles:
            self.ingress_intervals.append((current_cycle, self.ingress_available_cycle))
        return self.ingress_available_cycle

    def sample_fifo_metrics(self, current_cycle: int) -> dict[str, dict[str, int]]:
        """区分已保留的token credit与已到入队可见拍的实际occupancy，单位entry。"""
        for kind, entries in self.fifo_dequeue_pending.items():
            self.fifo_dequeue_pending[kind] = [(r, end) for r, end in entries if end > current_cycle]
        held = {kind: sum(end > current_cycle for _, end in entries)
                for kind, entries in self.fifo_dequeue_pending.items()}
        occupancy = {kind: sum(r.enqueue_ready_cycle <= current_cycle for r in queue)
                     + held[kind] for kind, queue in self.tokens.items()}
        return_reserved = int(any(channel.active_request is not None
            and channel.active_request.active_queue_class == 'speculative' for channel in self.channels))
        reserved = {kind: len(queue) + held[kind] for kind, queue in self.tokens.items()}
        reserved['speculative'] += max(0, return_reserved - held['speculative'])
        pending = {kind: reserved[kind] - occupancy[kind] for kind in self.tokens}
        for kind in self.tokens:
            self.fifo_reserved_credit_peaks[kind] = max(self.fifo_reserved_credit_peaks[kind], reserved[kind])
            self.fifo_occupancy_peaks[kind] = max(self.fifo_occupancy_peaks[kind], occupancy[kind])
            self.fifo_unenqueued_credit_peaks[kind] = max(self.fifo_unenqueued_credit_peaks[kind], pending[kind])
        snapshot = dict(cycle=current_cycle, reserved_credits=reserved, occupancy=occupancy,
                        reserved_not_enqueued=pending)
        if not self.fifo_snapshots or any(self.fifo_snapshots[-1][key] != snapshot[key]
                                         for key in ('reserved_credits', 'occupancy')):
            self.fifo_snapshots.append(snapshot)
        return dict(reserved_credits=reserved, occupancy=occupancy, reserved_not_enqueued=pending)

    def _observe_dequeue(self, kind: str, request: MemoryRequest) -> None:
        """push优先的物理FIFO在连续入队结束后的首个quiet cycle才能pop。"""
        if not self.rules.fifo_push_blocks_dequeue:
            return
        pushes = {r.enqueue_ready_cycle for queue in self.tokens.values() for r in queue}
        pushes.add(request.enqueue_ready_cycle)
        visible = self.current_cycle
        while visible in pushes:
            visible += 1
        if visible > self.current_cycle:
            self.fifo_dequeue_pending[kind].append((request, visible))
        self.sample_fifo_metrics(self.current_cycle)

    def new_request(
        self, *, layer_id: int, expert_id: int, matrix_type: str,
        tile_ids: tuple[int, ...], object_keys: tuple[Hashable, ...],
        transfer_bytes: int, request_type: MemoryRequestType, issue_cycle: int,
        score: float = 0.0, candidate_rank: int | None = None,
        speculative_scope: str = "",
        schedule_order: int | None = None,
    ) -> MemoryRequest:
        """Construct and place one request without starting it."""
        if transfer_bytes <= 0 or not tile_ids or len(tile_ids) != len(object_keys):
            raise ValidationError("memory request needs positive bytes and matching tile keys")
        request = MemoryRequest(
            self._next_request_id, layer_id, expert_id, matrix_type, tile_ids,
            object_keys, transfer_bytes, request_type, issue_cycle, score,
            candidate_rank, speculative_scope,
        )
        request.schedule_order = schedule_order
        self._next_request_id += 1
        request.channel_ids = self._placement_channels(request)
        return request

    def _placement_channels(self, request: MemoryRequest) -> tuple[int, ...]:
        if self.placement is MemoryPlacementPolicy.SINGLE_CHANNEL:
            return (0,)
        if self.placement is MemoryPlacementPolicy.ROUND_ROBIN_EXPERT:
            return (request.expert_id % len(self.channels),)
        return tuple(range(len(self.channels)))

    def enqueue(self, request: MemoryRequest, *, coalesce_adjacent: bool = False) -> MemoryRequest:
        """Queue a request, optionally merging an adjacent compatible burst."""
        if self.managed and coalesce_adjacent:
            raise ValidationError('有限 credits/burst 模式不支持隐式 coalescing')
        if coalesce_adjacent:
            for existing in reversed(self.pending):
                if (
                    self._can_coalesce(existing, request)
                    and existing.transfer_bytes + request.transfer_bytes <= self.profile.burst_bytes
                ):
                    existing.tile_ids += request.tile_ids
                    existing.object_keys += request.object_keys
                    existing.transfer_bytes += request.transfer_bytes
                    return existing
        self.current_cycle = max(self.current_cycle, request.issue_cycle)
        self.pending.append(request)
        request.enqueue_ready_cycle = max(request.issue_cycle + self.rules.dma_request_latency_cycles,
                                          self.next_enqueue_cycle)
        self.next_enqueue_cycle = request.enqueue_ready_cycle + self.rules.dma_request_interval_cycles
        if self.managed:
            if not self.can_admit(request.request_type, 1):
                self.pending.pop()
                raise ValidationError('请求未预留 descriptor/FIFO credit')
            self.live_descriptors[request.request_id] = request
            self.descriptor_peak = max(self.descriptor_peak, len(self.live_descriptors))
            self._push_token(request.request_type, request)
        return request

    def _push_token(self, kind: str, request: MemoryRequest) -> None:
        """保留 FIFO 引用；promotion 不销毁原 speculative token。"""
        queue = self.tokens['demand' if kind == 'background' else kind]
        queue.append(request)
        self.fifo_peaks['demand' if kind == 'background' else kind] = max(
            self.fifo_peaks['demand' if kind == 'background' else kind], len(queue))
        self.sample_fifo_metrics(self.current_cycle)

    def can_admit(self, kind: str, count: int = 2, *, current_cycle: int | None = None) -> bool:
        """预检完整 pair 的 table/FIFO credit，包括 active spec 回队保留位置。"""
        if current_cycle is not None:
            self.current_cycle = current_cycle
        if not self.managed:
            return True
        demand_reserve = self.rules.reserved_demand_descriptors if kind == 'speculative' else 0
        table_ok = not self.rules.descriptor_entries or len(self.live_descriptors) + count + demand_reserve <= self.rules.descriptor_entries
        queue_kind = 'demand' if kind == 'background' else kind
        held = sum(end > self.current_cycle for _, end in self.fifo_dequeue_pending[queue_kind])
        reserve = int(queue_kind == 'speculative' and any(
            channel.active_request is not None and channel.active_request.active_queue_class == 'speculative'
            for channel in self.channels))
        fifo_ok = not self.rules.dma_fifo_depth or len(self.tokens[queue_kind]) + count + max(held, reserve) <= self.rules.dma_fifo_depth
        return table_ok and fifo_ok

    def promote(self, request: MemoryRequest, current_cycle: int | None = None) -> bool:
        """queued hit 需要 promoted token credit；active/completed hit 无须重复排队。"""
        if current_cycle is not None:
            self.current_cycle = current_cycle
        if request.committed:
            return True
        if self.managed and not request.completed and (not request.started or request in self.tokens['speculative']):
            held = sum(end > self.current_cycle for _, end in self.fifo_dequeue_pending['promoted'])
            if self.rules.dma_fifo_depth and len(self.tokens['promoted']) + held >= self.rules.dma_fifo_depth:
                return False
            self._push_token('promoted', request)
            request.promotion_token_created = True
        request.committed = True
        request.useful = True
        if request.started:
            request.active_queue_class = 'promoted'
        self.retire_descriptors()
        return True

    def retire_descriptors(self, current_cycle: int | None = None) -> None:
        """payload 完成且 truth 已知、所有 FIFO 引用退休后才回收 descriptor。"""
        if current_cycle is not None:
            self.current_cycle = current_cycle
        referenced = {r.request_id for queue in self.tokens.values() for r in queue}
        referenced.update(r.request_id for entries in self.fifo_dequeue_pending.values()
                          for r, end in entries if end > self.current_cycle)
        for request_id, request in tuple(self.live_descriptors.items()):
            resolved = request.request_type != 'speculative' or request.useful is not None
            if (request.completed or request.canceled) and resolved and request_id not in referenced:
                deadline = self.retirement_deadlines.setdefault(request_id,
                    self.current_cycle + self.rules.descriptor_retirement_cycles)
                if deadline <= self.current_cycle:
                    del self.live_descriptors[request_id]
                    del self.retirement_deadlines[request_id]
                    self.final_retirement_cycle = self.current_cycle

    def _select_managed(self) -> MemoryRequest | None:
        """按三类 FIFO 选择；stale/canceled token 只在其队首被退休。"""
        if all(channel.active_request is not None for channel in self.channels):
            return None
        for request in tuple(self.continuations):
            if all(self.channels[c].active_request is None for c in request.channel_ids):
                self.continuations.remove(request)
                request.selection_needed = False
                return request
        for kind in ('demand', 'promoted', 'speculative'):
            queue = self.tokens[kind]
            while queue:
                request = queue[0]
                if request.enqueue_ready_cycle > self.current_cycle:
                    break
                stale = request.canceled or request.completed or (
                    kind == 'speculative' and request.promotion_token_created)
                if stale:
                    queue.pop(0)
                    self._observe_dequeue(kind, request)
                    self.tombstones_retired += 1
                    self.retire_descriptors()
                    continue
                if all(self.channels[c].active_request is None for c in request.channel_ids):
                    request.active_queue_class = kind
                    request.selection_needed = True
                    queue.pop(0)
                    self._observe_dequeue(kind, request)
                    return request
                break
        return None

    @staticmethod
    def _can_coalesce(left: MemoryRequest, right: MemoryRequest) -> bool:
        return (
            not left.started and not left.canceled
            and left.layer_id == right.layer_id
            and left.expert_id == right.expert_id
            and left.matrix_type == right.matrix_type
            and left.request_type == right.request_type
            and left.speculative_scope == right.speculative_scope
            and left.channel_ids == right.channel_ids
            and left.tile_ids[-1] + 1 == right.tile_ids[0]
        )

    def cancel(self, request: MemoryRequest) -> bool:
        """Cancel a queued request; active bursts remain non-preemptive."""
        if request.started or request.canceled:
            return False
        request.canceled = True
        self.canceled_bytes += request.transfer_bytes
        return True

    def start_ready(self, current_cycle: int) -> list[MemoryRequest]:
        """Start every currently feasible burst under channel capacity."""
        self.current_cycle = current_cycle
        self.sample_fifo_metrics(current_cycle)
        self.retire_descriptors()
        started: list[MemoryRequest] = []
        while True:
            eligible = [
                request for request in self.pending
                if not request.started and not request.canceled
                and all(self.channels[channel].active_request is None for channel in request.channel_ids)
            ]
            if not eligible and not self.managed:
                break
            request = self._select_managed() if self.managed else min(eligible, key=lambda item: (
                (0 if item.request_type == 'demand' else 1 if item.committed else 2) if self.pair_fifo else item.priority_class,
                0 if self.pair_fifo else item.schedule_order if item.schedule_order is not None else 2**31,
                item.request_id if self.pair_fifo else item.issue_cycle if item.priority_class < 2 else -item.score,
                item.candidate_rank if item.candidate_rank is not None else 2**31,
                item.layer_id, item.expert_id,
                0 if item.matrix_type == "gate_up" else 1,
                item.tile_ids,
                item.request_id,
            ))
            if request is None:
                break
            first = not request.started
            request.started = True
            if first:
                request.start_cycle = current_cycle
            size = min(self.profile.burst_bytes, request.transfer_bytes - request.transferred_bytes) if self.rules.burst_arbitration else request.transfer_bytes
            setup = self.profile.dma_setup_cycles + self.rules.dma_selection_cycles if first or request.selection_needed else self.rules.dma_burst_gap_cycles
            # payload rate 来自配置；禁止在 setup/handoff 期间累计 payload。
            payload_cycles = self.profile.dma_cycles(size, len(request.channel_ids)) - self.profile.dma_setup_cycles
            request.payload_cycle_multiplier = self.rules.dma_payload_cycle_multiplier
            payload_cycles *= request.payload_cycle_multiplier
            payload_end = current_cycle + setup + payload_cycles
            end = payload_end + self.rules.dma_return_cycles
            request.burst_bytes = size
            request.burst_end_cycle = end
            # history 的 end 是 payload 窗口末端，不含 completion/ready 返回尾部。
            request.burst_history.append((current_cycle, payload_end, size, current_cycle + setup))
            request.end_cycle = end if request.transferred_bytes + size == request.transfer_bytes else None
            shares = self._byte_shares(size, len(request.channel_ids))
            for channel_id, byte_share in zip(request.channel_ids, shares):
                channel = self.channels[channel_id]
                channel.active_request = request
                channel.intervals.append(MemoryInterval(
                    request.request_id, current_cycle, end,
                    byte_share, request.request_type,
                ))
            started.append(request)
        self.sample_fifo_metrics(current_cycle)
        return started

    @staticmethod
    def _byte_shares(total_bytes: int, count: int) -> list[int]:
        base, remainder = divmod(total_bytes, count)
        return [base + (index < remainder) for index in range(count)]

    def complete(self, request: MemoryRequest, current_cycle: int) -> bool:
        """Complete an active burst and update every participating channel."""
        self.current_cycle = current_cycle
        if request.burst_end_cycle != current_cycle:
            raise ValueError("memory completion cycle does not match reservation")
        shares = self._byte_shares(request.burst_bytes, len(request.channel_ids))
        for channel_id, byte_share in zip(request.channel_ids, shares):
            channel = self.channels[channel_id]
            if channel.active_request is not request:
                raise ValueError("memory channel active request mismatch")
            channel.active_request = None
            channel.transferred_bytes += byte_share
            if request.request_type in {"demand", "background"}:
                channel.demand_bytes += byte_share
            else:
                channel.speculative_bytes += byte_share
                if request.useful is True:
                    channel.useful_speculative_bytes += byte_share
                elif request.useful is False:
                    channel.wasted_speculative_bytes += byte_share
        request.transferred_bytes += request.burst_bytes
        if request.transferred_bytes < request.transfer_bytes:
            if request.committed:
                request.active_queue_class = 'promoted'
            # reserved credit 不是 FIFO 中已经可见的 token。物理入队前不能
            # 用尚未到达的 demand/promoted 迫使 speculative 在 burst 边界让出。
            higher = any(r.enqueue_ready_cycle <= current_cycle
                         for kind in ('demand', 'promoted') for r in self.tokens[kind])
            if request.active_queue_class == 'speculative' and higher:
                self._push_token('speculative', request)
            else:
                self.continuations.append(request)
            return False
        request.completed = True
        self.completed.append(request)
        self.retire_descriptors()
        return True

    def refresh_speculative_accounting(self) -> None:
        """Recompute final useful/wasted per-channel counters after resolve."""
        for channel in self.channels:
            channel.useful_speculative_bytes = 0
            channel.wasted_speculative_bytes = 0
        for request in self.completed:
            if request.request_type != "speculative":
                continue
            shares = self._byte_shares(request.transfer_bytes, len(request.channel_ids))
            for channel_id, byte_share in zip(request.channel_ids, shares):
                if request.useful is True:
                    self.channels[channel_id].useful_speculative_bytes += byte_share
                elif request.useful is False:
                    self.channels[channel_id].wasted_speculative_bytes += byte_share

    def metrics(self, makespan_cycles: int, *, include_fifo_timeline: bool=False) -> dict[str, object]:
        """Return physical/useful/demand/speculative utilization and channel bytes."""
        if makespan_cycles <= 0:
            raise ValidationError("memory metrics require a positive makespan")
        self.refresh_speculative_accounting()
        total_capacity_cycles = makespan_cycles * len(self.channels)
        busy = sum(channel.busy_cycles for channel in self.channels)
        transferred = sum(channel.transferred_bytes for channel in self.channels)
        demand = sum(channel.demand_bytes for channel in self.channels)
        speculative = sum(channel.speculative_bytes for channel in self.channels)
        useful_spec = sum(channel.useful_speculative_bytes for channel in self.channels)
        wasted = sum(channel.wasted_speculative_bytes for channel in self.channels)
        useful = demand + useful_spec
        issued_speculative = sum(
            request.transfer_bytes for request in self.pending
            if request.request_type == "speculative"
        )
        max_payload = (
            self.profile.effective_bytes_per_cycle_per_channel * total_capacity_cycles
        )
        seconds = makespan_cycles / self.profile.clock_hz
        return {
            "memory_busy_cycles": busy,
            "memory_idle_cycles": max(0, total_capacity_cycles - busy),
            "physical_bandwidth_utilization": busy / total_capacity_cycles,
            "useful_bandwidth_utilization": useful / max_payload,
            "demand_bandwidth_utilization": demand / max_payload,
            "speculative_bandwidth_utilization": speculative / max_payload,
            "wasted_bandwidth_fraction": wasted / transferred if transferred else 0.0,
            "useful_byte_fraction": useful / transferred if transferred else 0.0,
            "achieved_useful_bandwidth_bytes_per_sec": useful / seconds,
            "bytes_per_channel": tuple(channel.transferred_bytes for channel in self.channels),
            "per_channel_utilization": tuple(
                channel.busy_cycles / makespan_cycles for channel in self.channels
            ),
            "idle_cycles_per_channel": tuple(
                channel.idle_cycles(makespan_cycles) for channel in self.channels
            ),
            "transferred_bytes": transferred,
            "demand_bytes": demand,
            "speculative_bytes": speculative,
            "issued_speculative_bytes": issued_speculative,
            "canceled_speculative_bytes": self.canceled_bytes,
            "useful_speculative_bytes": useful_spec,
            "wasted_speculative_bytes": wasted,
            "descriptor_peak": self.descriptor_peak,
            "fifo_peaks": dict(self.fifo_peaks),
            "fifo_reserved_credit_peaks": dict(self.fifo_reserved_credit_peaks),
            "fifo_occupancy_peaks": dict(self.fifo_occupancy_peaks),
            "fifo_reserved_not_enqueued_peaks": dict(self.fifo_unenqueued_credit_peaks),
            "fifo_metric_semantics": {"fifo_peaks": "legacy_logical_token_reference_peak",
                "occupancy": "enqueue_visible_tokens_only", "unit": "entry"},
            **({"fifo_timeline": tuple(self.fifo_snapshots)} if include_fifo_timeline else {}),
            "descriptor_live_at_end": len(self.live_descriptors),
            "tombstone_tokens_retired": self.tombstones_retired,
            "credit_backpressure_count": self.admission_backpressure_count,
        }
