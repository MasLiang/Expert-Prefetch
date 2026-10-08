"""可配置的物理分配与执行规则，不绑定具体 FPGA。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schema import ValidationError


@dataclass(frozen=True)
class ExecutionRules:
    """零 slot 大小保留历史 byte OCM；所有大小为 byte，epoch 无量纲。"""

    ocm_slot_bytes: int = 0
    ocm_banks: int = 1
    reserved_demand_slots: int = 0
    epoch_base: int = 1
    atomic_tile_pair: bool = False
    compute_policy: Literal['expert_locked', 'tile_ready_first'] = 'expert_locked'
    retry_future_prefetch: bool = False
    request_order: Literal['legacy', 'pair_fifo'] = 'legacy'
    descriptor_entries: int = 0
    dma_fifo_depth: int = 0
    burst_arbitration: bool = False
    dma_selection_cycles: int = 0
    dma_burst_gap_cycles: int = 0
    router_load_cycles: int = 0
    router_control_cycles: int = 0
    tile_dispatch_cycles: int = 0
    nonlinear_handoff_cycles: int = 0
    layer_combine_cycles: int = 0
    layer_retirement_cycles: int = 0
    demand_dispatch_cycles: int = 0
    route_scan_cycles_per_entry: int = 0
    descriptor_retirement_cycles: int = 0
    dma_request_latency_cycles: int = 0
    dma_request_interval_cycles: int = 0
    dma_payload_cycle_multiplier: int = 1
    dma_return_cycles: int = 0
    router_truth_emit_cycles: int = 0
    prefetch_allocator_cycles: int = 0
    block_prefetch_during_layer_retirement: bool = False
    route_resolver_drain_cycles: int = 0
    route_commit_visibility_cycles: int = 0
    committed_launch_cycles: int = 0
    truth_scheduler_refresh_cycles: int = 0
    expert_combine_cycles: int = 0
    ocm_payload_visibility_cycles: int = 0
    ingress_pair_cycles: int = 0
    layer_retirement_probe_cycles: int = 0
    layer_retirement_wait_for_ingress: bool = False
    fifo_push_blocks_dequeue: bool = False
    reserved_demand_descriptors: int = 0
    wait_full_expert: bool = False
    # 每次 speculative pair admission 后的 allocator 启动间隔（cycle）；
    # demand admission 游标可选独占 ingress，直到 admission 完成而非 DMA 完成。
    prefetch_pair_interval_cycles: int = 0
    demand_ingress_blocks_prefetch: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.wait_full_expert, bool):
            raise ValidationError('execution_rules.wait_full_expert 必须为 bool')
        for name in ('ocm_slot_bytes', 'ocm_banks', 'reserved_demand_slots', 'epoch_base',
                     'descriptor_entries', 'reserved_demand_descriptors', 'dma_fifo_depth', 'dma_selection_cycles', 'dma_burst_gap_cycles',
                     'router_load_cycles', 'router_control_cycles', 'tile_dispatch_cycles',
                     'nonlinear_handoff_cycles', 'layer_combine_cycles', 'layer_retirement_cycles',
                     'demand_dispatch_cycles', 'route_scan_cycles_per_entry', 'descriptor_retirement_cycles',
                     'dma_request_latency_cycles', 'dma_request_interval_cycles',
                     'dma_payload_cycle_multiplier', 'dma_return_cycles', 'router_truth_emit_cycles',
                     'prefetch_allocator_cycles', 'route_resolver_drain_cycles',
                     'prefetch_pair_interval_cycles',
                     'route_commit_visibility_cycles', 'committed_launch_cycles',
                     'truth_scheduler_refresh_cycles', 'expert_combine_cycles',
                     'ocm_payload_visibility_cycles', 'ingress_pair_cycles',
                     'layer_retirement_probe_cycles'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValidationError(f'execution_rules.{name} 必须为整数')
            if value < 0:
                raise ValidationError(f'execution_rules.{name} 必须非负')
        if self.dma_payload_cycle_multiplier < 1:
            raise ValidationError('dma_payload_cycle_multiplier 必须为正整数')
        if self.ocm_slot_bytes < 0 or self.ocm_banks <= 0 or self.ocm_banks & (self.ocm_banks - 1):
            raise ValidationError('OCM slot bytes 非负，bank 数须为正的 2 次幂')
        if self.reserved_demand_slots < 0 or self.epoch_base < 0:
            raise ValidationError('demand slots 和 epoch_base 必须非负')
        if self.compute_policy not in ('expert_locked', 'tile_ready_first') or self.request_order not in ('legacy', 'pair_fifo'):
            raise ValidationError('未知 compute_policy/request_order')
        if self.reserved_demand_slots and not self.ocm_slot_bytes:
            raise ValidationError('demand slot 保留需要 fixed-slot OCM')
        if self.atomic_tile_pair and not self.ocm_slot_bytes:
            raise ValidationError('atomic_tile_pair 当前需要 fixed-slot OCM')
        if self.compute_policy == 'tile_ready_first' and not self.atomic_tile_pair:
            raise ValidationError('tile_ready_first 当前要求成对预留 DOWN credit')
        if (self.descriptor_entries or self.dma_fifo_depth) and not self.atomic_tile_pair:
            raise ValidationError('有限 credits 当前要求 atomic_tile_pair')
        if self.descriptor_entries == 1 or self.dma_fifo_depth == 1:
            raise ValidationError('pair admission 至少需要两个 descriptor/FIFO credits')
        if self.route_commit_visibility_cycles and not self.descriptor_entries:
            raise ValidationError('逐 descriptor resolver 时序需要有限 descriptor_entries')
        if self.reserved_demand_descriptors and (not self.descriptor_entries or
                self.reserved_demand_descriptors > self.descriptor_entries):
            raise ValidationError('demand descriptor 保留不能超过有限 table 容量')
        if self.ingress_pair_cycles and not self.atomic_tile_pair:
            raise ValidationError('ingress pair 服务需要 atomic_tile_pair')
        if self.layer_retirement_wait_for_ingress and self.layer_retirement_probe_cycles > self.layer_retirement_cycles:
            raise ValidationError('retirement probe 不能超过完整 retirement 控制链')
        if type(self.demand_ingress_blocks_prefetch) is not bool:
            raise ValidationError('demand_ingress_blocks_prefetch 必须为 bool')
        if (self.prefetch_pair_interval_cycles or self.demand_ingress_blocks_prefetch) and not self.atomic_tile_pair:
            raise ValidationError('allocator cadence/ingress arbitration 需要 atomic_tile_pair')
        if (self.prefetch_pair_interval_cycles or self.demand_ingress_blocks_prefetch) and not self.retry_future_prefetch:
            raise ValidationError('allocator cadence/ingress arbitration 需要 future pair 重试')
