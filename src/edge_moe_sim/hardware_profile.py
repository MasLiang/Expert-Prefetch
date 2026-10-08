"""Parameterized board-level memory, compute, and storage profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from .schema import HardwareConfig, ValidationError


@dataclass(frozen=True)
class HardwareProfile:
    """Overrideable hardware capabilities used by Milestone 6 models."""

    name: str
    clock_hz: int
    num_memory_channels: int
    bandwidth_bytes_per_sec_per_channel: float
    bandwidth_efficiency: float
    dma_setup_cycles: int
    max_inflight_per_channel: int
    burst_bytes: int
    supports_channel_striping: bool
    total_aie_units: int
    ops_per_cycle_per_aie: float
    aie_efficiency: float
    aie_task_startup_cycles: int
    ocm_bytes: int
    router_ops_per_cycle_per_aie: float | None = None
    down_ops_per_cycle_per_aie: float | None = None
    physical_aie_tiles: int | None = None
    physical_aie_tiles_per_compute_group: int = 1
    hardware_tag: str = "synthetic"

    def __post_init__(self) -> None:
        positive = {
            "clock_hz": self.clock_hz,
            "num_memory_channels": self.num_memory_channels,
            "bandwidth_bytes_per_sec_per_channel": self.bandwidth_bytes_per_sec_per_channel,
            "max_inflight_per_channel": self.max_inflight_per_channel,
            "burst_bytes": self.burst_bytes,
            "total_aie_units": self.total_aie_units,
            "ops_per_cycle_per_aie": self.ops_per_cycle_per_aie,
            "ocm_bytes": self.ocm_bytes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValidationError(f"hardware_profile.{name} must be positive")
        if not 0.0 < self.bandwidth_efficiency <= 1.0:
            raise ValidationError("bandwidth_efficiency must be in (0, 1]")
        if not 0.0 < self.aie_efficiency <= 1.0:
            raise ValidationError("aie_efficiency must be in (0, 1]")
        if self.dma_setup_cycles < 0 or self.aie_task_startup_cycles < 0:
            raise ValidationError("hardware setup cycles must be non-negative")
        if self.physical_aie_tiles is not None:
            if self.physical_aie_tiles <= 0 or self.physical_aie_tiles_per_compute_group <= 0:
                raise ValidationError("physical AIE tile quantities must be positive")
            if self.total_aie_units * self.physical_aie_tiles_per_compute_group > self.physical_aie_tiles:
                raise ValidationError("logical compute groups exceed physical AIE tile mapping")

    @property
    def theoretical_memory_bandwidth_bytes_per_sec(self) -> float:
        """Return channel peak sum without an efficiency claim."""
        return self.num_memory_channels * self.bandwidth_bytes_per_sec_per_channel

    @property
    def effective_memory_bandwidth_bytes_per_sec(self) -> float:
        """Return configured effective aggregate bandwidth."""
        return self.theoretical_memory_bandwidth_bytes_per_sec * self.bandwidth_efficiency

    @property
    def effective_bytes_per_cycle_per_channel(self) -> float:
        """Return effective payload bandwidth of one channel in bytes/cycle."""
        return (
            self.bandwidth_bytes_per_sec_per_channel
            * self.bandwidth_efficiency / self.clock_hz
        )

    @property
    def theoretical_compute_peak_ops_per_sec(self) -> float:
        """Return unadjusted configured AIE peak."""
        return self.total_aie_units * self.ops_per_cycle_per_aie * self.clock_hz

    @property
    def effective_compute_peak_ops_per_sec(self) -> float:
        """Return configured board compute ceiling after efficiency."""
        return self.theoretical_compute_peak_ops_per_sec * self.aie_efficiency

    def compute_rate_per_cycle(self, operation_type: str = "gate_up") -> float:
        """Return the per-AIE effective rate for an operation class."""
        raw = self.ops_per_cycle_per_aie
        if operation_type == "router" and self.router_ops_per_cycle_per_aie is not None:
            raw = self.router_ops_per_cycle_per_aie
        elif operation_type == "down" and self.down_ops_per_cycle_per_aie is not None:
            raw = self.down_ops_per_cycle_per_aie
        return raw * self.aie_efficiency

    def dma_cycles(self, transfer_bytes: int, channel_count: int = 1) -> int:
        """Return setup plus payload cycles for a non-preemptive burst."""
        if transfer_bytes < 0 or channel_count <= 0:
            raise ValidationError("invalid DMA transfer size or channel count")
        rate = self.effective_bytes_per_cycle_per_channel * channel_count
        return self.dma_setup_cycles + ceil(transfer_bytes / rate)

    def metadata(self) -> dict[str, Any]:
        """Return profile fields plus explicit theoretical/effective ceilings."""
        return {
            **asdict(self),
            "theoretical_memory_BW": self.theoretical_memory_bandwidth_bytes_per_sec,
            "effective_memory_BW": self.effective_memory_bandwidth_bytes_per_sec,
            "theoretical_compute_peak": self.theoretical_compute_peak_ops_per_sec,
        }

    @classmethod
    def from_legacy(cls, config: HardwareConfig, *, ocm_bytes: int = 2**63 - 1) -> "HardwareProfile":
        """Map old single-DMA/single-AIE configs without changing their rates."""
        return cls(
            "legacy_single_channel_single_aie", config.clock_hz,
            1, config.dma.bytes_per_cycle * config.clock_hz, 1.0,
            config.dma.startup_cycles, config.dma.max_inflight,
            2**31, False, 1, config.aie.gate_up_ops_per_cycle, 1.0,
            config.aie.startup_cycles, ocm_bytes,
            config.aie.router_ops_per_cycle, config.aie.down_ops_per_cycle,
        )


def vek280_like_theoretical_profile() -> HardwareProfile:
    """Return an optional synthetic VEK280-like theoretical study profile.

    The 102.4 GB/s aggregate is a theoretical input assumption, not achieved or
    measured bandwidth. Compute rate and efficiencies are deliberately explicit.
    """
    return HardwareProfile(
        name="vek280_like_theoretical_unmeasured",
        clock_hz=1_000_000_000,
        num_memory_channels=3,
        bandwidth_bytes_per_sec_per_channel=102.4e9 / 3,
        bandwidth_efficiency=0.70,
        dma_setup_cycles=200,
        max_inflight_per_channel=1,
        burst_bytes=4096,
        supports_channel_striping=True,
        total_aie_units=400,
        ops_per_cycle_per_aie=8.0,
        aie_efficiency=0.60,
        aie_task_startup_cycles=50,
        ocm_bytes=32 * 1024 * 1024,
    )
