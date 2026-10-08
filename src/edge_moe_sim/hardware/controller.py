"""PL controller latency model."""

from __future__ import annotations

from ..schema import PLControlConfig


class PLController:
    """Container for configured deterministic controller latencies."""

    def __init__(self, config: PLControlConfig) -> None:
        self.config = config

    def topk_dispatch_done(self, router_done_cycle: int) -> int:
        """Return completion after serial Top-K and dispatch control."""
        return router_done_cycle + self.config.topk_cycles + self.config.dispatch_cycles

    def combine_done(self, experts_done_cycle: int) -> int:
        """Return completion after final output combination."""
        return experts_done_cycle + self.config.combine_cycles

