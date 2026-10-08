"""Common resource reservation interface."""

from __future__ import annotations

from typing import Any, Protocol


class Resource(Protocol):
    """Protocol implemented by mutually exclusive simulated resources."""

    def earliest_start(self, task: Any, current_cycle: int) -> int:
        """Return the first cycle at which ``task`` can start."""
        ...

    def reserve(self, task: Any, start_cycle: int) -> int:
        """Reserve the resource and return the exclusive finish cycle."""
        ...

