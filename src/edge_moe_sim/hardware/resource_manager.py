"""Central owner of all resources for a single simulation run."""

from __future__ import annotations

from ..schema import HardwareConfig
from .aie import AIEEngine
from .controller import PLController
from .dma import DMAEngine
from .nonlinear import NonlinearEngine
from .ocm import OnChipMemory


class ResourceManager:
    """Construct and expose isolated hardware resource instances."""

    def __init__(self, config: HardwareConfig) -> None:
        self.aie = AIEEngine(config.aie)
        self.dma = DMAEngine(config.dma)
        self.nonlinear = NonlinearEngine(config.nonlinear)
        self.ocm = OnChipMemory()
        self.controller = PLController(config.pl_control)
