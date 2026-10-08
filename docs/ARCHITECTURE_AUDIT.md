# Architecture audit status

The historical audit found three separate engines: legacy `Simulator`, M5 `SequenceSimulator`, and M6 `AdvancedSimulator`. That finding remains part of repository history.

Milestone 7 adds a fourth explicit path, `UnifiedSequenceSimulator`. It does not redirect or remove the older engines. The new path combines M5-style source-milestone/future-target causality with shared M6 `MultiChannelMemory`, `AIEPool`, `CapacityOnChipMemory`, and `FTileGroup` execution on one global event queue. See `UNIFIED_SEQUENCE_SIMULATOR.md` for current semantics and limitations.
