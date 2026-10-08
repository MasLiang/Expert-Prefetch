# Unified Sequence Simulator

`UnifiedSequenceSimulator` is the explicit Milestone 7 path. It combines M5's source-layer milestone and future-target authority semantics with M6's batch routing, F-dimension tile groups, multi-channel memory, finite OCM, and capacity-based AIE pool.

All layers use one `EventQueue`, `MultiChannelMemory`, `AIEPool`, `CapacityOnChipMemory`, and nonlinear engine. A prediction specification names source and target layers. Its ready cycle is derived as `source milestone cycle + predictor latency`; a direct override is test-only and is tagged synthetic.

`UnifiedLayerSequence.inter_layer_config` reuses M5's `InterLayerConfig` and can reserve synthetic AIE-pool cycles, nonlinear cycles, controller cycles, and background DMA bytes between adjacent MoE layers. These operations use the same global resources as MoE execution. The legacy `inter_layer_cycles` mapping remains as a concise controller-delay convenience. Both forms create timing opportunity but are explicitly synthetic—not a real attention implementation or calibrated AIE/DMA composition.

At prediction ready, target GU prefix or full-Expert tile requests enter the same memory channels as source-layer demand. Target Router resolution commits true requests, cancels queued false requests, lets active false bursts finish non-preemptively, and issues demand only for missing true tiles. Tile identity is `(layer, expert, matrix, tile)` and required weights are released after their dependent GEMM. There is no persistent cache.

An Expert becomes executable after Router resolution when its startup GU tile is resident. Each F group follows GU weight → GU GEMM → nonlinear → Down weight → Down GEMM. Top-K Experts may overlap subject to `max_parallel_experts` and total AIE groups. Repeated batch assignments load one layer-local Expert weight copy while operations scale with `n_e`.

Results separate required inference completion from resource drain and report realized lead time, prefetch opportunity, startup readiness, useful/wasted/cancelled speculation, demand blocking, OCM peak, channel utilization, and AIE utilization. Optional traces include source/target/layer/expert/tile/resource metadata.

## Regression relationship

With one layer and prediction disabled, regression tests match `AdvancedSimulator` for Router resolve, first Expert start, required demand bytes, token counts, and makespan. Unified finite-OCM demand admission intentionally keeps a critical current-tile window and retries admission after consumption; the older advanced path may eagerly reserve more future demand tiles, so peak OCM can differ without changing execution timing. With one channel/group the source-trigger, future request, target authority, cancellation, and active-burst non-preemption ordering matches M5 causality, while cycle equality is not expected because M7 uses F tiles and M6 resources.

This remains an abstract performance simulator. It does not model real attention, KV cache, prefill, continuous batching, persistent Expert caching, cross-token decode timing, predictor hardware execution, NoC topology, LPDDR banks/pages, measured AIE scaling, power, or energy. Hardware profiles are synthetic/theoretical unless explicitly calibrated.
