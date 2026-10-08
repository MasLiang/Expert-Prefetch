# M7.1b Metric Cleanup Report

## Executive answers

Old `planned_speculative_bytes` was not a stable policy-allocation metric; it
could shrink after admission/cancellation. The canonical fixed-budget quantity
is now `initial_allocated_speculative_bytes`, recorded before resource effects.

Allocated bytes consume policy budget. Cancelled-before-issue bytes consume no
physical bandwidth. Issued counts whole requests entering service; completed
counts finished requests. At Router, partial in-flight payload is separate.
Completed plus partial equals physical-by-Router without double counting.

Old `actual_speculative_bytes_before_router` meant completed whole-request
bytes. Physical bytes can therefore be nonzero while that metric is zero.
Likewise executable readiness can be zero while transfer progress is nonzero;
the known deterministic case reduces startup gap from 360 to 106 cycles.

The critical Expert startup gap is Router resolve to the start of the Expert
whose completion determines layer completion. Aggregate blocking must not be a
DSE objective; sequence makespan remains primary.

Canonical paper metrics are allocation, cancelled/issued/completed/partial
physical traffic, useful/wasted physical traffic, continuous startup progress,
event-derived startup gaps, critical Expert gap, and canonical blocking names.
Deprecated aliases are listed in `docs/M71B_METRIC_SEMANTICS.md` and the metric
definitions CSV.

The narrow M2/M4 diagnosis confirms useful completed bytes are not monotonic
with latency because predictor rank, partial progress, request interference,
and the realized critical Expert—not aggregate useful completed bytes—determine
the completion path.

No scheduler, DMA priority, Router authority, OCM policy, or F-tile behavior
was intentionally changed.
