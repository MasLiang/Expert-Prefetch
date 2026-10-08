# M7.1b Metric Semantics

## 1. Why cleanup was required

Old `planned_speculative_bytes` actually reflected admitted/started requests in
some characterization paths, and `actual_speculative_bytes_before_router`
counted completed whole requests only. Neither represented initial allocation
or partial physical progress unambiguously.

## 2–4. Request lifecycle and Router-time traffic

The lifecycle is allocated → queued → cancelled-before-issue, or issued →
active → completed. Initial allocation consumes policy budget. Cancellation
does not consume bandwidth. Issued counts a whole request once. At Router,
completed request bytes and partial payload bytes of active requests are
separate; their sum is `physical_speculative_bytes_transferred_by_router`.

## 5–7. Usefulness, readiness, progress, and gap

Router truth determines useful physical bytes. Executable
`startup_readiness` counts completed resident tiles only.
`startup_transfer_progress` includes partial payload and is clamped to [0,1].
`startup_gap_cycles` is event-derived as Expert start minus Router resolve; it
therefore includes residual transfer, queueing, and resource effects.

## 8. Critical Expert

The critical completion Expert is the true Expert whose finish equals the
layer completion. Its readiness, physical progress, gap, and finish are
reported independently because averages need not predict makespan.

## 9. Blocking

`aggregate_demand_request_blocking_cycles` sums unique blocked intervals across
demand requests. `critical_path_demand_blocking_cycles` is its critical-Expert
subset. Neither is a makespan penalty or the default optimization objective.

## 10. OCM normalization

`demand_working_set_reference_bytes` is measured per workload from the smallest
no-prediction capacity within 1% of the large-OCM latency. Headroom is
`max(0, capacity-reference)` and is a static normalization, not instantaneous
free OCM.

## 11. Future DSE metrics

Minimize inference makespan. Use initial allocation, physical/wasted traffic,
and OCM as cost/Pareto axes. Use critical/weighted startup gap and continuous
progress as timing explanations. Recall, StartableRecall, and aggregate
blocking remain auxiliary.

## 12. Deprecated aliases

- `planned_speculative_bytes` → `initial_allocated_speculative_bytes`
- `actual_speculative_bytes_before_router` → `completed_speculative_bytes_before_router`
- `speculative_bytes` → use issued/completed/physical metric appropriate to context
- `useful_speculative_bytes` → use useful-completed or useful-physical metric
- `demand_blocked_by_future_speculation_cycles` → `aggregate_demand_request_blocking_cycles`

