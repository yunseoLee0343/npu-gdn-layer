# Completion latency model (TTFT-critical path)

## Why completion is not just a generic done-event

For GDN stateful inference, completion is part of user-visible token latency, not just an internal queue retire signal.

The runtime must model completion as a staged path:

1. IRQ observed
2. completion record available
3. waiting runtime thread wakeable
4. first-token critical callback
5. deferred non-critical bookkeeping

This split allows the first-token critical path to remain short while moving non-critical work out of the wake path.

## Why mean latency is insufficient

Mean latency hides tail behavior that dominates perceived responsiveness in stateful decoding.

A system can have acceptable average completion times while still violating first-token response expectations under bursty queue occupancy or wakeup contention.

Therefore completion design should optimize and monitor latency distribution, not only averages.

## Why p95/p99 TTFT matters more

For interactive inference, slow outliers determine quality of service.

- p50 TTFT indicates typical behavior.
- p95 TTFT indicates whether tail behavior is under control.
- p99 TTFT captures worst-case jitter likely caused by contention, queue backlog, or delayed wakeups.

Runtime decisions (fast-path callbacks, deferred bookkeeping, queue policy) should be evaluated against p95/p99 TTFT movement.

## Separate IRQ-to-wakeup and wakeup-to-first-token

The completion path should explicitly separate:

- **IRQ-to-completion-record** latency: interrupt handling + record publication cost.
- **completion-record-to-wakeup** latency: scheduler and waiter wake path cost.
- **wakeup-to-first-token** latency: runtime callback and token emission path cost.

Without this split, regressions get misattributed and the wrong subsystem gets optimized.

## Why completion path design is runtime architecture, not driver trivia

Driver IRQ handling is only one segment of end-to-end TTFT.

The runtime controls:

- wakeup policy,
- callback dispatch shape,
- deferral of non-critical accounting/cleanup,
- queue-to-completion correlation policy.

These decisions directly change p95/p99 TTFT and must be treated as first-class architecture choices.

## Metric set

- IRQ-to-completion-record latency
- completion-record-to-wakeup latency
- wakeup-to-first-token latency
- p50/p95/p99 TTFT
