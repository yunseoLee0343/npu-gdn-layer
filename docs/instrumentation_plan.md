# Instrumentation plan

## Goals

Add low-overhead, backend-agnostic trace hooks across allocation, submission, and completion paths so runtime behavior can be measured without hardwiring a specific tracing stack.

Implemented interface:

- `TraceSink` abstraction (`Emit(event)`)
- default logger sink (`DefaultLoggerSink`)
- global `SetTraceSink` / `GetTraceSink`
- `TraceEvent` with first-class timestamp, sequence id, ticket id, queue id, token index

## Event catalog and meaning

- `kBufferClassification`
  - emitted when a buffer/tensor request is classified into BO class/policy.
- `kBufferAcquire`
  - emitted when a pool/residency lease is acquired.
- `kBufferRelease`
  - emitted when a lease is released back to pool/residency manager.
- `kResidencyReuseHit`
  - emitted when an acquire reuses an existing compatible object.
- `kResidencyReuseMiss`
  - emitted when acquire must create/provision a new object.
- `kSubmissionPrepared`
  - emitted after `prepare_submission` assigns sequence/ticket.
- `kSubmissionPublished`
  - emitted when job transitions to device-visible / doorbell-safe stage.
- `kDoorbellRung`
  - emitted when queue doorbell is notified for runnable work.
- `kCompletionObserved`
  - emitted when completion record is published and wake path is notified.
- `kFirstTokenCallbackEntered`
  - emitted at fast-path first-token callback entry.
- `kDeferredCompletionWorkEntered`
  - emitted when deferred completion bookkeeping begins.

## Metrics derivable from events

- classification volume by BO class and request size (`kBufferClassification`)
- acquire/release throughput and pool pressure (`kBufferAcquire`, `kBufferRelease`)
- residency reuse hit-rate (`kResidencyReuseHit / (hit+miss)`)
- submission pipeline latency:
  - prepared -> published
  - published -> doorbell
- completion path latency decomposition:
  - doorbell -> completion observed
  - completion observed -> first-token callback
  - completion observed -> deferred bookkeeping start
- queue-level sequencing checks:
  - monotonic sequence/ticket progression per queue

## Mapping to future real-system tracing (ftrace/eBPF/kernel correlation)

These hooks are designed as host-runtime anchors that can later be correlated with kernel/device traces:

- host `kDoorbellRung` timestamp can be correlated with kernel MMIO/doorbell tracepoint.
- host `kCompletionObserved` can be correlated with IRQ handler and completion queue traces.
- host buffer reuse events can be correlated with BO bind/map/unmap kernel telemetry.
- sequence/ticket fields provide shared keys for joining host-side events with ftrace/eBPF streams.

Planned integration model:

1. keep current backend-neutral `TraceSink` API in runtime code,
2. add adapters that forward `TraceEvent` into:
   - user-space ring logger,
   - perf/ftrace user event bridge,
   - eBPF map/event channel,
3. align IDs (`queue_id`, `sequence_id`, `ticket_id`, `buffer_id`) with kernel-visible metadata for deterministic cross-layer analysis.
