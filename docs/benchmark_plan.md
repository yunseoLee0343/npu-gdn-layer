# Benchmark plan (host-side prototype)

## Scope and honesty statement

These benchmarks measure host-side middle-layer behavior only. They do **not** claim device-level Intel NPU performance.

Current results are readiness indicators for later OpenVINO/ivpu integration, not final production latency numbers.

## Benchmarks

- `bench_memory_policy.cpp`
  - measures pool reuse behavior and acquire latency distribution.
- `bench_submission_queue.cpp`
  - measures enqueue/publish overhead and contention behavior under multi-producer load.
- `bench_completion_path.cpp`
  - measures fast-path vs deferred completion callback behavior and simulated TTFT-critical path timing.

## Required outputs and where they come from

- **buffer pool hit rate**
  - from residency lease generation/reuse tracking in memory policy benchmark.
- **average and tail enqueue latency**
  - from enqueue timing samples (p50/p95/p99 + average if needed).
- **average and tail publish latency**
  - from publish timing samples in submission queue benchmark.
- **average and tail completion callback latency**
  - from wake + callback timing in completion benchmark.
- **simulated TTFT-related critical path timings**
  - from completion fast-path event timing (host-only simulation).

## Numbers we can measure now

- host-side buffer acquire/release timing distributions
- reuse hit/miss behavior from residency manager pooling model
- host-side enqueue contention under MPSC queue load
- publish wrapper overhead and doorbell batching overhead (simulated ring path)
- completion fast-path callback vs deferred callback timing deltas
- simulated TTFT critical path percentiles in host runtime path

## Numbers that require real Intel NPU integration later

- true hardware doorbell-to-execution latency
- true IRQ delivery and kernel wake latency under ivpu driver load
- BO bind/map/unmap latency under real sg_table and memory pressure conditions
- device-side scheduling/retire latency and its interaction with queue depth
- end-to-end TTFT including actual model execution on Intel NPU
- cross-layer correlation with ftrace/eBPF + driver tracepoints for verified bottleneck attribution

## Integration readiness usage

Use this benchmark harness to:

1. detect regressions in host middle-layer overhead before driver integration,
2. validate that queue/memory/completion abstractions stay measurable,
3. carry forward the same metric schema when OpenVINO/ivpu integration is added.
