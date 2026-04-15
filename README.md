# npu-gdn-layer

## Problem statement

Qwen3-Next GDN stateful execution paths need deterministic state residency and low-overhead command progression on Intel NPU stacks. Existing host-side integrations often treat model state transitions as opaque framework events, which causes:

- BO placement and migration decisions that are disconnected from token-step lifecycle.
- Submission paths that serialize on coarse locks instead of enforcing correctness through descriptor ordering.
- Completion handling that optimizes average latency but not TTFT and p95/p99 tail behavior.

This repository defines a systems runtime middle-layer that sits between host runtime logic and Intel NPU driver/runtime boundaries (OpenVINO + ivpu-facing flows), with explicit control of memory policy, submission ordering, and completion latency.

## Scope

The project covers middle-layer components required to execute GDN stateful workloads with implementation-level constraints:

- BO lifecycle management for weights, activations, KV/state buffers, and staging memory.
- sg_table-aware placement and pinning policy selection for steady-state decode and transition phases.
- Descriptor construction, sequencing, and doorbell submission for lock-free ordered dispatch.
- IRQ/completion path hooks for low-latency wakeup, accounting, and tracing.
- Runtime instrumentation focused on TTFT and p95/p99 completion latency.
- Test and benchmark harnesses for submission correctness and memory residency behavior.

## Non-goals

The project intentionally does **not** include:

- Model training, finetuning, or optimizer kernels.
- End-user serving APIs, UI, or deployment orchestration.
- Generic multi-vendor abstraction layers that hide Intel-specific memory/runtime behavior.
- Full OpenVINO frontend replacement.
- Marketing-grade performance claims without reproducible benchmark artifacts.

## Architecture overview

`npu-gdn-layer` is structured as a runtime middle-layer with three execution-critical planes:

1. **Memory plane**
   - Tracks BO ownership, mapping mode, and sg_table state.
   - Applies policy transitions at GDN state boundaries (prefill/decode/state-evict).

2. **Submission plane**
   - Builds and validates descriptor chains.
   - Enforces lock-free submission ordering using per-queue monotonic sequence IDs.
   - Rings doorbell with minimal host-side contention.

3. **Completion plane**
   - Handles IRQ-driven and polled completion signals.
   - Correlates descriptor completion with token-step milestones.
   - Exposes TTFT and p95/p99 latency telemetry.

See `docs/architecture.md` for boundary details and placement of each responsibility.

## Engineering tracks

### 1) Memory Policy

Focus:

- BO class definitions and residency contracts.
- sg_table fragmentation handling and pin/unpin policies.
- Zero-copy state transitions for GDN persistent context.

Primary outputs:

- Policy engine spec.
- Residency transition tests.
- Failure-mode matrix (eviction, remap, sync faults).

### 2) Submission Correctness

Focus:

- Descriptor ABI and validation.
- Lock-free queueing and sequence monotonicity.
- Doorbell ordering rules and replay/idempotency guards.

Primary outputs:

- Submission wrapper API.
- Concurrency stress tests.
- Ordering proof artifacts and invariant checks.

### 3) Completion Latency

Focus:

- IRQ fast path and fallback poll path.
- Completion attribution to token-level state transitions.
- TTFT and p95/p99 regression gating.

Primary outputs:

- Tracing schema and collectors.
- Latency dashboards/artifacts.
- Benchmark thresholds and CI checks.

## Repository layout

- `docs/` — architecture notes, milestones, integration notes, benchmark plans.
- `include/` — exported headers for middle-layer interfaces.
- `src/` — implementation of memory, submission, completion, and tracing paths.
- `tests/` — correctness and concurrency tests.
- `benchmarks/` — host-side microbenchmark harnesses.
- `scripts/` — developer and CI helper scripts.

## Build instructions

Current project state is a C++17 prototype without CMake/Bazel packaging yet.
Build individual test/benchmark targets directly with `g++`:

```bash
# unit tests
g++ -std=c++17 -pthread -Iinclude tests/test_residency_manager.cpp src/residency_manager.cpp src/trace.cpp -o /tmp/test_residency_manager
g++ -std=c++17 -pthread -Iinclude tests/test_submission_queue.cpp src/submission_queue.cpp src/publish_contract.cpp src/trace.cpp -o /tmp/test_submission_queue

# benchmarks
g++ -std=c++17 -Iinclude benchmarks/bench_memory_policy.cpp src/residency_manager.cpp src/trace.cpp -o /tmp/bench_memory_policy
g++ -std=c++17 -pthread -Iinclude benchmarks/bench_submission_queue.cpp src/submission_queue.cpp src/publish_contract.cpp src/trace.cpp -o /tmp/bench_submission_queue
g++ -std=c++17 -pthread -Iinclude benchmarks/bench_completion_path.cpp src/completion_path.cpp src/trace.cpp -o /tmp/bench_completion_path
```

## Test instructions

Run unit tests locally after build:

```bash
/tmp/test_residency_manager
/tmp/test_submission_queue
```

Expected behavior:

- process exits with code `0`.
- no assertion failures.
- stderr may include trace logs from default logger sink.

## Benchmark instructions

Run host-side harnesses locally after build:

```bash
/tmp/bench_memory_policy
/tmp/bench_submission_queue
/tmp/bench_completion_path
```

Use output fields as baseline artifacts in PRs (p50/p95/p99 and hit-rate fields). Keep hardware/system metadata with each run.

## Terminology

- **BO**: Buffer object managed by runtime/driver integration path.
- **sg_table (SGT)**: Scatter-gather layout metadata for BO backing segments.
- **Descriptor**: Device-consumed command/control structure for submission.
- **Doorbell**: Queue notification write signaling runnable work.
- **IRQ**: Completion interrupt signal from backend/driver path.
- **TTFT**: Time-to-first-token for user-visible first output token.
- **p95/p99**: Tail latency percentiles used for regression gating.
- **Fast path**: First-token-critical completion handling path.
- **Deferred path**: Non-critical completion bookkeeping/cleanup path.

## Current limitations

- Host-side simulation only; no real OpenVINO execution or ivpu driver wiring yet.
- Submission/doorbell/completion timings are prototype-level and not device-level performance claims.
- Build is command-line compilation, not yet packaged into a unified OSS build system.
- Queue/residency semantics are first-pass models; ABI and backend contracts may evolve.
- Trace backend defaults to stderr logger; no production telemetry exporter adapter included yet.

## Next integration targets

1. Wire buffer handles and residency operations to real OpenVINO plugin + driver BO creation/bind/map paths.
2. Replace simulated publish transitions with real doorbell write path integration.
3. Bind completion path to real completion interrupt handling path and runtime wake mechanisms.
4. Add cross-layer correlation using trace IDs (`queue_id`, `sequence_id`, `ticket_id`, `buffer_id`).
5. Introduce reproducible CI benchmark jobs with artifact capture and tail-latency regression thresholds.

## Current status

Prototype ready for backend integration planning. See `docs/roadmap.md` for staged milestones and PR-oriented integration targets.
