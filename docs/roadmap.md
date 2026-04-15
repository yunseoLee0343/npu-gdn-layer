# Roadmap

## Stage 0 — Bootstrap

Objective: establish repository and design baseline for implementation.

Deliverables:

- Project skeleton (`docs/`, `include/`, `src/`, `tests/`, `benchmarks/`, `scripts/`).
- Architecture boundary document.
- Milestone and acceptance criteria draft.

Exit criteria:

- Repository is ready for parallel implementation tracks.
- Terminology and metrics (TTFT, p95/p99, descriptor, BO, IRQ) are fixed in docs.

## Stage 1 — Memory policy prototype

Objective: validate BO and sg_table policy transitions across GDN state changes.

Deliverables:

- BO class taxonomy and policy table.
- Residency state machine prototype.
- Tests covering pin/unpin, migrate/retain, and zero-copy eligibility checks.

Exit criteria:

- Deterministic residency transitions for prefill/decode loops.
- Documented behavior for memory pressure and remap faults.

## Stage 2 — Submission wrapper

Objective: provide correctness-first descriptor submission API with lock-free ordering.

Deliverables:

- Submission wrapper interface and queue implementation.
- Descriptor validation and sequence invariant checks.
- Doorbell write protocol with memory ordering guarantees.

Exit criteria:

- No ordering violations under concurrency stress tests.
- Traceable sequence lineage from enqueue to retire.

## Stage 3 — Completion tracing

Objective: implement completion fast path instrumentation and latency attribution.

Deliverables:

- IRQ-path completion handler with fallback polling mode.
- Trace schema for enqueue, doorbell, IRQ, retire timestamps.
- Latency analysis scripts for TTFT and p95/p99.

Exit criteria:

- Stable per-token completion attribution.
- Regression-ready latency artifacts in benchmark pipeline.

## Stage 4 — Benchmark + PR packaging

Objective: publish reproducible benchmark baselines and integration-ready change set.

Deliverables:

- Benchmark scenarios (steady decode, burst decode, mixed-session contention).
- Baseline report with TTFT and p95/p99 by workload profile.
- PR package containing design notes, test evidence, benchmark artifacts, and host environment details.

Exit criteria:

- Reproducible benchmark instructions and artifacts committed.
- Review-ready PR with scoped risk and follow-up tasks.

## Stage 5 — OSS prototype hardening

Objective: make the repository presentable and reproducible for external reviewers.

Deliverables:

- Explicit build/test/benchmark instructions in README.
- Terminology glossary aligned with code and docs.
- Current limitations and non-claims section.
- Integration-target section mapping host prototype to OpenVINO/ivpu paths.

Exit criteria:

- New contributors can build tests and benchmarks using documented commands.
- PRs can reference stable benchmark output fields and tail-latency metrics.
- Public-facing docs avoid oversell and clearly separate simulated vs hardware-backed results.

## Stage 6 — OpenVINO/ivpu backend integration (target)

Objective: wire host abstractions to real backend surfaces without breaking measurement contracts.

Deliverables:

- Real buffer handle plumbing for BO creation/bind/map paths.
- Real doorbell write path wiring behind publish contract APIs.
- Completion interrupt handling path integration with runtime wake/callback flow.
- Cross-layer trace correlation plan with queue/ticket/sequence IDs.

Exit criteria:

- End-to-end path exercises real backend submissions.
- Host-side benchmark schema remains usable and comparable pre/post integration.
- PR evidence includes correctness, latency, and observability artifacts.

## Next PR series seed

- `series/openvino-ivpu-adapter-and-sequence-contract-hardening`: unify sequence ownership, add backend adapter interfaces, and wire OpenVINO/ivpu integration layers.
