# OpenVINO / ivpu integration design note

## Scope

This note maps the current host-side middle-layer prototype to likely Intel NPU integration boundaries.

It does **not** assume exact kernel internal file names. References to kernel behavior are conceptual, using terms such as driver BO creation path, doorbell write path, and completion interrupt handling path.

## 1) What is host-side only today

The following abstractions are host-runtime constructs and currently do not call into real kernel or device APIs directly:

- `HeuristicBufferPolicy` (`buffer_policy.cpp`)
  - classifies requests into buffer classes and policy decisions.
- `ResidencyManager` (`residency_manager.cpp`)
  - host-side pool/lease model for reuse and residency bookkeeping.
- `PublishContract` (`publish_contract.cpp`)
  - explicit prepare/publish/doorbell state transitions for safe submission ordering.
- `SubmissionQueueMpsc` (`submission_queue.cpp`)
  - low-lock queue model for producer/consumer submission staging.
- `CompletionPath` (`completion_path.cpp`)
  - fast-path vs deferred completion handling model for TTFT-sensitive flows.
- trace and benchmark harnesses
  - observability and host-only microbench validation.

These pieces should be treated as policy and orchestration surfaces ready for backend wiring.

## 2) What could map to OpenVINO plugin layers

A practical mapping for OpenVINO integration would place most middle-layer entry points in a plugin/runtime adapter layer that sits between graph execution orchestration and low-level NPU command submission:

- **Memory policy + residency**
  - `BufferPolicy` / `ResidencyManager` can map to plugin-managed buffer lifecycle decisions.
  - plugin context can pass tensor role hints (weight/state/activation/cmd) into policy classification.

- **Submission ordering wrapper**
  - `PublishContract` and `SubmissionQueueMpsc` can map to plugin-side request dispatch wrappers that serialize queue-visible publication without global coarse locking.

- **Completion handling**
  - `CompletionPath` can map to plugin/runtime completion callback management, splitting TTFT-critical callback wakeup from deferred housekeeping.

This keeps OpenVINO-visible behavior deterministic while allowing backend-specific implementation behind adapter boundaries.

## 3) Conceptual alignment with ivpu runtime/driver paths

The middle-layer abstractions conceptually align with common ivpu-style responsibilities as follows.

### 3.1 BO creation / bind / map

- `BufferClass`, `ResidencyPolicy`, `MappingPolicy`, and `BufferPolicyDecision`
  - align with **driver BO creation path** decisions (size, usage class, visibility intent).
- `ResidencyManager`
  - aligns with host decisions around reuse vs new allocation before entering **driver BO bind/map path**.
- `GdnStateRegion`
  - aligns with explicit state-region tracking passed alongside BO metadata.

### 3.2 Descriptor publication and doorbell

- `PublishContract`
  - aligns with visibility ordering guarantees before entering **doorbell write path**.
  - explicitly models descriptor + metadata + dynamic state publication readiness.
- `SubmissionQueueMpsc`
  - aligns with low-lock producer staging and queue-order preservation prior to doorbell notification.

### 3.3 Completion and wake path

- `CompletionPath`
  - aligns with host behavior immediately downstream of **completion interrupt handling path**.
  - supports first-token-critical wake/callback path and deferred non-critical work path.

## 4) Likely placement of key GDN runtime responsibilities

### Persistent GDN state residency

Likely to live primarily in the host/plugin layer where request semantics are known:

- classify persistent recurrent state buffers,
- enforce warm-path reuse policy,
- minimize remap/rebind churn before entering driver BO map/bind operations.

Kernel/driver layers then execute concrete BO operations; policy ownership stays in host runtime where model state transitions are visible.

### Submission ordering wrapper

Likely to live in host/plugin dispatch wrapper immediately before backend command submission:

- enforce prepare/publish ordering,
- ensure doorbell-safe state before notifying device queue,
- keep per-queue sequencing explicit for correlation and debugging.

### Completion fast path

Likely split across boundaries:

- lower layer supplies completion signal/record (completion interrupt handling path + completion queue surface),
- host/plugin layer executes TTFT-aware wakeup and callback ordering (`CompletionPath` fast path),
- deferred cleanup/statistics stays in non-critical worker path.

## 5) Integration sequence (recommended)

1. Replace simulated BO identifiers with real backend buffer handles in policy/residency interfaces.
2. Wire `PublishContract` transitions to real publication primitives and doorbell write invocation.
3. Bind `CompletionPath` input to real completion records and wake sources.
4. Reuse trace event IDs (`sequence_id`, `ticket_id`, `queue_id`, `buffer_id`) for cross-layer correlation with runtime/driver telemetry.
5. Promote microbench metrics into integration benchmarks that include real OpenVINO request execution and ivpu-facing completion timing.

## PR-worthy claims

1. Reduced mapping churn on warm stateful paths through explicit persistent-state residency policy and reuse management.
2. Safer publication contract for command/metadata/dynamic-state visibility before doorbell writes.
3. Lower host-side lock contention in submission staging via explicit low-lock queueing and per-queue sequencing.
4. Reduced host-side TTFT tail risk by separating first-token-critical completion handling from deferred bookkeeping.
5. Improved observability for upstream evaluation through stable trace identifiers across memory, submission, and completion planes.
