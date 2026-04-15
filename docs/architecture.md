# Architecture

## 1. Host runtime side

The host runtime owns model/session lifecycle and token scheduling policy. For this project, host runtime responsibilities are:

- Session creation/teardown and model graph ownership.
- Token-step orchestration (prefill, decode iterations, stop conditions).
- Invocation of middle-layer APIs for state transition boundaries.

The host runtime should not directly manage low-level descriptor queues or BO migration policy beyond declared intents.

## 2. Middle-layer role

The middle-layer is the control surface that translates host intents into NPU-executable work with deterministic ordering and memory behavior.

Core responsibilities:

- **State residency manager**
  - Classifies BO objects by role (weights, KV cache, temp, command BO).
  - Tracks mapping/pinning and sg_table shape constraints.
  - Executes residency transitions at GDN boundaries without avoidable copy paths.

- **Submission manager**
  - Encodes descriptor sequences for command streams.
  - Assigns sequence IDs and enforces monotonic queue order.
  - Performs doorbell writes only after descriptor visibility/synchronization conditions are satisfied.

- **Completion manager**
  - Handles IRQ notifications and optional poll fallback.
  - Correlates completion records to sequence ID and token-step.
  - Exposes timing points for TTFT and tail latency (p95/p99).

## 3. OpenVINO/ivpu boundary

Boundary definition:

- **Above boundary (host + middle-layer):**
  - Execution planning, policy selection, descriptor preparation, runtime tracing.

- **At boundary:**
  - Submission of prepared descriptors/command buffers to OpenVINO/ivpu-facing interfaces.
  - BO metadata and synchronization primitives passed to driver/runtime.

- **Below boundary (OpenVINO runtime + ivpu/driver path):**
  - Hardware scheduling, MMU/memory mapping realization, IRQ generation.

The middle-layer should preserve observability at boundary crossings: enqueue timestamp, doorbell timestamp, completion IRQ timestamp, and retired timestamp.

## 4. Placement of key GDN responsibilities

### 4.1 GDN state residency

Belongs in the **middle-layer memory plane**:

- Maintains residency tables keyed by session and GDN state node.
- Decides BO placement transitions (retain/migrate/evict) before each token-step.
- Applies zero-copy rules where BO compatibility allows direct reuse.

### 4.2 Submission ordering

Belongs in the **middle-layer submission plane**:

- Uses lock-free producer structures where feasible.
- Maintains strict descriptor sequence monotonicity per queue.
- Guards against out-of-order doorbell rings and duplicate submissions.

### 4.3 Completion fast path

Belongs in the **middle-layer completion plane**:

- Prioritizes IRQ path for minimal wake latency.
- Keeps poll path as bounded fallback for diagnostics/degraded modes.
- Emits per-step latency buckets for TTFT and p95/p99 regression tracking.

## 5. Initial module map (planned)

- `include/npu_gdn/memory_policy.h`
- `include/npu_gdn/submission_queue.h`
- `include/npu_gdn/completion_path.h`
- `src/memory/`
- `src/submission/`
- `src/completion/`

This map is a bootstrap target; exact interfaces will evolve with milestone validation.
