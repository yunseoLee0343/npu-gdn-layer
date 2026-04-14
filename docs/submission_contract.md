# Submission ordering contract (first pass)

## Purpose

The publish contract layer defines a single legal path from host writes to NPU doorbell notification:

1. `prepare_submission(job)`
2. `publish_submission(job)`
3. `ring_doorbell(queue)`
4. `await_completion(ticket)`

The goal is to replace implicit barrier folklore with explicit publication states and transitions.

## Why “works on x86” is not enough

Relying on x86 behavior is insufficient for runtime contracts because:

- host/device interaction is not just CPU-core ordering; it includes cache visibility to device-side consumers,
- MMIO doorbell publication can race ahead of descriptor/metadata/state visibility if contract edges are vague,
- portability and future compiler/runtime changes can invalidate assumptions hidden in ad hoc code paths.

A correct middle-layer must model publication intent explicitly instead of inheriting incidental ordering behavior.

## Why this matters for Intel NPU / OpenVINO / ivpu style interaction

For Intel NPU/OpenVINO/ivpu style flows, descriptor BOs, metadata BOs, and dynamic GDN state regions are consumed by device execution queues after doorbell signaling.

If any of these are stale when the doorbell is rung, queue execution can observe inconsistent state. This leads to nondeterministic failures that look like sporadic timeouts or wrong-sequence retirements.

The contract therefore requires that descriptor + metadata + dynamic state all reach device-visible state before a submission is marked doorbell-safe.

## Acquire/release style publication vs full barrier everywhere

The contract models a two-step publication strategy:

- **prepare stage**: validate host writes complete, allocate sequence/ticket.
- **publish stage**: transition host-write-complete -> device-visible -> doorbell-safe using release-style publication semantics.

This is different from issuing heavyweight full barriers around every step:

- release-style edges are targeted to publication points,
- doorbell safety is explicit and auditable per ticket,
- completion waits can use sequence checks without globally serializing all queues.

## How this reduces lock contention and avoids over-serialization

The wrapper avoids global “big lock + big barrier” behavior by:

- assigning per-queue monotonic sequences at prepare time,
- allowing independent submissions to be prepared without immediate doorbell operations,
- limiting strict ordering checks to the queue and ticket involved,
- enabling lock scope to remain small while preserving publication correctness.

## Failure modes this contract is designed to prevent

- stale descriptor visibility
- stale metadata visibility
- sporadic completion timeout
- wrong-sequence completion
