# Memory policy subsystem (first pass)

## Purpose

The memory policy layer turns model/runtime intent into BO-class, mapping, and bind behavior that is actionable for Intel NPU OpenVINO/ivpu execution.

It is responsible for classifying runtime requests into:

- persistent weights
- persistent recurrent state
- transient activations
- low-latency command/descriptor buffers

and producing a policy decision that includes:

- preferred residency
- host visibility
- device visibility
- expected reuse
- mutability
- low-latency control-plane flag

## Why this is not just kmalloc/vmalloc theory

`kmalloc`/`vmalloc` vocabulary explains allocator internals, but it does not encode NPU runtime behavior across bind/unbind cycles, queue publication safety, or descriptor ordering.

For this project, policy quality is determined by whether we can:

- keep long-lived BOs resident when decode loops are hot,
- avoid repeated map/unmap churn for recurrent GDN state,
- keep command/descriptor buffers publication-safe for doorbell paths,
- and control latency impact from remap and SGT fragmentation.

Allocator words alone do not capture those constraints.

## Mapping to BO class / mapping / bind-unbind behavior

### Buffer class mapping

- `WEIGHT_PERSISTENT_RO`
  - residency: persistent device-local preference
  - mapping: device-only by default, optional host read-mostly for observability
  - bind/unbind: bind once per session where possible

- `STATE_PERSISTENT_RW`
  - residency: persistent RW, typically coherent when host inspection is enabled
  - mapping: coherent RW when host-visible state updates are required
  - bind/unbind: long-lived bind with explicit state-region validation

- `ACTIVATION_EPHEMERAL_RW`
  - residency: stream/pool/recycle
  - mapping: device-only by default
  - bind/unbind: frequent reuse from pool; avoid per-step allocation churn

- `CMD_DESC_LOWLAT`
  - residency: no-evict / low-latency preferred
  - mapping: host write-combined when descriptor preparation is host-side
  - bind/unbind: publication-safe handling before doorbell writes

## Why SGT length, mapping churn, and reuse matter

For NPU middle-layer behavior, practical performance and correctness are dominated by:

1. **SGT segment count**
   - Higher segment counts increase translation and mapping overhead.
   - Long SGT chains can inflate bind/map costs and tail latency.

2. **Mapping churn**
   - Repeated map/unmap of recurrent state creates avoidable latency and cache/TLB disturbance.
   - Churn also increases synchronization points around host/device visibility.

3. **Reuse behavior**
   - Persistent reuse for weights/state improves residency hit rate and stabilizes TTFT/p95/p99.
   - Activation pooling reduces allocation pressure and submission-side stalls.

These are runtime-facing controls; they matter more than abstract allocator labels.

## First-pass heuristic used in code

- long-lived read-only weights -> persistent residency (`WEIGHT_PERSISTENT_RO`)
- recurrent GDN state -> persistent RW-oriented mapping (`STATE_PERSISTENT_RW`)
- ephemeral activations -> stream/pool/recycle (`ACTIVATION_EPHEMERAL_RW`)
- command/descriptor buffers -> low-latency publication-safe handling (`CMD_DESC_LOWLAT`)

## What to measure

- bind latency
- map latency
- SGT segment count
- remap count
- residency hit rate
