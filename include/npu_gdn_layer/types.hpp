#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace npu_gdn_layer {

// High-level routing tag for roadmap ownership.
enum class RuntimePath : std::uint8_t {
  kMemoryPolicy,
  kSubmissionCorrectness,
  kCompletionLatency,
};

// Distinguishes data-plane and control-plane BO usage.
enum class BufferPlane : std::uint8_t {
  kDataPlane,
  kControlPlane,
};

// Buffer classes are explicit so policy can separate persistent state,
// transient activation/scratch, and low-latency command descriptors.
enum class BufferClass : std::uint8_t {
  WEIGHT_PERSISTENT_RO,
  STATE_PERSISTENT_RW,
  ACTIVATION_EPHEMERAL_RW,
  CMD_DESC_LOWLAT,
};

// Residency intent for BO placement across prefill/decode/state transitions.
enum class ResidencyPolicy : std::uint8_t {
  kPinnedDeviceLocal,      // Keep BO resident on NPU-local memory when possible.
  kHostVisibleCoherent,    // Host-visible mapping for frequent host/NPU sync points.
  kStreamedOnDemand,       // Allow migration/eviction for ephemeral buffers.
  kLatencyCriticalNoEvict, // For descriptor/control BOs on critical path.
};

// Mapping mode controls host visibility and synchronization semantics.
enum class MappingPolicy : std::uint8_t {
  kDeviceOnly,           // No host mapping during steady state.
  kHostReadMostly,       // Host diagnostics/trace reads.
  kHostWriteCombined,    // Host command or staging writes.
  kHostReadWriteCoherent // Debug/validation mode; usually higher overhead.
};

// Stable handle for sequence-scoped GDN state ownership.
// Ownership: minted by runtime on session/state bind, valid until release.
struct SequenceStateHandle {
  std::uint64_t session_id = 0;
  std::uint64_t sequence_id = 0;
  std::uint32_t generation = 0;

  // Memory Policy + Submission Correctness: binds state residency and queue order
  // decisions to a specific decode sequence timeline.
  static constexpr RuntimePath kPrimaryPath = RuntimePath::kSubmissionCorrectness;
};

// Byte region in a persistent state BO used by GDN stateful paths.
// Ownership: references memory owned by a BufferObject contract; never owning raw
// allocations directly.
struct GdnStateRegion {
  std::uint64_t buffer_id = 0;
  std::uint64_t byte_offset = 0;
  std::uint64_t byte_length = 0;
  std::uint32_t state_slot = 0;

  // Memory Policy: explicit residency surface for persistent KV/GDN state.
  static constexpr RuntimePath kPrimaryPath = RuntimePath::kMemoryPolicy;
};

// Contract for publishing descriptors to hardware-visible queue memory before
// ringing doorbell. This keeps ordering explicit in public API shape.
struct DoorbellPublishContract {
  std::uint32_t queue_id = 0;
  std::uint64_t expected_tail_seq = 0;
  bool require_release_fence = true;
  bool require_mmio_posted_write_flush = true;

  // Submission Correctness: formalizes the ordering invariants required before
  // doorbell publication.
  static constexpr RuntimePath kPrimaryPath = RuntimePath::kSubmissionCorrectness;
};

// Optional textual label helper for diagnostics; non-owning view.
constexpr std::string_view ToString(BufferClass cls) {
  switch (cls) {
    case BufferClass::WEIGHT_PERSISTENT_RO:
      return "WEIGHT_PERSISTENT_RO";
    case BufferClass::STATE_PERSISTENT_RW:
      return "STATE_PERSISTENT_RW";
    case BufferClass::ACTIVATION_EPHEMERAL_RW:
      return "ACTIVATION_EPHEMERAL_RW";
    case BufferClass::CMD_DESC_LOWLAT:
      return "CMD_DESC_LOWLAT";
  }
  return "UNKNOWN";
}

}  // namespace npu_gdn_layer
