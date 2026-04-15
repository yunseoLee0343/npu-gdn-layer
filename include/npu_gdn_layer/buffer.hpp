#pragma once

#include <cstdint>

#include "npu_gdn_layer/types.hpp"

namespace npu_gdn_layer {

// First-pass BO metadata contract for middle-layer policy decisions.
// Ownership/lifetime:
// - `buffer_id` is runtime-assigned and stable for object lifetime.
// - Memory allocation is owned by allocator/driver integration layer.
// - This struct is a policy/control record and does not own mapped memory.
struct BufferObjectDesc {
  std::uint64_t buffer_id = 0;
  BufferClass buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  BufferPlane plane = BufferPlane::kDataPlane;

  std::uint64_t bytes = 0;
  std::uint32_t alignment = 0;

  ResidencyPolicy residency = ResidencyPolicy::kStreamedOnDemand;
  MappingPolicy mapping = MappingPolicy::kDeviceOnly;

  // ivpu/OpenVINO-facing integration values.
  std::uint64_t bo_handle = 0;
  std::uint64_t sg_table_entries = 0;

  // Scope tags for policy scheduling.
  std::uint64_t session_id = 0;
  std::uint64_t sequence_id = 0;

  // Memory Policy: primary abstraction for classifying placement and mapping.
  static constexpr RuntimePath kPrimaryPath = RuntimePath::kMemoryPolicy;
};

// Non-owning host mapping lease for explicit map/unmap boundaries.
// Ownership/lifetime:
// - Valid only between successful map and unmap API calls.
// - `host_ptr` may become invalid after queue submission depending on policy.
// - Caller must not persist raw pointer outside lease lifetime.
struct BufferMappingLease {
  std::uint64_t buffer_id = 0;
  void* host_ptr = nullptr;
  std::uint64_t mapped_bytes = 0;
  MappingPolicy mapping = MappingPolicy::kDeviceOnly;
  bool requires_explicit_sync = true;

  static constexpr RuntimePath kPrimaryPath = RuntimePath::kMemoryPolicy;
};

// Memory Policy track interface surface.
class BufferPolicy {
 public:
  virtual ~BufferPolicy() = default;

  virtual ResidencyPolicy ResolveResidency(BufferClass buffer_class,
                                           const SequenceStateHandle& seq) const = 0;

  virtual MappingPolicy ResolveMapping(BufferClass buffer_class,
                                       bool host_observability_enabled) const = 0;

  virtual bool ValidateStateRegion(const GdnStateRegion& region,
                                   const BufferObjectDesc& buffer) const = 0;
};

}  // namespace npu_gdn_layer
