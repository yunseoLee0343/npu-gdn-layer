#pragma once

#include <cstdint>

#include "npu_gdn_layer/buffer.hpp"

namespace npu_gdn_layer {

// Lifetime expectation for policy heuristics.
enum class ReusePattern : std::uint8_t {
  kOneShot,
  kStepLocal,
  kSequenceLocal,
  kSessionPersistent,
};

// Mutability expectation for memory behavior and map strategy.
enum class Mutability : std::uint8_t {
  kReadOnly,
  kWriteRare,
  kReadWriteHot,
};

// Input description used to classify external tensor/buffer requests.
// Ownership/lifetime:
// - Caller-owned value object.
// - Passed by value/reference into classifier; no retained pointers.
struct BufferRequest {
  std::uint64_t request_id = 0;
  std::uint64_t bytes = 0;

  // Signals from model/runtime metadata.
  bool is_weight = false;
  bool is_recurrent_state = false;
  bool is_activation = false;
  bool is_command_or_descriptor = false;

  bool requires_host_access = false;
  bool latency_critical = false;

  SequenceStateHandle sequence;
};

// Consolidated policy output used by allocator, mapper, and bind/unbind hooks.
// Ownership/lifetime:
// - Runtime-owned value object copied across policy/application boundaries.
struct BufferPolicyDecision {
  BufferClass buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  ResidencyPolicy preferred_residency = ResidencyPolicy::kStreamedOnDemand;
  MappingPolicy mapping_policy = MappingPolicy::kDeviceOnly;

  bool host_visible = false;
  bool device_visible = true;
  ReusePattern expected_reuse = ReusePattern::kOneShot;
  Mutability mutability = Mutability::kReadWriteHot;
  bool low_latency_control_plane = false;

  static constexpr RuntimePath kPrimaryPath = RuntimePath::kMemoryPolicy;
};

// First-pass heuristic policy engine for Qwen3-Next GDN stateful paths.
class HeuristicBufferPolicy final : public BufferPolicy {
 public:
  HeuristicBufferPolicy() = default;
  ~HeuristicBufferPolicy() override = default;

  BufferPolicyDecision Classify(const BufferRequest& request) const;

  ResidencyPolicy ResolveResidency(BufferClass buffer_class,
                                   const SequenceStateHandle& seq) const override;

  MappingPolicy ResolveMapping(BufferClass buffer_class,
                               bool host_observability_enabled) const override;

  bool ValidateStateRegion(const GdnStateRegion& region,
                           const BufferObjectDesc& buffer) const override;
};

}  // namespace npu_gdn_layer
