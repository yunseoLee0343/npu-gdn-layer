#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

#include "npu_gdn_layer/buffer_policy.hpp"

namespace npu_gdn_layer {

// Coarse event stream for future observability integrations.
enum class ResidencyEventType : std::uint8_t {
  kCreate,
  kAcquire,
  kReuse,
  kRelease,
  kReject,
};

struct ResidencyEvent {
  ResidencyEventType type = ResidencyEventType::kCreate;
  BufferClass buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  std::uint64_t buffer_id = 0;
  std::uint64_t logical_key = 0;
};

struct BufferAcquireRequest {
  BufferClass buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  std::uint64_t bytes = 0;

  // Stable key for persistent objects (weights/state).
  std::uint64_t logical_key = 0;

  // Used to prevent invalid mutable access patterns.
  Mutability requested_mutability = Mutability::kReadWriteHot;

  // Hint only; no kernel mapping is performed in this host-side prototype.
  bool prefer_mapped = false;
};

struct BufferLease {
  std::uint64_t buffer_id = 0;
  BufferClass buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  std::uint64_t bytes = 0;
  Mutability mutability = Mutability::kReadWriteHot;

  // Monotonic generation increments on each acquire.
  std::uint32_t generation = 0;
};

struct ResidencyStats {
  std::uint64_t create_count = 0;
  std::uint64_t acquire_count = 0;
  std::uint64_t reuse_count = 0;
  std::uint64_t release_count = 0;
  std::uint64_t reject_count = 0;
};

// Host-side middle-layer buffer pool + residency prototype.
// No kernel/driver calls are made here; this models policy and lifecycle shape
// that will later be wired to OpenVINO/driver-facing code.
class ResidencyManager {
 public:
  using EventHook = std::function<void(const ResidencyEvent&)>;

  ResidencyManager() = default;

  std::optional<BufferLease> Acquire(const BufferAcquireRequest& request);
  bool Release(std::uint64_t buffer_id);

  void SetEventHook(EventHook hook);
  ResidencyStats Stats() const;

 private:
  struct Slot {
    std::uint64_t buffer_id = 0;
    BufferClass buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
    std::uint64_t bytes = 0;
    Mutability mutability = Mutability::kReadWriteHot;

    bool in_use = false;
    std::uint32_t generation = 0;
    bool mapped_hint = false;
  };

  std::optional<BufferLease> AcquirePersistentWeight(const BufferAcquireRequest& request);
  std::optional<BufferLease> AcquirePersistentState(const BufferAcquireRequest& request);
  std::optional<BufferLease> AcquireEphemeralActivation(const BufferAcquireRequest& request);
  std::optional<BufferLease> AcquireCommandDescriptor(const BufferAcquireRequest& request);

  std::optional<BufferLease> CreateSlot(const BufferAcquireRequest& request,
                                        Mutability stored_mutability);
  std::optional<BufferLease> LeaseSlot(Slot& slot, bool reused, std::uint64_t logical_key = 0);
  void Emit(ResidencyEventType type, BufferClass cls, std::uint64_t buffer_id,
            std::uint64_t logical_key = 0);

  std::uint64_t next_buffer_id_ = 1;

  std::unordered_map<std::uint64_t, Slot> by_id_;

  // Separate pools by class.
  std::unordered_map<std::uint64_t, std::uint64_t> weight_by_key_;
  std::unordered_map<std::uint64_t, std::uint64_t> state_by_key_;
  std::vector<std::uint64_t> free_activation_ids_;
  std::vector<std::uint64_t> free_cmd_ids_;

  EventHook hook_;
  ResidencyStats stats_;
};

}  // namespace npu_gdn_layer
