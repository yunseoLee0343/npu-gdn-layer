#include "npu_gdn_layer/residency_manager.hpp"
#include "npu_gdn_layer/trace.hpp"

namespace npu_gdn_layer {

std::optional<BufferLease> ResidencyManager::Acquire(const BufferAcquireRequest& request) {
  switch (request.buffer_class) {
    case BufferClass::WEIGHT_PERSISTENT_RO:
      return AcquirePersistentWeight(request);
    case BufferClass::STATE_PERSISTENT_RW:
      return AcquirePersistentState(request);
    case BufferClass::ACTIVATION_EPHEMERAL_RW:
      return AcquireEphemeralActivation(request);
    case BufferClass::CMD_DESC_LOWLAT:
      return AcquireCommandDescriptor(request);
  }
  return std::nullopt;
}

bool ResidencyManager::Release(std::uint64_t buffer_id) {
  auto it = by_id_.find(buffer_id);
  if (it == by_id_.end()) {
    return false;
  }

  Slot& slot = it->second;
  if (!slot.in_use) {
    return false;
  }

  slot.in_use = false;
  ++stats_.release_count;
  Emit(ResidencyEventType::kRelease, slot.buffer_class, slot.buffer_id);
  TraceEmit({TraceEventType::kBufferRelease, TraceNowNs(), 0, 0, 0, 0, slot.buffer_id, 0, 0,
             static_cast<std::uint64_t>(slot.buffer_class), 0, "release"});

  if (slot.buffer_class == BufferClass::ACTIVATION_EPHEMERAL_RW) {
    free_activation_ids_.push_back(slot.buffer_id);
  } else if (slot.buffer_class == BufferClass::CMD_DESC_LOWLAT) {
    free_cmd_ids_.push_back(slot.buffer_id);
  }

  return true;
}

void ResidencyManager::SetEventHook(EventHook hook) { hook_ = std::move(hook); }

ResidencyStats ResidencyManager::Stats() const { return stats_; }

std::optional<BufferLease> ResidencyManager::AcquirePersistentWeight(
    const BufferAcquireRequest& request) {
  if (request.requested_mutability != Mutability::kReadOnly) {
    ++stats_.reject_count;
    Emit(ResidencyEventType::kReject, request.buffer_class, 0, request.logical_key);
    return std::nullopt;
  }

  if (request.logical_key == 0) {
    ++stats_.reject_count;
    Emit(ResidencyEventType::kReject, request.buffer_class, 0, request.logical_key);
    return std::nullopt;
  }

  auto existing = weight_by_key_.find(request.logical_key);
  if (existing != weight_by_key_.end()) {
    Slot& slot = by_id_.at(existing->second);
    if (slot.bytes < request.bytes) {
      ++stats_.reject_count;
      Emit(ResidencyEventType::kReject, request.buffer_class, slot.buffer_id,
           request.logical_key);
      return std::nullopt;
    }
    TraceEmit({TraceEventType::kResidencyReuseHit, TraceNowNs(), 0, 0, 0, 0, slot.buffer_id, 0, 0,
               request.logical_key, request.bytes, "reuse_hit"});
    return LeaseSlot(slot, true, request.logical_key);
  }

  TraceEmit({TraceEventType::kResidencyReuseMiss, TraceNowNs(), 0, 0, 0, 0, 0, 0, 0,
             request.logical_key, request.bytes, "reuse_miss"});
  auto lease = CreateSlot(request, Mutability::kReadOnly);
  if (lease.has_value()) {
    weight_by_key_[request.logical_key] = lease->buffer_id;
  }
  return lease;
}

std::optional<BufferLease> ResidencyManager::AcquirePersistentState(
    const BufferAcquireRequest& request) {
  if (request.logical_key == 0) {
    ++stats_.reject_count;
    Emit(ResidencyEventType::kReject, request.buffer_class, 0, request.logical_key);
    return std::nullopt;
  }

  auto existing = state_by_key_.find(request.logical_key);
  if (existing != state_by_key_.end()) {
    Slot& slot = by_id_.at(existing->second);
    if (slot.bytes < request.bytes) {
      ++stats_.reject_count;
      Emit(ResidencyEventType::kReject, request.buffer_class, slot.buffer_id,
           request.logical_key);
      return std::nullopt;
    }
    TraceEmit({TraceEventType::kResidencyReuseHit, TraceNowNs(), 0, 0, 0, 0, slot.buffer_id, 0, 0,
               request.logical_key, request.bytes, "reuse_hit"});
    return LeaseSlot(slot, true, request.logical_key);
  }

  TraceEmit({TraceEventType::kResidencyReuseMiss, TraceNowNs(), 0, 0, 0, 0, 0, 0, 0,
             request.logical_key, request.bytes, "reuse_miss"});
  auto lease = CreateSlot(request, Mutability::kReadWriteHot);
  if (lease.has_value()) {
    state_by_key_[request.logical_key] = lease->buffer_id;
  }
  return lease;
}

std::optional<BufferLease> ResidencyManager::AcquireEphemeralActivation(
    const BufferAcquireRequest& request) {
  for (auto it = free_activation_ids_.begin(); it != free_activation_ids_.end(); ++it) {
    Slot& slot = by_id_.at(*it);
    if (slot.bytes >= request.bytes && !slot.in_use) {
      const std::uint64_t id = slot.buffer_id;
      free_activation_ids_.erase(it);
      return LeaseSlot(slot, true);
    }
  }

  TraceEmit({TraceEventType::kResidencyReuseMiss, TraceNowNs(), 0, 0, 0, 0, 0, 0, 0,
             request.logical_key, request.bytes, "reuse_miss"});
  return CreateSlot(request, Mutability::kReadWriteHot);
}

std::optional<BufferLease> ResidencyManager::AcquireCommandDescriptor(
    const BufferAcquireRequest& request) {
  for (auto it = free_cmd_ids_.begin(); it != free_cmd_ids_.end(); ++it) {
    Slot& slot = by_id_.at(*it);
    if (slot.bytes >= request.bytes && !slot.in_use) {
      free_cmd_ids_.erase(it);
      return LeaseSlot(slot, true);
    }
  }

  TraceEmit({TraceEventType::kResidencyReuseMiss, TraceNowNs(), 0, 0, 0, 0, 0, 0, 0,
             request.logical_key, request.bytes, "reuse_miss"});
  return CreateSlot(request, Mutability::kWriteRare);
}

std::optional<BufferLease> ResidencyManager::CreateSlot(const BufferAcquireRequest& request,
                                                        Mutability stored_mutability) {
  Slot slot;
  slot.buffer_id = next_buffer_id_++;
  slot.buffer_class = request.buffer_class;
  slot.bytes = request.bytes;
  slot.mutability = stored_mutability;
  slot.in_use = false;
  slot.generation = 0;
  slot.mapped_hint = request.prefer_mapped;

  auto [it, inserted] = by_id_.emplace(slot.buffer_id, slot);
  if (!inserted) {
    ++stats_.reject_count;
    Emit(ResidencyEventType::kReject, request.buffer_class, 0, request.logical_key);
    return std::nullopt;
  }

  ++stats_.create_count;
  Emit(ResidencyEventType::kCreate, request.buffer_class, it->second.buffer_id,
       request.logical_key);
  return LeaseSlot(it->second, false, request.logical_key);
}

std::optional<BufferLease> ResidencyManager::LeaseSlot(Slot& slot, bool reused,
                                                       std::uint64_t logical_key) {
  if (slot.in_use) {
    // First-pass API forbids concurrent leases of the same slot.
    ++stats_.reject_count;
    Emit(ResidencyEventType::kReject, slot.buffer_class, slot.buffer_id, logical_key);
    return std::nullopt;
  }

  slot.in_use = true;
  ++slot.generation;
  ++stats_.acquire_count;
  Emit(ResidencyEventType::kAcquire, slot.buffer_class, slot.buffer_id, logical_key);

  if (reused) {
    ++stats_.reuse_count;
    Emit(ResidencyEventType::kReuse, slot.buffer_class, slot.buffer_id, logical_key);
  }

  BufferLease lease;
  lease.buffer_id = slot.buffer_id;
  lease.buffer_class = slot.buffer_class;
  lease.bytes = slot.bytes;
  lease.mutability = slot.mutability;
  lease.generation = slot.generation;
  return lease;
}

void ResidencyManager::Emit(ResidencyEventType type, BufferClass cls,
                            std::uint64_t buffer_id, std::uint64_t logical_key) {
  if (!hook_) {
    return;
  }
  ResidencyEvent event;
  event.type = type;
  event.buffer_class = cls;
  event.buffer_id = buffer_id;
  event.logical_key = logical_key;
  hook_(event);
}

}  // namespace npu_gdn_layer
