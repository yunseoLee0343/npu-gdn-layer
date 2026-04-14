#include "npu_gdn_layer/buffer_policy.hpp"
#include "npu_gdn_layer/trace.hpp"

namespace npu_gdn_layer {

namespace {

BufferPolicyDecision BuildWeightDecision(const BufferRequest& request) {
  BufferPolicyDecision out;
  out.buffer_class = BufferClass::WEIGHT_PERSISTENT_RO;
  out.preferred_residency = ResidencyPolicy::kPinnedDeviceLocal;
  out.mapping_policy = request.requires_host_access ? MappingPolicy::kHostReadMostly
                                                    : MappingPolicy::kDeviceOnly;
  out.host_visible = request.requires_host_access;
  out.device_visible = true;
  out.expected_reuse = ReusePattern::kSessionPersistent;
  out.mutability = Mutability::kReadOnly;
  out.low_latency_control_plane = false;
  return out;
}

BufferPolicyDecision BuildStateDecision(const BufferRequest& request) {
  BufferPolicyDecision out;
  out.buffer_class = BufferClass::STATE_PERSISTENT_RW;
  out.preferred_residency = ResidencyPolicy::kHostVisibleCoherent;
  out.mapping_policy = request.requires_host_access ? MappingPolicy::kHostReadWriteCoherent
                                                    : MappingPolicy::kDeviceOnly;
  out.host_visible = request.requires_host_access;
  out.device_visible = true;
  out.expected_reuse = ReusePattern::kSequenceLocal;
  out.mutability = Mutability::kReadWriteHot;
  out.low_latency_control_plane = false;
  return out;
}

BufferPolicyDecision BuildCommandDecision(const BufferRequest& request) {
  BufferPolicyDecision out;
  out.buffer_class = BufferClass::CMD_DESC_LOWLAT;
  out.preferred_residency = ResidencyPolicy::kLatencyCriticalNoEvict;
  out.mapping_policy = request.requires_host_access ? MappingPolicy::kHostWriteCombined
                                                    : MappingPolicy::kDeviceOnly;
  out.host_visible = request.requires_host_access;
  out.device_visible = true;
  out.expected_reuse = ReusePattern::kStepLocal;
  out.mutability = Mutability::kWriteRare;
  out.low_latency_control_plane = true;
  return out;
}

BufferPolicyDecision BuildActivationDecision() {
  BufferPolicyDecision out;
  out.buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  out.preferred_residency = ResidencyPolicy::kStreamedOnDemand;
  out.mapping_policy = MappingPolicy::kDeviceOnly;
  out.host_visible = false;
  out.device_visible = true;
  out.expected_reuse = ReusePattern::kStepLocal;
  out.mutability = Mutability::kReadWriteHot;
  out.low_latency_control_plane = false;
  return out;
}

}  // namespace

BufferPolicyDecision HeuristicBufferPolicy::Classify(
    const BufferRequest& request) const {
  TraceEvent event;
  event.type = TraceEventType::kBufferClassification;
  event.tsc = TraceNowNs();
  event.sequence_id = request.sequence.sequence_id;
  event.session_id = request.sequence.session_id;
  event.value0 = request.bytes;
  if (request.is_command_or_descriptor || request.latency_critical) {
    auto out = BuildCommandDecision(request);
    event.value1 = static_cast<std::uint64_t>(out.buffer_class);
    TraceEmit(event);
    return out;
  }
  if (request.is_weight) {
    auto out = BuildWeightDecision(request);
    event.value1 = static_cast<std::uint64_t>(out.buffer_class);
    TraceEmit(event);
    return out;
  }
  if (request.is_recurrent_state) {
    auto out = BuildStateDecision(request);
    event.value1 = static_cast<std::uint64_t>(out.buffer_class);
    TraceEmit(event);
    return out;
  }
  if (request.is_activation) {
    auto out = BuildActivationDecision();
    event.value1 = static_cast<std::uint64_t>(out.buffer_class);
    TraceEmit(event);
    return out;
  }

  // Conservative fallback: transient RW data-plane allocation.
  return BuildActivationDecision();
}

ResidencyPolicy HeuristicBufferPolicy::ResolveResidency(
    BufferClass buffer_class, const SequenceStateHandle& /*seq*/) const {
  switch (buffer_class) {
    case BufferClass::WEIGHT_PERSISTENT_RO:
      return ResidencyPolicy::kPinnedDeviceLocal;
    case BufferClass::STATE_PERSISTENT_RW:
      return ResidencyPolicy::kHostVisibleCoherent;
    case BufferClass::ACTIVATION_EPHEMERAL_RW:
      return ResidencyPolicy::kStreamedOnDemand;
    case BufferClass::CMD_DESC_LOWLAT:
      return ResidencyPolicy::kLatencyCriticalNoEvict;
  }
  return ResidencyPolicy::kStreamedOnDemand;
}

MappingPolicy HeuristicBufferPolicy::ResolveMapping(
    BufferClass buffer_class, bool host_observability_enabled) const {
  switch (buffer_class) {
    case BufferClass::WEIGHT_PERSISTENT_RO:
      return host_observability_enabled ? MappingPolicy::kHostReadMostly
                                        : MappingPolicy::kDeviceOnly;
    case BufferClass::STATE_PERSISTENT_RW:
      return host_observability_enabled ? MappingPolicy::kHostReadWriteCoherent
                                        : MappingPolicy::kDeviceOnly;
    case BufferClass::ACTIVATION_EPHEMERAL_RW:
      return host_observability_enabled ? MappingPolicy::kHostReadMostly
                                        : MappingPolicy::kDeviceOnly;
    case BufferClass::CMD_DESC_LOWLAT:
      return host_observability_enabled ? MappingPolicy::kHostWriteCombined
                                        : MappingPolicy::kDeviceOnly;
  }
  return MappingPolicy::kDeviceOnly;
}

bool HeuristicBufferPolicy::ValidateStateRegion(const GdnStateRegion& region,
                                                const BufferObjectDesc& buffer) const {
  if (buffer.buffer_class != BufferClass::STATE_PERSISTENT_RW) {
    return false;
  }
  if (region.buffer_id != buffer.buffer_id) {
    return false;
  }
  if (region.byte_length == 0) {
    return false;
  }
  const std::uint64_t end = region.byte_offset + region.byte_length;
  return end <= buffer.bytes;
}

}  // namespace npu_gdn_layer
