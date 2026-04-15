#pragma once

#include <cstdint>

#include "npu_gdn_layer/types.hpp"

namespace npu_gdn_layer {

// Queue submission token returned after descriptor enqueue/publish.
// Ownership/lifetime:
// - Issued by runtime; immutable after creation.
// - Valid until completion retire or explicit cancellation.
// - Used as key for completion lookup and latency attribution.
struct SubmissionTicket {
  std::uint64_t session_id = 0;
  std::uint64_t sequence_id = 0;
  std::uint64_t queue_id = 0;
  std::uint64_t ticket_id = 0;

  std::uint64_t descriptor_head_offset = 0;
  std::uint32_t descriptor_count = 0;

  std::uint64_t enqueue_tsc = 0;
  std::uint64_t publish_tsc = 0;

  static constexpr RuntimePath kPrimaryPath = RuntimePath::kSubmissionCorrectness;
};

// Submission packet binds descriptor bytes to ordering contract.
// Ownership/lifetime:
// - Descriptor memory is caller-owned until Publish returns success.
// - For CMD_DESC_LOWLAT path, recommended memory class is non-evictable control BO.
struct SubmissionRequest {
  SequenceStateHandle seq;
  std::uint32_t queue_id = 0;

  std::uint64_t descriptor_bo_id = 0;
  std::uint64_t descriptor_offset = 0;
  std::uint32_t descriptor_bytes = 0;

  DoorbellPublishContract publish_contract;

  bool is_state_transition_boundary = false;

  static constexpr RuntimePath kPrimaryPath = RuntimePath::kSubmissionCorrectness;
};

class SubmissionQueue {
 public:
  virtual ~SubmissionQueue() = default;

  // Reserve a monotonic sequence number and queue slot for lock-free writers.
  virtual std::uint64_t ReserveSequence(std::uint32_t queue_id) = 0;

  // Publish descriptor visibility + doorbell with explicit ordering contract.
  virtual SubmissionTicket Publish(const SubmissionRequest& request) = 0;

  // Pre-submit validation for sequence monotonicity and descriptor boundaries.
  virtual bool Validate(const SubmissionRequest& request) const = 0;
};

}  // namespace npu_gdn_layer
