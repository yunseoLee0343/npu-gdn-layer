#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include "npu_gdn_layer/submission.hpp"

namespace npu_gdn_layer {

// Host-side publication states make ordering transitions explicit.
enum class PublishState : std::uint8_t {
  kAllocated,
  kHostWriteComplete,
  kDeviceVisible,
  kDoorbellSafe,
  kDoorbellRung,
  kCompleted,
};

// Input record for one submission operation.
// Ownership/lifetime:
// - Descriptor and metadata references are external; this struct tracks publish
//   progress and ordering guarantees in the middle-layer.
struct SubmissionJob {
  std::uint64_t job_id = 0;
  std::uint32_t queue_id = 0;

  std::uint64_t descriptor_bo_id = 0;
  std::uint64_t descriptor_offset = 0;
  std::uint32_t descriptor_bytes = 0;

  std::uint64_t metadata_bo_id = 0;
  std::uint64_t metadata_offset = 0;
  std::uint32_t metadata_bytes = 0;

  GdnStateRegion dynamic_state;

  bool host_descriptor_write_complete = false;
  bool host_metadata_write_complete = false;
  bool host_state_write_complete = false;

  PublishState state = PublishState::kAllocated;
};

// Records publish transitions for observability and triage.
struct PublishTracePoint {
  std::uint64_t tsc = 0;
  PublishState state = PublishState::kAllocated;
};

struct AwaitResult {
  bool completed = false;
  bool sequence_match = false;
  SubmissionTicket ticket;
};

// Explicit submission ordering contract wrapper.
// This wrapper is the only legal host-side path from host writes to doorbell.
class PublishContract {
 public:
  PublishContract() = default;

  // Stage 1: validate host writes complete and allocate queue sequence/ticket.
  std::optional<SubmissionTicket> prepare_submission(const SubmissionJob& job);

  // Stage 2: transition from host-write-complete to device-visible to doorbell-safe.
  bool publish_submission(std::uint64_t ticket_id);

  // Stage 3: ring doorbell only for doorbell-safe submissions.
  bool ring_doorbell(std::uint32_t queue_id);

  // Stage 4: wait for completion and verify expected sequence identity.
  AwaitResult await_completion(std::uint64_t ticket_id);

  // Prototype-only test hook to simulate retire notification.
  bool mark_completed(std::uint64_t ticket_id);

 private:
  struct Record {
    SubmissionJob job;
    SubmissionTicket ticket;
    PublishTracePoint host_write_complete;
    PublishTracePoint device_visible;
    PublishTracePoint doorbell_safe;
    PublishTracePoint doorbell_rung;
    PublishTracePoint completed;
  };

  static std::uint64_t ReadMonotonicTsc();
  static bool HasHostWriteComplete(const SubmissionJob& job);

  std::uint64_t next_ticket_id_ = 1;
  std::unordered_map<std::uint64_t, std::uint64_t> next_seq_by_queue_;
  std::unordered_map<std::uint64_t, Record> by_ticket_;
  std::unordered_map<std::uint32_t, std::uint64_t> last_doorbell_ticket_by_queue_;
};

}  // namespace npu_gdn_layer
