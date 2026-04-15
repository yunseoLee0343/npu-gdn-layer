#include "npu_gdn_layer/publish_contract.hpp"
#include "npu_gdn_layer/trace.hpp"

#include <chrono>

namespace npu_gdn_layer {

std::optional<SubmissionTicket> PublishContract::prepare_submission(
    const SubmissionJob& job) {
  if (!HasHostWriteComplete(job)) {
    return std::nullopt;
  }

  Record record;
  record.job = job;
  record.job.state = PublishState::kHostWriteComplete;

  const std::uint64_t ticket_id = next_ticket_id_++;
  const std::uint64_t seq = ++next_seq_by_queue_[job.queue_id];

  record.ticket.session_id = 0;
  record.ticket.sequence_id = seq;
  record.ticket.queue_id = job.queue_id;
  record.ticket.ticket_id = ticket_id;
  record.ticket.descriptor_head_offset = job.descriptor_offset;
  record.ticket.descriptor_count = 1;
  record.ticket.enqueue_tsc = ReadMonotonicTsc();
  record.ticket.publish_tsc = 0;

  record.host_write_complete.tsc = record.ticket.enqueue_tsc;
  record.host_write_complete.state = PublishState::kHostWriteComplete;

  by_ticket_[ticket_id] = record;
  TraceEmit({TraceEventType::kSubmissionPrepared, TraceNowNs(), 0, seq, ticket_id, job.job_id, 0,
             job.queue_id, 0, job.descriptor_bytes, job.metadata_bytes, "prepared"});
  return by_ticket_.at(ticket_id).ticket;
}

bool PublishContract::publish_submission(std::uint64_t ticket_id) {
  auto it = by_ticket_.find(ticket_id);
  if (it == by_ticket_.end()) {
    return false;
  }

  Record& r = it->second;
  if (r.job.state != PublishState::kHostWriteComplete) {
    return false;
  }

  // Host->device publication contract:
  // 1) host writes complete, 2) release-style publication to device-visible,
  // 3) doorbell-safe transition only after descriptor+metadata+state visibility.
  r.job.state = PublishState::kDeviceVisible;
  r.device_visible.tsc = ReadMonotonicTsc();
  r.device_visible.state = PublishState::kDeviceVisible;

  r.job.state = PublishState::kDoorbellSafe;
  r.doorbell_safe.tsc = ReadMonotonicTsc();
  r.doorbell_safe.state = PublishState::kDoorbellSafe;

  r.ticket.publish_tsc = r.doorbell_safe.tsc;
  TraceEmit({TraceEventType::kSubmissionPublished, TraceNowNs(), 0, r.ticket.sequence_id,
             r.ticket.ticket_id, r.job.job_id, 0, static_cast<std::uint32_t>(r.ticket.queue_id), 0, 0, 0, "published"});
  return true;
}

bool PublishContract::ring_doorbell(std::uint32_t queue_id) {
  // Select the highest sequence ticket on this queue that is doorbell-safe.
  std::uint64_t selected_ticket_id = 0;
  std::uint64_t selected_seq = 0;

  for (auto& [ticket_id, rec] : by_ticket_) {
    if (rec.ticket.queue_id != queue_id) {
      continue;
    }
    if (rec.job.state != PublishState::kDoorbellSafe) {
      continue;
    }
    if (rec.ticket.sequence_id >= selected_seq) {
      selected_ticket_id = ticket_id;
      selected_seq = rec.ticket.sequence_id;
    }
  }

  if (selected_ticket_id == 0) {
    return false;
  }

  Record& r = by_ticket_.at(selected_ticket_id);
  r.job.state = PublishState::kDoorbellRung;
  r.doorbell_rung.tsc = ReadMonotonicTsc();
  r.doorbell_rung.state = PublishState::kDoorbellRung;

  last_doorbell_ticket_by_queue_[queue_id] = selected_ticket_id;
  TraceEmit({TraceEventType::kDoorbellRung, TraceNowNs(), 0, r.ticket.sequence_id,
             r.ticket.ticket_id, r.job.job_id, 0, queue_id, 0, 0, 0, "doorbell"});
  return true;
}

AwaitResult PublishContract::await_completion(std::uint64_t ticket_id) {
  AwaitResult out;

  auto it = by_ticket_.find(ticket_id);
  if (it == by_ticket_.end()) {
    return out;
  }

  const Record& r = it->second;
  out.ticket = r.ticket;
  out.completed = (r.job.state == PublishState::kCompleted);

  const auto last_it = last_doorbell_ticket_by_queue_.find(r.ticket.queue_id);
  if (last_it != last_doorbell_ticket_by_queue_.end()) {
    const Record& last = by_ticket_.at(last_it->second);
    out.sequence_match = (r.ticket.sequence_id <= last.ticket.sequence_id);
  }

  return out;
}

bool PublishContract::mark_completed(std::uint64_t ticket_id) {
  auto it = by_ticket_.find(ticket_id);
  if (it == by_ticket_.end()) {
    return false;
  }

  Record& r = it->second;
  if (r.job.state != PublishState::kDoorbellRung) {
    return false;
  }

  r.job.state = PublishState::kCompleted;
  r.completed.tsc = ReadMonotonicTsc();
  r.completed.state = PublishState::kCompleted;
  return true;
}

std::uint64_t PublishContract::ReadMonotonicTsc() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

bool PublishContract::HasHostWriteComplete(const SubmissionJob& job) {
  return job.host_descriptor_write_complete && job.host_metadata_write_complete &&
         job.host_state_write_complete;
}

}  // namespace npu_gdn_layer
