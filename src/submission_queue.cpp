#include "npu_gdn_layer/submission_queue.hpp"

#include <chrono>

namespace npu_gdn_layer {

SubmissionQueueMpsc::SubmissionQueueMpsc(std::size_t capacity,
                                         PublishContract* contract)
    : capacity_(capacity),
      contract_(contract),
      ring_(capacity),
      completion_seen_(capacity) {
  for (auto& v : completion_seen_) {
    v.store(0, std::memory_order_relaxed);
  }
}

std::optional<std::uint64_t> SubmissionQueueMpsc::Enqueue(const SubmissionJob& job) {
  if (contract_ == nullptr || capacity_ == 0) {
    ++enqueue_drop_;
    return std::nullopt;
  }

  std::optional<SubmissionTicket> ticket;
  {
    std::lock_guard<std::mutex> lk(contract_mu_);
    ticket = contract_->prepare_submission(job);
  }
  if (!ticket.has_value()) {
    ++enqueue_drop_;
    return std::nullopt;
  }

  const std::uint64_t tail = tail_.fetch_add(1, std::memory_order_acq_rel);
  const std::uint64_t in_flight = tail - head_;
  if (in_flight >= capacity_) {
    ++enqueue_drop_;
    return std::nullopt;
  }

  Slot& slot = ring_[tail % capacity_];
  if (slot.occupied.load(std::memory_order_acquire)) {
    ++enqueue_drop_;
    return std::nullopt;
  }

  QueueEntry e;
  e.job_id = job.job_id;
  e.ticket = *ticket;
  e.ticket.sequence_id = next_local_sequence_.fetch_add(1, std::memory_order_relaxed);
  e.enqueue_tsc = ReadMonotonicTsc();

  slot.entry = e;
  slot.occupied.store(true, std::memory_order_release);

  ++enqueue_ok_;
  return e.ticket.ticket_id;
}

std::optional<QueueEntry> SubmissionQueueMpsc::Dequeue() {
  if (capacity_ == 0) {
    return std::nullopt;
  }

  Slot& slot = ring_[head_ % capacity_];
  if (!slot.occupied.load(std::memory_order_acquire)) {
    return std::nullopt;
  }

  QueueEntry out = slot.entry;
  out.dequeue_tsc = ReadMonotonicTsc();

  slot.occupied.store(false, std::memory_order_release);
  ++head_;
  ++dequeue_ok_;
  return out;
}

bool SubmissionQueueMpsc::PublishOne(const QueueEntry& entry) {
  if (contract_ == nullptr) {
    return false;
  }

  std::lock_guard<std::mutex> lk(contract_mu_);
  if (!contract_->publish_submission(entry.ticket.ticket_id)) {
    return false;
  }

  return contract_->ring_doorbell(static_cast<std::uint32_t>(entry.ticket.queue_id));
}

std::size_t SubmissionQueueMpsc::PublishBatch(std::size_t max_batch,
                                              std::uint32_t queue_id) {
  if (contract_ == nullptr || max_batch == 0) {
    return 0;
  }

  std::size_t published = 0;
  while (published < max_batch) {
    auto maybe = Dequeue();
    if (!maybe.has_value()) {
      break;
    }

    std::lock_guard<std::mutex> lk(contract_mu_);
    if (!contract_->publish_submission(maybe->ticket.ticket_id)) {
      continue;
    }
    ++published;
  }

  if (published > 0) {
    std::lock_guard<std::mutex> lk(contract_mu_);
    if (contract_->ring_doorbell(queue_id)) {
    ++doorbell_batches_;
    }
  }

  return published;
}

bool SubmissionQueueMpsc::CorrelateCompletion(std::uint64_t ticket_id, AwaitResult* out) {
  if (out == nullptr || contract_ == nullptr || capacity_ == 0) {
    return false;
  }

  const std::size_t idx = ticket_id % capacity_;
  std::uint64_t expected = 0;
  std::uint64_t tag =
      (completion_seen_mask_seed_.load(std::memory_order_relaxed) ^ ticket_id) | 1ULL;

  if (!completion_seen_[idx].compare_exchange_strong(expected, tag,
                                                      std::memory_order_acq_rel)) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(contract_mu_);
    *out = contract_->await_completion(ticket_id);
  }
  return true;
}

QueueStats SubmissionQueueMpsc::Stats() const {
  QueueStats s;
  s.enqueue_ok = enqueue_ok_.load(std::memory_order_relaxed);
  s.enqueue_drop = enqueue_drop_.load(std::memory_order_relaxed);
  s.dequeue_ok = dequeue_ok_.load(std::memory_order_relaxed);
  s.doorbell_batches = doorbell_batches_.load(std::memory_order_relaxed);
  return s;
}

std::uint64_t SubmissionQueueMpsc::ReadMonotonicTsc() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

}  // namespace npu_gdn_layer
