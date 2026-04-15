#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include <mutex>

#include "npu_gdn_layer/publish_contract.hpp"

namespace npu_gdn_layer {

// Concurrency model: MPSC (multi-producer, single-consumer) ring queue.
// - Producers reserve slots with atomic fetch_add on tail (no global coarse lock).
// - Consumer drains in sequence order via head cursor.
// - Queue is bounded; failed enqueue indicates backpressure.
//
// Placement in runtime pipeline:
// prepare_submission -> enqueue ticket/job -> consumer publish_submission /
// ring_doorbell batching -> await_completion correlation.
struct QueueEntry {
  std::uint64_t job_id = 0;
  SubmissionTicket ticket;

  std::uint64_t enqueue_tsc = 0;
  std::uint64_t dequeue_tsc = 0;
};

struct QueueStats {
  std::uint64_t enqueue_ok = 0;
  std::uint64_t enqueue_drop = 0;
  std::uint64_t dequeue_ok = 0;
  std::uint64_t doorbell_batches = 0;
};

class SubmissionQueueMpsc {
 public:
  explicit SubmissionQueueMpsc(std::size_t capacity, PublishContract* contract);

  // Producer path: prepare + queue without taking global lock.
  std::optional<std::uint64_t> Enqueue(const SubmissionJob& job);

  // Consumer path: pop next entry in sequence order.
  std::optional<QueueEntry> Dequeue();

  // Consumer path: publish + doorbell in queue order.
  bool PublishOne(const QueueEntry& entry);

  // Consumer path: publish N entries then single doorbell ring for batch.
  std::size_t PublishBatch(std::size_t max_batch, std::uint32_t queue_id);

  // Completion correlation should be called exactly once per ticket.
  bool CorrelateCompletion(std::uint64_t ticket_id, AwaitResult* out);

  QueueStats Stats() const;

 private:
  struct Slot {
    std::atomic<bool> occupied{false};
    QueueEntry entry;
  };

  static std::uint64_t ReadMonotonicTsc();

  std::size_t capacity_;
  PublishContract* contract_;

  std::vector<Slot> ring_;
  std::atomic<std::uint64_t> tail_{0};
  std::uint64_t head_{0};

  std::atomic<std::uint64_t> enqueue_ok_{0};
  std::atomic<std::uint64_t> enqueue_drop_{0};
  std::atomic<std::uint64_t> dequeue_ok_{0};
  std::atomic<std::uint64_t> doorbell_batches_{0};

  std::atomic<std::uint64_t> next_local_sequence_{1};
  std::atomic<std::uint64_t> completion_seen_mask_seed_{1};
  std::vector<std::atomic<std::uint64_t>> completion_seen_;

  mutable std::mutex contract_mu_;
};

}  // namespace npu_gdn_layer
