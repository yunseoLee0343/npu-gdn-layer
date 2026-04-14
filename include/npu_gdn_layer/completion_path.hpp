#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "npu_gdn_layer/completion.hpp"

namespace npu_gdn_layer {

struct CompletionLatencyBreakdown {
  std::uint64_t irq_to_record_ns = 0;
  std::uint64_t record_to_wakeup_ns = 0;
  std::uint64_t wakeup_to_first_token_ns = 0;
  std::uint64_t ttft_ns = 0;
};

struct CompletionTicketState {
  SubmissionTicket ticket;

  std::uint64_t irq_observed_tsc = 0;
  std::uint64_t record_available_tsc = 0;
  std::uint64_t wakeup_tsc = 0;
  std::uint64_t first_token_callback_tsc = 0;

  bool wait_thread_wakeable = false;
  bool first_token_critical = false;
  bool deferred_bookkeeping_pending = false;

  CompletionLatencyBreakdown latency;
};

// Completion path for TTFT-critical stateful inference.
// Fast path: IRQ -> completion record -> wake waiters -> first-token callback.
// Slow path: deferred bookkeeping (stats/cleanup/background correlation).
class CompletionPath {
 public:
  using FirstTokenCallback = std::function<void(const CompletionEvent&)>;
  using DeferredBookkeeping = std::function<void(const CompletionEvent&)>;

  CompletionPath() = default;

  void SetFirstTokenCallback(FirstTokenCallback cb);
  void SetDeferredBookkeeping(DeferredBookkeeping cb);

  // Stage 1: IRQ observed (minimal work, timestamp only).
  void ObserveIrq(const SubmissionTicket& ticket, std::uint64_t irq_tsc);

  // Stage 2: completion record available; marks waiters wakeable.
  bool PublishCompletionRecord(const SubmissionTicket& ticket,
                               std::uint64_t record_tsc,
                               bool success,
                               std::uint32_t status_code,
                               std::uint32_t token_index,
                               bool first_token_critical);

  // Stage 3: waiting runtime thread consumes completion readiness.
  bool WaitWakeable(CompletionEvent* out_event, std::uint32_t timeout_ms);

  // Stage 4: execute deferred non-critical bookkeeping.
  std::size_t DrainDeferred(std::size_t max_events);

 private:
  static std::uint64_t ReadMonotonicTsc();
  static CompletionLatencyBreakdown BuildLatency(const CompletionTicketState& state);

  std::mutex mu_;
  std::condition_variable cv_;

  std::deque<CompletionEvent> wakeable_queue_;
  std::deque<CompletionEvent> deferred_queue_;
  std::unordered_map<std::uint64_t, CompletionTicketState> by_ticket_;

  FirstTokenCallback first_token_cb_;
  DeferredBookkeeping deferred_cb_;
};

}  // namespace npu_gdn_layer
