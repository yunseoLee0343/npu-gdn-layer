#pragma once

#include <cstdint>

#include "npu_gdn_layer/submission.hpp"

namespace npu_gdn_layer {

enum class CompletionSource : std::uint8_t {
  kIrq,
  kPolling,
  kSynchronousFallback,
};

// Completion record for latency and correctness accounting.
// Ownership/lifetime:
// - Produced by completion path when retire is observed.
// - Immutable value object that can be persisted to trace/log sink.
struct CompletionEvent {
  SubmissionTicket ticket;

  CompletionSource source = CompletionSource::kIrq;
  bool success = false;
  std::uint32_t status_code = 0;

  std::uint64_t irq_tsc = 0;
  std::uint64_t retire_tsc = 0;

  // Runtime-derived metrics for TTFT and tail latency tracking.
  std::uint64_t queue_latency_ns = 0;
  std::uint64_t completion_latency_ns = 0;
  std::uint64_t end_to_end_latency_ns = 0;

  // Token-level observability for GDN stateful decode paths.
  std::uint32_t token_index = 0;
  bool contributes_to_ttft = false;

  static constexpr RuntimePath kPrimaryPath = RuntimePath::kCompletionLatency;
};

class CompletionObserver {
 public:
  virtual ~CompletionObserver() = default;

  virtual void OnCompletion(const CompletionEvent& event) = 0;
};

class CompletionQueue {
 public:
  virtual ~CompletionQueue() = default;

  // Poll one completion event if available; returns false when queue is empty.
  virtual bool TryPop(CompletionEvent* out_event) = 0;

  // Optional blocking wait with implementation-defined timeout policy.
  virtual bool WaitPop(CompletionEvent* out_event, std::uint32_t timeout_ms) = 0;
};

}  // namespace npu_gdn_layer
