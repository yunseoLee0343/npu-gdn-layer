#include "npu_gdn_layer/completion_path.hpp"
#include "npu_gdn_layer/trace.hpp"

#include <chrono>
#include <unordered_map>

namespace npu_gdn_layer {

void CompletionPath::SetFirstTokenCallback(FirstTokenCallback cb) {
  std::lock_guard<std::mutex> lk(mu_);
  first_token_cb_ = std::move(cb);
}

void CompletionPath::SetDeferredBookkeeping(DeferredBookkeeping cb) {
  std::lock_guard<std::mutex> lk(mu_);
  deferred_cb_ = std::move(cb);
}

void CompletionPath::ObserveIrq(const SubmissionTicket& ticket, std::uint64_t irq_tsc) {
  std::lock_guard<std::mutex> lk(mu_);
  auto& state = by_ticket_[ticket.ticket_id];
  state.ticket = ticket;
  state.irq_observed_tsc = irq_tsc;
}

bool CompletionPath::PublishCompletionRecord(const SubmissionTicket& ticket,
                                             std::uint64_t record_tsc,
                                             bool success,
                                             std::uint32_t status_code,
                                             std::uint32_t token_index,
                                             bool first_token_critical) {
  CompletionEvent fast_event;

  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& state = by_ticket_[ticket.ticket_id];
    state.ticket = ticket;
    if (state.irq_observed_tsc == 0) {
      state.irq_observed_tsc = record_tsc;
    }

    state.record_available_tsc = record_tsc;
    state.wait_thread_wakeable = true;
    state.first_token_critical = first_token_critical;
    state.deferred_bookkeeping_pending = true;

    fast_event.ticket = ticket;
    fast_event.source = CompletionSource::kIrq;
    fast_event.success = success;
    fast_event.status_code = status_code;
    fast_event.irq_tsc = state.irq_observed_tsc;
    fast_event.retire_tsc = record_tsc;
    fast_event.token_index = token_index;
    fast_event.contributes_to_ttft = first_token_critical;

    state.latency = BuildLatency(state);
    fast_event.queue_latency_ns = state.latency.irq_to_record_ns;
    fast_event.completion_latency_ns = state.latency.record_to_wakeup_ns;
    fast_event.end_to_end_latency_ns = state.latency.ttft_ns;

    wakeable_queue_.push_back(fast_event);
    deferred_queue_.push_back(fast_event);
  }

  TraceEmit({TraceEventType::kCompletionObserved, TraceNowNs(), 0, ticket.sequence_id,
             ticket.ticket_id, 0, 0, static_cast<std::uint32_t>(ticket.queue_id), token_index,
             record_tsc, 0, "completion_observed"});
  cv_.notify_one();
  return true;
}

bool CompletionPath::WaitWakeable(CompletionEvent* out_event, std::uint32_t timeout_ms) {
  if (out_event == nullptr) {
    return false;
  }

  CompletionEvent event;
  std::uint64_t wake_tsc = 0;

  {
    std::unique_lock<std::mutex> lk(mu_);
    const bool ready = cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]() {
      return !wakeable_queue_.empty();
    });
    if (!ready) {
      return false;
    }

    event = wakeable_queue_.front();
    wakeable_queue_.pop_front();
    wake_tsc = ReadMonotonicTsc();

    auto it = by_ticket_.find(event.ticket.ticket_id);
    if (it != by_ticket_.end()) {
      CompletionTicketState& state = it->second;
      state.wakeup_tsc = wake_tsc;
      state.wait_thread_wakeable = false;
      state.latency = BuildLatency(state);
      event.completion_latency_ns = state.latency.record_to_wakeup_ns;
    }
  }

  const bool fast_path = event.contributes_to_ttft;
  if (fast_path) {
    FirstTokenCallback cb;
    {
      std::lock_guard<std::mutex> lk(mu_);
      cb = first_token_cb_;
    }
    if (cb) {
      TraceEmit({TraceEventType::kFirstTokenCallbackEntered, TraceNowNs(), 0,
                 event.ticket.sequence_id, event.ticket.ticket_id, 0, 0,
                 static_cast<std::uint32_t>(event.ticket.queue_id), event.token_index, 0, 0,
                 "first_token_cb"});
      cb(event);
    }

    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_ticket_.find(event.ticket.ticket_id);
    if (it != by_ticket_.end()) {
      CompletionTicketState& state = it->second;
      state.first_token_callback_tsc = ReadMonotonicTsc();
      state.latency = BuildLatency(state);
      event.end_to_end_latency_ns = state.latency.ttft_ns;
    }
  }

  *out_event = event;
  return true;
}

std::size_t CompletionPath::DrainDeferred(std::size_t max_events) {
  std::size_t drained = 0;

  while (drained < max_events) {
    CompletionEvent event;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (deferred_queue_.empty()) {
        break;
      }
      event = deferred_queue_.front();
      deferred_queue_.pop_front();

      auto it = by_ticket_.find(event.ticket.ticket_id);
      if (it != by_ticket_.end()) {
        it->second.deferred_bookkeeping_pending = false;
      }
    }

    DeferredBookkeeping cb;
    {
      std::lock_guard<std::mutex> lk(mu_);
      cb = deferred_cb_;
    }
    if (cb) {
      TraceEmit({TraceEventType::kDeferredCompletionWorkEntered, TraceNowNs(), 0,
                 event.ticket.sequence_id, event.ticket.ticket_id, 0, 0,
                 static_cast<std::uint32_t>(event.ticket.queue_id), event.token_index, 0, 0,
                 "deferred_work"});
      cb(event);
    }

    ++drained;
  }

  return drained;
}

std::uint64_t CompletionPath::ReadMonotonicTsc() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

CompletionLatencyBreakdown CompletionPath::BuildLatency(
    const CompletionTicketState& state) {
  CompletionLatencyBreakdown out;

  if (state.record_available_tsc >= state.irq_observed_tsc) {
    out.irq_to_record_ns = state.record_available_tsc - state.irq_observed_tsc;
  }
  if (state.wakeup_tsc >= state.record_available_tsc) {
    out.record_to_wakeup_ns = state.wakeup_tsc - state.record_available_tsc;
  }
  if (state.first_token_callback_tsc >= state.wakeup_tsc) {
    out.wakeup_to_first_token_ns = state.first_token_callback_tsc - state.wakeup_tsc;
  }
  if (state.first_token_callback_tsc >= state.irq_observed_tsc) {
    out.ttft_ns = state.first_token_callback_tsc - state.irq_observed_tsc;
  }

  return out;
}

}  // namespace npu_gdn_layer
