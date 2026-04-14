#include "npu_gdn_layer/trace.hpp"

#include <chrono>
#include <cstdio>
#include <mutex>

namespace npu_gdn_layer {

namespace {
std::mutex g_trace_mu;
std::shared_ptr<TraceSink> g_sink = std::make_shared<DefaultLoggerSink>();
}

void DefaultLoggerSink::Emit(const TraceEvent& event) {
  std::fprintf(stderr,
               "trace type=%u tsc=%llu sess=%llu seq=%llu ticket=%llu job=%llu "
               "buf=%llu q=%u tok=%u v0=%llu v1=%llu note=%.*s\n",
               static_cast<unsigned>(event.type),
               static_cast<unsigned long long>(event.tsc),
               static_cast<unsigned long long>(event.session_id),
               static_cast<unsigned long long>(event.sequence_id),
               static_cast<unsigned long long>(event.ticket_id),
               static_cast<unsigned long long>(event.job_id),
               static_cast<unsigned long long>(event.buffer_id),
               static_cast<unsigned>(event.queue_id),
               static_cast<unsigned>(event.token_index),
               static_cast<unsigned long long>(event.value0),
               static_cast<unsigned long long>(event.value1),
               static_cast<int>(event.note.size()), event.note.data());
}

void SetTraceSink(std::shared_ptr<TraceSink> sink) {
  std::lock_guard<std::mutex> lk(g_trace_mu);
  g_sink = sink ? std::move(sink) : std::make_shared<DefaultLoggerSink>();
}

std::shared_ptr<TraceSink> GetTraceSink() {
  std::lock_guard<std::mutex> lk(g_trace_mu);
  return g_sink;
}

std::uint64_t TraceNowNs() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void TraceEmit(const TraceEvent& event) {
  std::shared_ptr<TraceSink> sink;
  {
    std::lock_guard<std::mutex> lk(g_trace_mu);
    sink = g_sink;
  }
  if (sink) {
    sink->Emit(event);
  }
}

}  // namespace npu_gdn_layer
