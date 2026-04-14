#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "npu_gdn_layer/completion_path.hpp"
#include "npu_gdn_layer/trace.hpp"

namespace {

using namespace npu_gdn_layer;

std::uint64_t Percentile(std::vector<std::uint64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const std::size_t idx = static_cast<std::size_t>(p * (values.size() - 1));
  return values[idx];
}

}  // namespace

int main() {
  CompletionPath path;
  std::atomic<std::uint64_t> first_token_cb_ns{0};
  std::atomic<std::uint64_t> deferred_cb_ns{0};

  path.SetFirstTokenCallback([&](const CompletionEvent&) {
    const auto t0 = std::chrono::steady_clock::now();
    // Simulated critical callback work.
    for (volatile int i = 0; i < 100; ++i) {
    }
    const auto t1 = std::chrono::steady_clock::now();
    first_token_cb_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
  });

  path.SetDeferredBookkeeping([&](const CompletionEvent&) {
    const auto t0 = std::chrono::steady_clock::now();
    // Simulated deferred cleanup/stats work.
    for (volatile int i = 0; i < 1000; ++i) {
    }
    const auto t1 = std::chrono::steady_clock::now();
    deferred_cb_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
  });

  constexpr int kIters = 10000;
  std::vector<std::uint64_t> ttft_path_ns;
  std::vector<std::uint64_t> callback_ns;

  for (int i = 0; i < kIters; ++i) {
    SubmissionTicket t;
    t.ticket_id = static_cast<std::uint64_t>(i + 1);
    t.sequence_id = static_cast<std::uint64_t>(i + 1);
    t.queue_id = 0;

    const auto irq = TraceNowNs();
    path.ObserveIrq(t, irq);
    path.PublishCompletionRecord(t, TraceNowNs(), true, 0, i % 8, true);

    CompletionEvent out;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = path.WaitWakeable(&out, 10);
    const auto t1 = std::chrono::steady_clock::now();
    if (!ok) continue;

    ttft_path_ns.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    callback_ns.push_back(out.end_to_end_latency_ns);
    path.DrainDeferred(1);
  }

  const auto avg_fast_cb = first_token_cb_ns.load(std::memory_order_relaxed) / kIters;
  const auto avg_deferred_cb = deferred_cb_ns.load(std::memory_order_relaxed) / kIters;

  std::cout << "bench_completion_path\n";
  std::cout << "  avg_fast_path_callback_ns: " << avg_fast_cb << "\n";
  std::cout << "  avg_deferred_callback_ns: " << avg_deferred_cb << "\n";
  std::cout << "  completion_callback_p50_ns: " << Percentile(ttft_path_ns, 0.50) << "\n";
  std::cout << "  completion_callback_p95_ns: " << Percentile(ttft_path_ns, 0.95) << "\n";
  std::cout << "  completion_callback_p99_ns: " << Percentile(ttft_path_ns, 0.99) << "\n";
  std::cout << "  simulated_ttft_critical_p50_ns: " << Percentile(callback_ns, 0.50) << "\n";
  std::cout << "  simulated_ttft_critical_p95_ns: " << Percentile(callback_ns, 0.95) << "\n";
  std::cout << "  simulated_ttft_critical_p99_ns: " << Percentile(callback_ns, 0.99) << "\n";
  return 0;
}
