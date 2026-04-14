#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "npu_gdn_layer/submission_queue.hpp"

namespace {

using namespace npu_gdn_layer;

SubmissionJob MakeJob(std::uint64_t job_id) {
  SubmissionJob j;
  j.job_id = job_id;
  j.queue_id = 1;
  j.descriptor_bo_id = 1;
  j.descriptor_offset = job_id * 64;
  j.descriptor_bytes = 64;
  j.metadata_bo_id = 2;
  j.metadata_offset = job_id * 32;
  j.metadata_bytes = 32;
  j.dynamic_state.buffer_id = 3;
  j.dynamic_state.byte_length = 128;
  j.host_descriptor_write_complete = true;
  j.host_metadata_write_complete = true;
  j.host_state_write_complete = true;
  return j;
}

std::uint64_t Percentile(std::vector<std::uint64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const std::size_t idx = static_cast<std::size_t>(p * (values.size() - 1));
  return values[idx];
}

}  // namespace

int main() {
  PublishContract contract;
  SubmissionQueueMpsc queue(8192, &contract);

  constexpr int kThreads = 4;
  constexpr int kPerThread = 5000;

  std::vector<std::thread> producers;
  std::vector<std::uint64_t> enqueue_ns;
  std::mutex lat_mu;

  for (int t = 0; t < kThreads; ++t) {
    producers.emplace_back([&, t]() {
      for (int i = 0; i < kPerThread; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        queue.Enqueue(MakeJob(static_cast<std::uint64_t>(t * 100000 + i)));
        const auto t1 = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        std::lock_guard<std::mutex> lk(lat_mu);
        enqueue_ns.push_back(static_cast<std::uint64_t>(ns));
      }
    });
  }
  for (auto& th : producers) th.join();

  std::vector<std::uint64_t> publish_ns;
  while (true) {
    auto e = queue.Dequeue();
    if (!e.has_value()) break;

    const auto t0 = std::chrono::steady_clock::now();
    queue.PublishOne(*e);
    const auto t1 = std::chrono::steady_clock::now();
    publish_ns.push_back(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }

  const QueueStats s = queue.Stats();
  std::cout << "bench_submission_queue\n";
  std::cout << "  enqueue_ok: " << s.enqueue_ok << "\n";
  std::cout << "  enqueue_drop: " << s.enqueue_drop << "\n";
  std::cout << "  queue_contention_threads: " << kThreads << "\n";
  std::cout << "  enqueue_p50_ns: " << Percentile(enqueue_ns, 0.50) << "\n";
  std::cout << "  enqueue_p95_ns: " << Percentile(enqueue_ns, 0.95) << "\n";
  std::cout << "  enqueue_p99_ns: " << Percentile(enqueue_ns, 0.99) << "\n";
  std::cout << "  publish_p50_ns: " << Percentile(publish_ns, 0.50) << "\n";
  std::cout << "  publish_p95_ns: " << Percentile(publish_ns, 0.95) << "\n";
  std::cout << "  publish_p99_ns: " << Percentile(publish_ns, 0.99) << "\n";
  return 0;
}
