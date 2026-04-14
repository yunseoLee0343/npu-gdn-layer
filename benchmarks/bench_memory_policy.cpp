#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "npu_gdn_layer/residency_manager.hpp"

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
  ResidencyManager mgr;

  constexpr int kIters = 20000;
  std::mt19937_64 rng(123);
  std::uniform_int_distribution<int> cls_dist(0, 3);
  std::uniform_int_distribution<int> key_dist(1, 128);

  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::vector<std::uint64_t> acquire_ns;

  for (int i = 0; i < kIters; ++i) {
    BufferAcquireRequest req;
    const int cls = cls_dist(rng);
    req.bytes = 1024 + (i % 16) * 256;

    switch (cls) {
      case 0:
        req.buffer_class = BufferClass::WEIGHT_PERSISTENT_RO;
        req.logical_key = key_dist(rng);
        req.requested_mutability = Mutability::kReadOnly;
        break;
      case 1:
        req.buffer_class = BufferClass::STATE_PERSISTENT_RW;
        req.logical_key = key_dist(rng);
        req.requested_mutability = Mutability::kReadWriteHot;
        break;
      case 2:
        req.buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
        req.logical_key = 0;
        break;
      default:
        req.buffer_class = BufferClass::CMD_DESC_LOWLAT;
        req.logical_key = 0;
        break;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto lease = mgr.Acquire(req);
    const auto t1 = std::chrono::steady_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    acquire_ns.push_back(static_cast<std::uint64_t>(ns));

    if (lease.has_value()) {
      // Heuristic for reuse estimation: generation > 1 means previously reused.
      if (lease->generation > 1) {
        ++hits;
      } else {
        ++misses;
      }
      mgr.Release(lease->buffer_id);
    }
  }

  const double hit_rate = (hits + misses) ? (100.0 * hits / (hits + misses)) : 0.0;
  std::cout << "bench_memory_policy\n";
  std::cout << "  iterations: " << kIters << "\n";
  std::cout << "  buffer_pool_hit_rate_percent: " << hit_rate << "\n";
  std::cout << "  acquire_p50_ns: " << Percentile(acquire_ns, 0.50) << "\n";
  std::cout << "  acquire_p95_ns: " << Percentile(acquire_ns, 0.95) << "\n";
  std::cout << "  acquire_p99_ns: " << Percentile(acquire_ns, 0.99) << "\n";
  return 0;
}
