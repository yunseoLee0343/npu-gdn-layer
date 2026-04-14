#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

#include "npu_gdn_layer/submission_queue.hpp"

namespace {

using namespace npu_gdn_layer;

SubmissionJob MakeJob(std::uint64_t job_id, std::uint32_t queue_id) {
  SubmissionJob job;
  job.job_id = job_id;
  job.queue_id = queue_id;
  job.descriptor_bo_id = 1;
  job.descriptor_offset = job_id * 64;
  job.descriptor_bytes = 64;
  job.metadata_bo_id = 2;
  job.metadata_offset = job_id * 32;
  job.metadata_bytes = 32;
  job.dynamic_state.buffer_id = 3;
  job.dynamic_state.byte_offset = 0;
  job.dynamic_state.byte_length = 128;
  job.host_descriptor_write_complete = true;
  job.host_metadata_write_complete = true;
  job.host_state_write_complete = true;
  return job;
}

void TestEnqueueDequeueCorrectness() {
  PublishContract contract;
  SubmissionQueueMpsc q(64, &contract);

  auto t1 = q.Enqueue(MakeJob(1, 0));
  auto t2 = q.Enqueue(MakeJob(2, 0));
  assert(t1.has_value());
  assert(t2.has_value());

  auto e1 = q.Dequeue();
  auto e2 = q.Dequeue();
  assert(e1.has_value());
  assert(e2.has_value());
  assert(e1->job_id == 1);
  assert(e2->job_id == 2);
  assert(e1->enqueue_tsc > 0);
  assert(e1->dequeue_tsc >= e1->enqueue_tsc);
}

void TestMonotonicSequenceAssignment() {
  PublishContract contract;
  SubmissionQueueMpsc q(64, &contract);

  q.Enqueue(MakeJob(10, 0));
  q.Enqueue(MakeJob(11, 0));
  q.Enqueue(MakeJob(12, 0));

  auto a = q.Dequeue();
  auto b = q.Dequeue();
  auto c = q.Dequeue();
  assert(a.has_value() && b.has_value() && c.has_value());
  assert(a->ticket.sequence_id < b->ticket.sequence_id);
  assert(b->ticket.sequence_id < c->ticket.sequence_id);
}

void TestNoDuplicateCompletionCorrelation() {
  PublishContract contract;
  SubmissionQueueMpsc q(64, &contract);

  auto ticket_id = q.Enqueue(MakeJob(100, 2));
  assert(ticket_id.has_value());

  auto e = q.Dequeue();
  assert(e.has_value());
  assert(q.PublishOne(*e));
  assert(contract.mark_completed(e->ticket.ticket_id));

  AwaitResult first;
  bool first_ok = q.CorrelateCompletion(e->ticket.ticket_id, &first);
  assert(first_ok);

  AwaitResult second;
  bool second_ok = q.CorrelateCompletion(e->ticket.ticket_id, &second);
  assert(!second_ok);
}

void TestBasicConcurrency() {
  PublishContract contract;
  SubmissionQueueMpsc q(2048, &contract);

  constexpr int kThreads = 4;
  constexpr int kPerThread = 100;

  std::vector<std::thread> producers;
  std::atomic<int> produced{0};

  for (int t = 0; t < kThreads; ++t) {
    producers.emplace_back([t, &q, &produced]() {
      for (int i = 0; i < kPerThread; ++i) {
        const std::uint64_t job_id = static_cast<std::uint64_t>(t * 1000 + i);
        if (q.Enqueue(MakeJob(job_id, 1)).has_value()) {
          produced.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto& th : producers) {
    th.join();
  }

  int consumed = 0;
  std::vector<std::uint64_t> seqs;
  while (true) {
    auto e = q.Dequeue();
    if (!e.has_value()) {
      break;
    }
    ++consumed;
    seqs.push_back(e->ticket.sequence_id);
  }

  assert(consumed == produced.load(std::memory_order_relaxed));
  assert(std::is_sorted(seqs.begin(), seqs.end()));
}

}  // namespace

int main() {
  TestEnqueueDequeueCorrectness();
  TestMonotonicSequenceAssignment();
  TestNoDuplicateCompletionCorrelation();
  TestBasicConcurrency();
  return 0;
}
