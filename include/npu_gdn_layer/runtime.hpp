#pragma once

#include <cstdint>

#include "npu_gdn_layer/buffer.hpp"
#include "npu_gdn_layer/completion.hpp"
#include "npu_gdn_layer/submission.hpp"

namespace npu_gdn_layer {

// Runtime-wide knobs to make memory/submission/completion policy explicit.
struct RuntimeConfig {
  bool enable_irq_completion = true;
  bool enable_poll_fallback = true;
  bool enable_host_observability_maps = false;

  std::uint32_t submission_queue_count = 1;
  std::uint32_t completion_queue_depth = 1024;

  // Latency budget used to tune doorbell batching vs low-latency publish.
  std::uint32_t target_ttft_us = 0;
  std::uint32_t target_p99_completion_us = 0;
};

// Top-level middle-layer interface for Qwen3-Next GDN stateful execution paths.
// Ownership/lifetime:
// - Runtime owns queue state, ticket namespace, and observer wiring.
// - Buffers and descriptor backing memory remain owned by caller/integration layer.
// - Sequence handles and tickets are runtime-issued value handles.
class Runtime {
 public:
  virtual ~Runtime() = default;

  // Session/sequence state handling.
  virtual SequenceStateHandle BindSequenceState(std::uint64_t session_id,
                                                std::uint64_t sequence_id) = 0;
  virtual void ReleaseSequenceState(const SequenceStateHandle& handle) = 0;

  // Memory Policy track entry points.
  virtual bool RegisterBuffer(const BufferObjectDesc& buffer) = 0;
  virtual bool UpdateStateRegion(const SequenceStateHandle& seq,
                                 const GdnStateRegion& region) = 0;
  virtual bool ApplyResidencyTransition(const SequenceStateHandle& seq,
                                        bool entering_decode_step) = 0;

  // Submission Correctness track entry points.
  virtual SubmissionTicket Submit(const SubmissionRequest& request) = 0;

  // Completion Latency track entry points.
  virtual bool PollCompletion(CompletionEvent* out_event) = 0;
  virtual void AttachObserver(CompletionObserver* observer) = 0;
};

}  // namespace npu_gdn_layer
