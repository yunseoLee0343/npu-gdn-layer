#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

namespace npu_gdn_layer {

enum class TraceEventType : std::uint16_t {
  kBufferClassification = 1,
  kBufferAcquire,
  kBufferRelease,
  kResidencyReuseHit,
  kResidencyReuseMiss,
  kSubmissionPrepared,
  kSubmissionPublished,
  kDoorbellRung,
  kCompletionObserved,
  kFirstTokenCallbackEntered,
  kDeferredCompletionWorkEntered,
};

struct TraceEvent {
  TraceEventType type = TraceEventType::kBufferClassification;
  std::uint64_t tsc = 0;

  std::uint64_t session_id = 0;
  std::uint64_t sequence_id = 0;
  std::uint64_t ticket_id = 0;
  std::uint64_t job_id = 0;
  std::uint64_t buffer_id = 0;

  std::uint32_t queue_id = 0;
  std::uint32_t token_index = 0;

  std::uint64_t value0 = 0;
  std::uint64_t value1 = 0;

  std::string_view note;
};

class TraceSink {
 public:
  virtual ~TraceSink() = default;
  virtual void Emit(const TraceEvent& event) = 0;
};

class DefaultLoggerSink final : public TraceSink {
 public:
  void Emit(const TraceEvent& event) override;
};

void SetTraceSink(std::shared_ptr<TraceSink> sink);
std::shared_ptr<TraceSink> GetTraceSink();
std::uint64_t TraceNowNs();
void TraceEmit(const TraceEvent& event);

}  // namespace npu_gdn_layer
