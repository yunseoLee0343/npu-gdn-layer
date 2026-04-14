#include <cassert>
#include <cstdint>
#include <iostream>

#include "npu_gdn_layer/residency_manager.hpp"

namespace {

using namespace npu_gdn_layer;

void TestPersistentStateReuseAcrossSteps() {
  ResidencyManager mgr;

  BufferAcquireRequest req;
  req.buffer_class = BufferClass::STATE_PERSISTENT_RW;
  req.bytes = 4096;
  req.logical_key = 101;
  req.requested_mutability = Mutability::kReadWriteHot;

  auto lease1 = mgr.Acquire(req);
  assert(lease1.has_value());
  assert(lease1->buffer_class == BufferClass::STATE_PERSISTENT_RW);
  const std::uint64_t id = lease1->buffer_id;
  assert(mgr.Release(id));

  auto lease2 = mgr.Acquire(req);
  assert(lease2.has_value());
  assert(lease2->buffer_id == id);
  assert(lease2->generation > lease1->generation);
}

void TestEphemeralActivationReuseFromPool() {
  ResidencyManager mgr;

  BufferAcquireRequest req;
  req.buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  req.bytes = 8192;

  auto lease1 = mgr.Acquire(req);
  assert(lease1.has_value());
  const std::uint64_t first_id = lease1->buffer_id;
  assert(mgr.Release(first_id));

  auto lease2 = mgr.Acquire(req);
  assert(lease2.has_value());
  assert(lease2->buffer_id == first_id);
}

void TestCommandBufferClassIsSeparate() {
  ResidencyManager mgr;

  BufferAcquireRequest cmd_req;
  cmd_req.buffer_class = BufferClass::CMD_DESC_LOWLAT;
  cmd_req.bytes = 1024;

  BufferAcquireRequest act_req;
  act_req.buffer_class = BufferClass::ACTIVATION_EPHEMERAL_RW;
  act_req.bytes = 1024;

  auto cmd = mgr.Acquire(cmd_req);
  auto act = mgr.Acquire(act_req);
  assert(cmd.has_value());
  assert(act.has_value());
  assert(cmd->buffer_class == BufferClass::CMD_DESC_LOWLAT);
  assert(act->buffer_class == BufferClass::ACTIVATION_EPHEMERAL_RW);
  assert(cmd->buffer_id != act->buffer_id);
}

void TestWeightImmutableGuard() {
  ResidencyManager mgr;

  BufferAcquireRequest bad_req;
  bad_req.buffer_class = BufferClass::WEIGHT_PERSISTENT_RO;
  bad_req.bytes = 16384;
  bad_req.logical_key = 7;
  bad_req.requested_mutability = Mutability::kReadWriteHot;

  auto rejected = mgr.Acquire(bad_req);
  assert(!rejected.has_value());

  BufferAcquireRequest ok_req = bad_req;
  ok_req.requested_mutability = Mutability::kReadOnly;
  auto accepted = mgr.Acquire(ok_req);
  assert(accepted.has_value());
  assert(accepted->mutability == Mutability::kReadOnly);
}

}  // namespace

int main() {
  TestPersistentStateReuseAcrossSteps();
  TestEphemeralActivationReuseFromPool();
  TestCommandBufferClassIsSeparate();
  TestWeightImmutableGuard();

  std::cout << "residency_manager tests passed\n";
  return 0;
}
