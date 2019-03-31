#include "UpdateStressCommand.hpp"

namespace adm = axis::domain::materials;
namespace afm = axis::foundation::memory;

adm::UpdateStressCommand::UpdateStressCommand(void)
{
  // nothing to do here
}

adm::UpdateStressCommand::~UpdateStressCommand(void)
{
  // nothing to do here
}

void adm::UpdateStressCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  DoRun(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, gridDim, 
    blockDim, streamPtr, elementBlockSize_, reducedModelPtr_, currentTime_, 
    lastTimeIncrement_, nextTimeIncrement_);
}

void adm::UpdateStressCommand::SetParameters( uint64 elementBlockSize,
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  elementBlockSize_ = elementBlockSize;
  reducedModelPtr_ = reducedModelPtr;
  currentTime_ = currentTime;
  lastTimeIncrement_ = lastTimeIncrement;
  nextTimeIncrement_ = nextTimeIncrement;
}
