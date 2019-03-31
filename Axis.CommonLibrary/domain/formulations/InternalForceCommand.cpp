#include "InternalForceCommand.hpp"

namespace adf = axis::domain::formulations;
namespace afm = axis::foundation::memory;

adf::InternalForceCommand::InternalForceCommand(void)
{
  // nothing to do here
}

adf::InternalForceCommand::~InternalForceCommand(void)
{
  // nothing to do here
}

void adf::InternalForceCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  DoRun(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, gridDim, 
    blockDim, streamPtr, elementBlockSize_, reducedModelPtr_, currentTime_, 
    lastTimeIncrement_, nextTimeIncrement_);
}

void adf::InternalForceCommand::SetParameters( uint64 elementBlockSize,
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  elementBlockSize_ = elementBlockSize;
  reducedModelPtr_ = reducedModelPtr;
  currentTime_ = currentTime;
  lastTimeIncrement_ = lastTimeIncrement;
  nextTimeIncrement_ = nextTimeIncrement;
}
