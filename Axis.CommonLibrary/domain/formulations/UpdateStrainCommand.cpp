#include "UpdateStrainCommand.hpp"

namespace adf = axis::domain::formulations;
namespace afm = axis::foundation::memory;

adf::UpdateStrainCommand::UpdateStrainCommand(void)
{
  // nothing to do here
}

adf::UpdateStrainCommand::~UpdateStrainCommand(void)
{
  // nothing to do here
}

void adf::UpdateStrainCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  DoRun(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, gridDim, 
    blockDim, streamPtr, elementBlockSize_, reducedModelPtr_, currentTime_, 
    lastTimeIncrement_, nextTimeIncrement_);
}

void adf::UpdateStrainCommand::SetParameters( uint64 elementBlockSize,
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  elementBlockSize_ = elementBlockSize;
  reducedModelPtr_ = reducedModelPtr;
  currentTime_ = currentTime;
  lastTimeIncrement_ = lastTimeIncrement;
  nextTimeIncrement_ = nextTimeIncrement;
}
