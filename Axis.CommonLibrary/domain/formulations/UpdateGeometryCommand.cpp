#include "UpdateGeometryCommand.hpp"

namespace adf = axis::domain::formulations;
namespace afm = axis::foundation::memory;

adf::UpdateGeometryCommand::UpdateGeometryCommand( void )
{
  // nothing to do here
}

adf::UpdateGeometryCommand::~UpdateGeometryCommand( void )
{
  // nothing to do here
}

void adf::UpdateGeometryCommand::SetParameters( uint64 elementBlockSize, 
  const afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  elementBlockSize_ = elementBlockSize;
  reducedModelPtr_ = reducedModelPtr;
  currentTime_ = currentTime;
  lastTimeIncrement_ = lastTimeIncrement;
  nextTimeIncrement_ = nextTimeIncrement;
}

void adf::UpdateGeometryCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  DoRun(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, gridDim, 
    blockDim, streamPtr, elementBlockSize_, reducedModelPtr_, currentTime_, 
    lastTimeIncrement_, nextTimeIncrement_);
}
