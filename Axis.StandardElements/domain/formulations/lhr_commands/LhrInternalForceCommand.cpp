#include "LhrInternalForceCommand.hpp"
#include "gpu_lhr_kernel.hpp"

namespace adflhr = axis::domain::formulations::lhr_commands;
namespace afm = axis::foundation::memory;

adflhr::LhrInternalForceCommand::LhrInternalForceCommand(void)
{
  // nothing to do here
}

adflhr::LhrInternalForceCommand::~LhrInternalForceCommand(void)
{
  // nothing to do here
}

void adflhr::LhrInternalForceCommand::DoRun( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  RunInternalForceCommandOnGPU(numThreadsToUse, startIndex, 
    baseMemoryAddressOnGPU, gridDim, blockDim, streamPtr, elementBlockSize, 
    reducedModelPtr, currentTime, lastTimeIncrement, nextTimeIncrement);
}
