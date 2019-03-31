#include "LhrStrainCommand.hpp"
#include "gpu_lhr_kernel.hpp"

namespace adflhr = axis::domain::formulations::lhr_commands;
namespace afm = axis::foundation::memory;

adflhr::LhrStrainCommand::LhrStrainCommand(void)
{
  // nothing to do here
}

adflhr::LhrStrainCommand::~LhrStrainCommand(void)
{
  // nothing to do here
}

void adflhr::LhrStrainCommand::DoRun( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize,
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  RunStrainCommandOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, elementBlockSize, reducedModelPtr, 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}
