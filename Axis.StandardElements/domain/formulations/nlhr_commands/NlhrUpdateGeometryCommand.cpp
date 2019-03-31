#include "stdafx.h"
#include "NlhrUpdateGeometryCommand.hpp"
#include "gpu_nlhr_kernel.hpp"

namespace adfn = axis::domain::formulations::nlhr_commands;
namespace afm = axis::foundation::memory;

adfn::NlhrUpdateGeometryCommand::NlhrUpdateGeometryCommand(void)
{
  // nothing to do here
}


adfn::NlhrUpdateGeometryCommand::~NlhrUpdateGeometryCommand(void)
{
  // nothing to do here
}

void adfn::NlhrUpdateGeometryCommand::DoRun( uint64 numThreadsToUse, uint64 
  startIndex, void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize, 
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  RunUpdateGeometryCommandOnGPU(numThreadsToUse, startIndex, 
    baseMemoryAddressOnGPU, gridDim, blockDim, streamPtr, elementBlockSize, 
    reducedModelPtr, currentTime, lastTimeIncrement, nextTimeIncrement);
}
