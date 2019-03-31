#include "stdafx.h"
#include "NeoHookeanStressCommand.hpp"
#include "neohookean_kernel.hpp"

namespace admn = axis::domain::materials::neohookean_commands;
namespace afm = axis::foundation::memory;

admn::NeoHookeanStressCommand::NeoHookeanStressCommand(void)
{
  // nothing to do here
}

admn::NeoHookeanStressCommand::~NeoHookeanStressCommand(void)
{
  // nothing to do here
}

void admn::NeoHookeanStressCommand::DoRun( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  RunNeoHookeanOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, elementBlockSize, reducedModelPtr);
}
