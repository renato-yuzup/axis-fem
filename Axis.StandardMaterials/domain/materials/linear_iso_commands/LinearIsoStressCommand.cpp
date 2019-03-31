#include "stdafx.h"
#include "LinearIsoStressCommand.hpp"
#include "linear_iso_elastic_kernel.hpp"

namespace admlie = axis::domain::materials::linear_iso_commands;
namespace afm = axis::foundation::memory;

admlie::LinearIsoStressCommand::LinearIsoStressCommand(void)
{
  // nothing to do here
}

admlie::LinearIsoStressCommand::~LinearIsoStressCommand(void)
{
  // nothing to do here
}

void admlie::LinearIsoStressCommand::DoRun( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  afm::RelativePointer& reducedModelPtr, real currentTime, 
  real lastTimeIncrement, real nextTimeIncrement )
{
  RunLinearIsoElasticOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, elementBlockSize, reducedModelPtr, 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}
