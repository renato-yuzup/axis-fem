#include "stdafx.h"
#include "BiLinearPlasticityStressCommand.hpp"
#include "bilinear_plasticity_kernel.hpp"

namespace admb = axis::domain::materials::bilinear_plasticity_commands;
namespace afm  = axis::foundation::memory;

admb::BiLinearPlasticityStressCommand::BiLinearPlasticityStressCommand(void)
{
  // nothing to do here
}

admb::BiLinearPlasticityStressCommand::~BiLinearPlasticityStressCommand(void)
{
  // nothing to do here
}

void admb::BiLinearPlasticityStressCommand::DoRun( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, uint64 elementBlockSize, 
  afm::RelativePointer& reducedModelPtr, real, real, real nextTimeIncrement )
{
  RunBiLinearPlasticityOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, elementBlockSize, reducedModelPtr, 
    nextTimeIncrement);
}
