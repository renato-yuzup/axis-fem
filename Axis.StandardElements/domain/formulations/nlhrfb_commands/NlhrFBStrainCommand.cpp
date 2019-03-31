#include "stdafx.h"
#include "NlhrFBStrainCommand.hpp"
#include "gpu_nlhrfb_kernel.hpp"

namespace adfn = axis::domain::formulations::nlhrfb_commands;
namespace afm = axis::foundation::memory;

adfn::NlhrFBStrainCommand::NlhrFBStrainCommand(void)
{
  // nothing to do here
}

adfn::NlhrFBStrainCommand::~NlhrFBStrainCommand(void)
{
  // nothing to do here
}

void adfn::NlhrFBStrainCommand::DoRun( uint64 numThreadsToUse, uint64 startIndex, 
                                    void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
                                    const axis::Dimension3D& blockDim, void * streamPtr, uint64 elementBlockSize, 
                                    afm::RelativePointer& reducedModelPtr, real currentTime, 
                                    real lastTimeIncrement, real nextTimeIncrement )
{
  RunStrainCommandOnGPU(numThreadsToUse, startIndex, baseMemoryAddressOnGPU, 
    gridDim, blockDim, streamPtr, elementBlockSize, reducedModelPtr, 
    currentTime, lastTimeIncrement, nextTimeIncrement);
}
