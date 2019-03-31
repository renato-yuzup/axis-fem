#include "stdafx.h"
#include "NlhrFBInternalForceCommand.hpp"
#include "gpu_nlhrfb_kernel.hpp"

namespace adfn = axis::domain::formulations::nlhrfb_commands;
namespace afm = axis::foundation::memory;

adfn::NlhrFBInternalForceCommand::NlhrFBInternalForceCommand(void)
{
	// nothing to do here
}

adfn::NlhrFBInternalForceCommand::~NlhrFBInternalForceCommand(void)
{
	// nothing to do here
}

void adfn::NlhrFBInternalForceCommand::DoRun( uint64 numThreadsToUse, 
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
