#include "UpdateReactionForceCommand.hpp"
#include "UpdateReactionForceCommand_Kernel.hpp"

namespace adal = axis::domain::algorithms;
namespace afm = axis::foundation::memory;

adal::UpdateReactionForceCommand::UpdateReactionForceCommand(void)
{
  // nothing to do here
}

adal::UpdateReactionForceCommand::~UpdateReactionForceCommand(void)
{
  // nothing to do here
}

void adal::UpdateReactionForceCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  UpdateReactionForceOnGPU(reaction_, external_, internal_, numThreadsToUse, startIndex, 
    baseMemoryAddressOnGPU, gridDim, blockDim, streamPtr);
}

void adal::UpdateReactionForceCommand::SetDynamicVectors( const afm::RelativePointer& reactionForcePtr, 
  const afm::RelativePointer& externalLoadPtr, const afm::RelativePointer& internalForcePtr )
{
  reaction_ = reactionForcePtr;
  external_ = externalLoadPtr;
  internal_ = internalForcePtr;
}
