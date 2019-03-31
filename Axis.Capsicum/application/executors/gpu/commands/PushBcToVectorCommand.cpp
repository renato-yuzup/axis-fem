#include "PushBcToVectorCommand.hpp"
#include "kernels/push_bc_to_vector_kernel.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace afm   = axis::foundation::memory;

aaegc::PushBcToVectorCommand::PushBcToVectorCommand( 
  afm::RelativePointer& vectorPtr, afm::RelativePointer& vectorMaskPtr, 
  bool ignoreMask, int bcBlockSize) :
vectorPtr_(vectorPtr), vectorMaskPtr_(vectorMaskPtr)
{
  ignoreMask_ = ignoreMask;
  bcBlockSize_ = bcBlockSize;
}

aaegc::PushBcToVectorCommand::~PushBcToVectorCommand( void )
{
  // nothing to do here
}

void aaegc::PushBcToVectorCommand::Run( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr )
{
  kernels::RunPushBcVectorOnGPU(numThreadsToUse, startIndex, 
    baseMemoryAddressOnGPU, gridDim, blockDim, streamPtr, vectorPtr_, 
    vectorMaskPtr_, ignoreMask_, bcBlockSize_);
}
