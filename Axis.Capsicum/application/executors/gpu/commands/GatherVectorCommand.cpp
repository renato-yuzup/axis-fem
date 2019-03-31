#include "GatherVectorCommand.hpp"
#include "kernels/gather_vector_kernel.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace afm   = axis::foundation::memory;

aaegc::GatherVectorCommand::GatherVectorCommand( afm::RelativePointer& modelPtr, 
  afm::RelativePointer& vectorPtr ) : modelPtr_(modelPtr), vectorPtr_(vectorPtr)
{
  // nothing to do here
}

aaegc::GatherVectorCommand::~GatherVectorCommand(void)
{
  // nothing to do here
}

void aaegc::GatherVectorCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  kernels::RunGatherVectorOnGPU(numThreadsToUse, startIndex, gridDim, blockDim, 
    streamPtr, vectorPtr_, modelPtr_);
}
