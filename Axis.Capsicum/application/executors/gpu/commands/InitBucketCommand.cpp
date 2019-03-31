#include "InitBucketCommand.hpp"
#include "kernels/init_element_bucket_kernel.hpp"

namespace aaegc = axis::application::executors::gpu::commands;
namespace afm = axis::foundation::memory;

aaegc::InitBucketCommand::InitBucketCommand(afm::RelativePointer& modelPtr, 
  size_type elementBlockSize ) : modelPtr_(modelPtr)
{
  elementBlockSize_ = elementBlockSize;
}

aaegc::InitBucketCommand::~InitBucketCommand(void)
{
  // nothing to do here
}

void aaegc::InitBucketCommand::Run( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  kernels::RunInitElementBucketOnGPU(numThreadsToUse, startIndex, 
    baseMemoryAddressOnGPU, gridDim, blockDim, streamPtr, modelPtr_, 
    elementBlockSize_);
}
