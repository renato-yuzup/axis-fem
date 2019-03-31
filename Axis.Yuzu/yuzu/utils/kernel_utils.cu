#include "kernel_utils.hpp"

namespace ay = axis::yuzu;

GPU_ONLY uint64 ay::GetThreadIndex( const dim3& gridDim, const dim3& blockIdx, 
  const dim3& blockDim, const dim3& threadIdx, uint64 startIndex )
{
  return startIndex + GetBaseThreadIndex(gridDim, blockIdx, blockDim, threadIdx);
}

GPU_ONLY uint64 ay::GetBaseThreadIndex( const dim3& gridDim, 
  const dim3& blockIdx, const dim3& blockDim, const dim3& threadIdx )
{
  uint64 blockItemCount = blockDim.x * blockDim.y;
  uint64 outerIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockItemCount;
  uint64 innerIndex = threadIdx.y * blockDim.x + threadIdx.x;
  uint64 index = outerIndex + innerIndex;
  return index;
}

GPU_ONLY bool ay::IsActiveThread( uint64 threadIndex, uint64 maxThreadNum )
{
  return threadIndex < maxThreadNum;
}

