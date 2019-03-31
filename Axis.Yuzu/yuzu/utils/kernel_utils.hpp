#pragma once
#include "yuzu/common/gpu.hpp"

namespace axis { namespace yuzu {

GPU_ONLY uint64 GetThreadIndex(const dim3& gridDim, const dim3& blockIdx, 
  const dim3& blockDim, const dim3& threadIdx, uint64 startIndex);
GPU_ONLY uint64 GetBaseThreadIndex(const dim3& gridDim, const dim3& blockIdx, 
  const dim3& blockDim, const dim3& threadIdx);

GPU_ONLY bool IsActiveThread(uint64 threadIndex, uint64 maxThreadNum);

} } // namespace axis::yuzu
