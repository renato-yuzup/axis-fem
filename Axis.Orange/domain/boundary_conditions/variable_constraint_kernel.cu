#include "variable_constraint_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/domain/boundary_conditions/BoundaryConditionData.hpp"
#include "yuzu/common/gpu.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "stdio.h"

#define DOF_STATUS_FREE                         0

namespace ay = axis::yuzu;
namespace ayfm = axis::yuzu::foundation::memory;
namespace aydbc = axis::yuzu::domain::boundary_conditions;

struct VariableConstraintData
{
  const real *CurveDataPtr;
  real ScalingFactor;
  real ReleaseTime;
};

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  UpdateConstraintOnGPUKernel(uint64 numThreads, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, real time, 
  axis::yuzu::foundation::memory::RelativePointer vectorMaskPtr)
{
  uint64 index = ay::GetThreadIndex(gridDim, blockIdx, blockDim, 
    threadIdx, startIndex);
  if (!ay::IsActiveThread(index, numThreads)) return;
  aydbc::BoundaryConditionData bcData(baseMemoryAddressOnGPU, index, sizeof(VariableConstraintData));
  real *bucket = bcData.GetOutputBucket();
  VariableConstraintData *data = 
    (VariableConstraintData *)bcData.GetCustomData();
  real curveCurVal = *data->CurveDataPtr;
  real scaleFactor = data->ScalingFactor;
  real releaseTime = data->ReleaseTime;
  uint64 dofId = bcData.GetDofId();
  *bucket = curveCurVal * scaleFactor;
  if (releaseTime >= 0 && time > releaseTime)
  {
    char *vectorMask = axis::yabsptr<char>(vectorMaskPtr);
    vectorMask[dofId] = DOF_STATUS_FREE;
  }
}

void axis::domain::boundary_conditions::UpdateConstraintOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, real time, 
  axis::foundation::memory::RelativePointer vectorMaskPtr )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  UpdateConstraintOnGPUKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, time,
    reinterpret_cast<ayfm::RelativePointer&>(vectorMaskPtr));
}
