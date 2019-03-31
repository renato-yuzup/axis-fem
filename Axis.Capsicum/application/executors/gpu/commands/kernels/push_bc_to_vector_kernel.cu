#include "push_bc_to_vector_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/domain/boundary_conditions/BoundaryConditionData.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"

#define DOF_STATUS_FREE   0

namespace afm   = axis::foundation::memory;
namespace ay    = axis::yuzu;
namespace aydbc = axis::yuzu::domain::boundary_conditions;
namespace ayfb  = axis::yuzu::foundation::blas;
namespace ayfm  = axis::yuzu::foundation::memory;

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
  RunPushBcVectorKernel(uint64 numThreadsToUse, uint64 startIndex, 
   void *baseMemoryAddressOnGPU, ayfm::RelativePointer vectorPtr, 
   ayfm::RelativePointer vectorMaskPtr, bool ignoreMask,
   int bcBlockSize)
{
  using axis::yabsref;
  using axis::yabsptr;
  uint64 threadIndex = 
    ay::GetThreadIndex(gridDim, blockIdx, blockDim, threadIdx, startIndex);
  if (!ay::IsActiveThread(threadIndex, numThreadsToUse)) return;
  ayfb::ColumnVector& vector = yabsref<ayfb::ColumnVector>(vectorPtr);
  char *vectorMask = yabsptr<char>(vectorMaskPtr);
  aydbc::BoundaryConditionData bc(baseMemoryAddressOnGPU, threadIndex, bcBlockSize);
  uint64 dofId = bc.GetDofId();
  real *bcVal = bc.GetOutputBucket();
  if (ignoreMask || vectorMask[dofId] != DOF_STATUS_FREE)
  {
    vector(dofId) = *bcVal;
  }
}

void axis::application::executors::gpu::commands::kernels::RunPushBcVectorOnGPU( 
  uint64 numThreadsToUse, uint64 startIndex,  void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, afm::RelativePointer& vectorPtr, afm::RelativePointer& vectorMaskPtr, 
  bool ignoreMask, int bcBlockSize )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  RunPushBcVectorKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU,
    reinterpret_cast<ayfm::RelativePointer&>(vectorPtr), 
    reinterpret_cast<ayfm::RelativePointer&>(vectorMaskPtr), ignoreMask, 
    bcBlockSize);
}
