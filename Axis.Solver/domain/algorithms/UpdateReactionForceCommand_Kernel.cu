#include "UpdateReactionForceCommand_Kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"

namespace adal = axis::domain::algorithms;
namespace afm = axis::foundation::memory;
namespace ayfb = axis::yuzu::foundation::blas;
namespace ayfm = axis::yuzu::foundation::memory;

__global__ void UpdateReactionForceKernel(ayfm::RelativePointer reactionForcePtr, 
  ayfm::RelativePointer externalLoadPtr, ayfm::RelativePointer internalLoadPtr, uint64 numThreads, 
  uint64 startIndex)
{
  uint64 blockItemCount = blockDim.x * blockDim.y;
  uint64 outerIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockItemCount;
  uint64 innerIndex = threadIdx.y * blockDim.x + threadIdx.x;
  uint64 index = startIndex + outerIndex + innerIndex;
  if (index < numThreads)
  {
    ayfb::ColumnVector& reactionForce = axis::yabsref<ayfb::ColumnVector>(reactionForcePtr);
    ayfb::ColumnVector& externalLoad = axis::yabsref<ayfb::ColumnVector>(externalLoadPtr);
    ayfb::ColumnVector& internalForce = axis::yabsref<ayfb::ColumnVector>(internalLoadPtr);
    reactionForce(index) = -externalLoad.GetElement(index) - internalForce.GetElement(index);
  }
}

extern void axis::domain::algorithms::UpdateReactionForceOnGPU( afm::RelativePointer& reactionForcePtr, 
  afm::RelativePointer& externalLoadPtr, afm::RelativePointer& internalLoadPtr, uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr )
{
  ayfm::RelativePointer gpuReaction = reinterpret_cast<ayfm::RelativePointer&>(reactionForcePtr);
  ayfm::RelativePointer gpuExternal = reinterpret_cast<ayfm::RelativePointer&>(externalLoadPtr);
  ayfm::RelativePointer gpuInternal = reinterpret_cast<ayfm::RelativePointer&>(internalLoadPtr);
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
#if defined(_DEBUG) || defined(DEBUG)
  cudaGetLastError(); // clear last error
#endif
  UpdateReactionForceKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(gpuReaction, gpuExternal, 
    gpuInternal, numThreadsToUse, startIndex);
}
