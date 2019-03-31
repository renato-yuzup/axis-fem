#include "init_element_bucket_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/domain/elements/ElementData.hpp"
#include "yuzu/domain/analyses/ReducedNumericalModel.hpp"
#include "yuzu/utils/kernel_utils.hpp"
#include "yuzu/common/gpu.hpp"

namespace afm = axis::foundation::memory;

namespace ay = axis::yuzu;
namespace ayda = axis::yuzu::domain::analyses;
namespace ayde = axis::yuzu::domain::elements;
namespace ayfm = axis::yuzu::foundation::memory;

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK)
InitElementBucketKernel(uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, ayfm::RelativePointer modelPtr, 
  size_type elementDataBlockSize )
{
  uint64 baseIndex = 
    ay::GetBaseThreadIndex(gridDim, blockIdx, blockDim, threadIdx);
  if (!ay::IsActiveThread(baseIndex + startIndex, numThreadsToUse)) return;
  ayda::ReducedNumericalModel& model =
    axis::yabsref<ayda::ReducedNumericalModel>(modelPtr);
  ayde::ElementData eData(baseMemoryAddressOnGPU, baseIndex, 
    elementDataBlockSize);
  uint64 elementId = eData.GetId();
  real *outputBucket = eData.GetOutputBuffer();
  model.SetElementOutputBucket((size_type)elementId, outputBucket);
}

void axis::application::executors::gpu::commands::kernels::
  RunInitElementBucketOnGPU( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, const axis::Dimension3D& gridDim, 
  const axis::Dimension3D& blockDim, void * streamPtr, 
  afm::RelativePointer& modelPtr, size_type elementDataBlockSize )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  cudaStream_t stream = (cudaStream_t)streamPtr;
  InitElementBucketKernel<<<grid, block, 0, stream>>>(numThreadsToUse, 
    startIndex, baseMemoryAddressOnGPU, 
    reinterpret_cast<ayfm::RelativePointer&>(modelPtr), elementDataBlockSize);
}

