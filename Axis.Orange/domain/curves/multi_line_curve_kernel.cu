#include "multi_line_curve_kernel.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/memory/pointer.hpp"
#include "yuzu/domain/curves/CurveData.hpp"
#include "yuzu/common/gpu.hpp"
#include "yuzu/utils/kernel_utils.hpp"

#define X(x)  points[2*(x) + 0]
#define Y(x)  points[2*(x) + 1]

namespace ay = axis::yuzu;
namespace ayfm = axis::yuzu::foundation::memory;
namespace aydcu = axis::yuzu::domain::curves;

__global__ void __launch_bounds__(AXIS_YUZU_MAX_THREADS_PER_BLOCK) 
  UpdateCurveOnGPUKernel( uint64 numThreadsToUse, uint64 startIndex, 
  void *baseMemoryAddressOnGPU, real time )
{
  uint64 index = ay::GetThreadIndex(gridDim, blockIdx, blockDim, threadIdx, 
    startIndex);
  if (!ay::IsActiveThread(index, numThreadsToUse)) return;
  aydcu::CurveData curve(baseMemoryAddressOnGPU, index, 
    sizeof(ayfm::RelativePointer));
  ayfm::RelativePointer& dataPtr = 
    *(ayfm::RelativePointer *)curve.GetCurveData();
  real &outputBucket = *curve.GetOutputBucket();

  void *curveDataRegion = *dataPtr;
  uint64 numPoints = *(uint64 *)curveDataRegion;
  const real *points = (real *)((uint64)curveDataRegion + sizeof(uint64));
  for (size_t i = 1; i < numPoints; i++)
  {
    if ((X(i) > time) || (i == numPoints-1 && (abs(X(i) - time) <= 1e-15)))	
    {	
      // trivial case: horizontal line
      //       if (abs(Y(i-1) - Y(i)) <= 1e-15)
      //       {
      //         return Y(i);
      //       }
      real a = Y(i)   * (time - X(i-1));
      real b = Y(i-1) * (time - X(i));
      real c = 1.0 / (X(i) - X(i-1));
      a = (a - b) * c;
      outputBucket = a;
      return;
      //       ret urn (a - b) / c;
      // return (X(i)-X(i-1));
      //       return ((Y(i)-Y(i-1))) * (xCoord - X(i-1)) / (X(i)-X(i-1));
      //       return ((Y(i)-Y(i-1)) * (xCoord-X(i-1)) / (X(i)-X(i-1))) + Y(i-1);
    }
  }

  // consider last curve point
  outputBucket = Y(numPoints-1);
}

void axis::domain::curves::UpdateCurveOnGPU( uint64 numThreadsToUse, 
  uint64 startIndex, void *baseMemoryAddressOnGPU, 
  const axis::Dimension3D& gridDim, const axis::Dimension3D& blockDim, 
  void * streamPtr, real time )
{
  dim3 grid, block;
  grid.x = gridDim.X; grid.y = gridDim.Y; grid.z = gridDim.Z;
  block.x = blockDim.X; block.y = blockDim.Y; block.z = blockDim.Z;
  UpdateCurveOnGPUKernel<<<grid, block, 0, (cudaStream_t)streamPtr>>>(
    numThreadsToUse, startIndex, baseMemoryAddressOnGPU, time);
}
